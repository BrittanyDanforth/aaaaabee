"""Phase-7 runtime fixes — unit-level tests.

Covers:
  * CRIT2: ``_smooth_aim(None)`` returns the cached anchor instead of
    hard-resetting when the lock has just expired.
  * HIGH4: ``_release_ads_inner`` uses ``soft_reset`` so the next
    observation step-caps from the prior anchor rather than teleporting.
  * MED9:  ``RuntimeController.apply_config_patch`` propagates the
    three previously-missing pull tuning keys
    (``magnetism_min_pull_scale``, ``fov_edge_min_pull_scale``,
    ``smoothing_curve``).
  * MED10: ``detector.draw_debug`` honours the live detection mode.
  * MED11: ``apply_config_patch`` flips the root logger level when
    ``verbose_logging`` changes.
"""

from __future__ import annotations

import logging
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

import detector
from runtime_controller import RuntimeController


# --- helpers -------------------------------------------------------------


def _make_runtime():
    """Build an AssistRuntime stub minimal enough to exercise _smooth_aim
    and _release_ads_inner without needing pynput/mss/tkinter."""
    from runtime import AssistRuntime
    from motion import TargetTracker
    import threading

    cfg = {
        "capture_fps": 30,
        "fov_radius_pixels": 140,
        "fov_radius_ads_pixels": 0,
        "fov_ads_scale": 1.3,
        "detection_motion_assist": True,
        "detection_motion_threshold": 10,
        "allow_live_mouse": False,
        "profile": "apex_style_dry_run",
        "ads_input_mode": "disabled",
        "target_process_name": "",
        "pause_on_target_closed": False,
        "kill_switch_key": "f10",
        "_runtime_detect_fov": 200,
    }
    runtime = AssistRuntime.__new__(AssistRuntime)
    runtime.config = cfg
    runtime.config_path = Path("/tmp/config.json")
    runtime._lock = threading.RLock()
    runtime._aim_tracker = TargetTracker()
    runtime._last_motion = None
    runtime._locked_target = None
    runtime._target_lost_frames = 0
    runtime._switch_candidate = None
    runtime._switch_frames = 0
    runtime._frame_cx = 400.0
    runtime._frame_cy = 225.0
    runtime._is_firing = False
    runtime._pull = None
    runtime._detect_ctx = detector.DetectionContext()
    return runtime


# --- CRIT2 ---------------------------------------------------------------


class SmoothAimNoneAnchorTests(unittest.TestCase):
    def test_smooth_aim_none_returns_cached_motion(self) -> None:
        rt = _make_runtime()
        # Observe once to populate _last_motion.
        from detector import Target
        t = Target(centroid_x=400, centroid_y=225, area=200, bbox_x=380,
                   bbox_y=190, bbox_w=40, bbox_h=70, body_shape_score=0.9)
        motion = rt._smooth_aim(t, 0.0, stale=False)
        self.assertIsNotNone(motion)
        # Now call with target=None — should NOT hard-reset; should
        # return the cached _last_motion.
        result = rt._smooth_aim(None, 0.033, stale=False)
        self.assertIs(result, motion)
        # Tracker still has the smoothed anchor preserved (CRIT2).
        self.assertIsNotNone(rt._aim_tracker._smooth_x)
        self.assertIsNotNone(rt._aim_tracker._smooth_y)

    def test_smooth_aim_none_with_no_prior_motion_hard_resets(self) -> None:
        rt = _make_runtime()
        result = rt._smooth_aim(None, 0.0, stale=False)
        self.assertIsNone(result)
        # Tracker is in fully-cleared state.
        self.assertIsNone(rt._aim_tracker._smooth_x)


# --- HIGH4 ---------------------------------------------------------------


class ReleaseAdsSoftResetTests(unittest.TestCase):
    def test_release_ads_inner_preserves_anchor(self) -> None:
        rt = _make_runtime()
        from detector import Target
        t = Target(centroid_x=400, centroid_y=225, area=200, bbox_x=380,
                   bbox_y=190, bbox_w=40, bbox_h=70, body_shape_score=0.9)
        rt._smooth_aim(t, 0.0, stale=False)
        smooth_x_before = rt._aim_tracker._smooth_x
        smooth_y_before = rt._aim_tracker._smooth_y
        last_meas_x_before = rt._aim_tracker._last_meas_x
        self.assertIsNotNone(smooth_x_before)
        rt._release_ads_inner()
        # _smooth_x/_smooth_y survive (soft_reset).
        self.assertEqual(rt._aim_tracker._smooth_x, smooth_x_before)
        self.assertEqual(rt._aim_tracker._smooth_y, smooth_y_before)
        self.assertEqual(rt._aim_tracker._last_meas_x, last_meas_x_before)
        # Lock state cleared.
        self.assertIsNone(rt._locked_target)
        self.assertIsNone(rt._last_motion)

    def test_release_ads_then_observe_step_caps_measurement(self) -> None:
        """After RMB release, the smoother's _last_meas_x anchor is
        preserved (soft_reset) so the next observation's measurement
        step is capped instead of teleporting through the smoother.
        (The final clamp to the new body bbox at output is unrelated
        — what matters is the SMOOTHER's measurement chain doesn't
        lose its anchor.)
        """
        rt = _make_runtime()
        from detector import Target
        t1 = Target(centroid_x=400, centroid_y=225, area=200, bbox_x=380,
                    bbox_y=190, bbox_w=40, bbox_h=70, body_shape_score=0.9)
        rt._smooth_aim(t1, 0.0, stale=False)
        # Snapshot the smoother anchor BEFORE release_ads.
        anchor_x = rt._aim_tracker._last_meas_x
        anchor_smooth_x = rt._aim_tracker._smooth_x
        self.assertEqual(anchor_x, 400.0)
        rt._release_ads_inner()
        # Anchor PRESERVED across the release.
        self.assertEqual(rt._aim_tracker._last_meas_x, anchor_x)
        self.assertEqual(rt._aim_tracker._smooth_x, anchor_smooth_x)
        # New observation 200 px away.
        t2 = Target(centroid_x=600, centroid_y=225, area=200, bbox_x=580,
                    bbox_y=190, bbox_w=40, bbox_h=70, body_shape_score=0.9)
        rt._smooth_aim(t2, 0.033, stale=False)
        # The measurement should have been step-capped from 400 by no
        # more than max(6, h*0.24)*1.0 px (16.8 px) instead of jumping
        # the full 200 px. _last_meas_x records what the smoother saw.
        self.assertLessEqual(
            rt._aim_tracker._last_meas_x, 420.0,
            f"step cap bypassed: _last_meas_x={rt._aim_tracker._last_meas_x} "
            f"(should be near 400+step_cap, not the raw 600)",
        )


# --- MED9 ----------------------------------------------------------------


class HotReloadCompletenessTests(unittest.TestCase):
    def test_update_tuning_propagates_new_keys(self) -> None:
        ctrl = RuntimeController({"profile": "apex_style_dry_run"}, Path("/tmp/x.json"))
        rt = MagicMock()
        rt.running = True
        pull = MagicMock()
        rt._pull = pull
        rt._aim_tracker = MagicMock()
        rt._detect_ctx = MagicMock(motion_assist=True, motion_threshold=10)
        ctrl._runtime = rt
        patch_dict = {
            "magnetism_min_pull_scale": 0.42,
            "fov_edge_min_pull_scale": 0.77,
            "smoothing_curve": "ease_in_out",
        }
        with patch.object(ctrl, "save_config"):
            ctrl.apply_config_patch(patch_dict)
        kwargs = pull.update_tuning.call_args.kwargs
        self.assertEqual(kwargs["magnetism_min_scale"], 0.42)
        self.assertEqual(kwargs["fov_edge_min_scale"], 0.77)
        self.assertEqual(kwargs["smoothing_curve"], "ease_in_out")


# --- MED10 ---------------------------------------------------------------


class DrawDebugLiveModeTests(unittest.TestCase):
    def test_draw_debug_uses_live_detection_mode(self) -> None:
        frame = np.zeros((100, 160, 3), dtype=np.uint8)
        captured = {}

        original = detector.build_detection_mask

        def spy(frame_bgr, hsv_ranges, **kwargs):
            captured["detection_mode"] = kwargs.get("detection_mode")
            return original(frame_bgr, hsv_ranges, **kwargs)

        with patch.object(detector, "build_detection_mask", side_effect=spy):
            detector.draw_debug(
                frame, None, 30, 80, 50,
                detection_mode=detector.DETECTION_MODE_APEX,
            )
        self.assertEqual(captured["detection_mode"], detector.DETECTION_MODE_APEX)

    def test_draw_debug_default_falls_back_to_shape(self) -> None:
        frame = np.zeros((100, 160, 3), dtype=np.uint8)
        captured = {}

        original = detector.build_detection_mask

        def spy(frame_bgr, hsv_ranges, **kwargs):
            captured["detection_mode"] = kwargs.get("detection_mode")
            return original(frame_bgr, hsv_ranges, **kwargs)

        with patch.object(detector, "build_detection_mask", side_effect=spy):
            detector.draw_debug(frame, None, 30, 80, 50)
        self.assertEqual(captured["detection_mode"], detector.DETECTION_MODE_SHAPE)


# --- MED11 ---------------------------------------------------------------


class VerboseLoggingHotReloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root_level = logging.getLogger().level
        self.aba_level = logging.getLogger("aba").level

    def tearDown(self) -> None:
        logging.getLogger().setLevel(self.root_level)
        logging.getLogger("aba").setLevel(self.aba_level)

    def test_verbose_logging_patch_flips_root_level(self) -> None:
        ctrl = RuntimeController({"profile": "apex_style_dry_run"}, Path("/tmp/x.json"))
        with patch.object(ctrl, "save_config"):
            ctrl.apply_config_patch({"verbose_logging": True})
        self.assertEqual(logging.getLogger().level, logging.DEBUG)
        with patch.object(ctrl, "save_config"):
            ctrl.apply_config_patch({"verbose_logging": False})
        self.assertEqual(logging.getLogger().level, logging.INFO)




class ConfigPatchValidationTests(unittest.TestCase):
    def test_invalid_patch_is_rejected_and_not_persisted(self) -> None:
        ctrl = RuntimeController({"profile": "apex_style_dry_run"}, Path("/tmp/x.json"))
        before = dict(ctrl._config)
        with patch.object(ctrl, "save_config") as save_mock:
            with self.assertRaises(Exception):
                ctrl.apply_config_patch({"pull_strength": -1.0})
        self.assertEqual(ctrl._config, before)
        save_mock.assert_not_called()


class RuntimeFovUpdateApiTests(unittest.TestCase):
    def test_runtime_uses_pull_api_not_private_tuning_write(self) -> None:
        text = (Path(__file__).resolve().parents[1] / "runtime.py").read_text(encoding="utf-8")
        self.assertIn("set_runtime_fov_radius", text)
        self.assertNotIn("_pull._tuning.fov_radius", text)



class StaleTargetSafetyTests(unittest.TestCase):
    def test_smooth_aim_none_returns_last_motion_once(self) -> None:
        rt = _make_runtime()
        from detector import Target
        t = Target(centroid_x=400, centroid_y=230, area=200, bbox_x=380, bbox_y=190, bbox_w=40, bbox_h=70, body_shape_score=0.9)
        m1 = rt._smooth_aim(t, 0.0, stale=False)
        self.assertIsNotNone(m1)
        m2 = rt._smooth_aim(None, 0.033, stale=False)
        self.assertIsNotNone(m2)
        m3 = rt._smooth_aim(None, 0.066, stale=False)
        self.assertIsNone(m3)

    def test_runtime_disables_pull_after_second_stale_frame(self) -> None:
        text = (Path(__file__).resolve().parents[1] / "runtime.py").read_text(encoding="utf-8")
        self.assertIn("if stale_det and self._target_lost_frames >= 2", text)
        self.assertIn("pull_target = None", text)



class PullTargetSafetyGateTests(unittest.TestCase):
    def test_blocks_large_jump_without_body_upgrade(self) -> None:
        rt = _make_runtime()
        from detector import Target
        rt.config["_runtime_detect_fov"] = 200
        rt._frame_cy = 225
        a = Target(centroid_x=400, centroid_y=230, area=200, bbox_x=380, bbox_y=190, bbox_w=40, bbox_h=70, body_shape_score=0.90)
        b = Target(centroid_x=480, centroid_y=120, area=120, bbox_x=460, bbox_y=80, bbox_w=30, bbox_h=40, body_shape_score=0.92)
        self.assertTrue(rt._allow_pull_target(a, stale_det=False))
        self.assertFalse(rt._allow_pull_target(b, stale_det=False))

    def test_blocks_stale_after_first_lost_frame(self) -> None:
        rt = _make_runtime()
        from detector import Target
        rt._frame_cy = 225
        t = Target(centroid_x=400, centroid_y=230, area=200, bbox_x=380, bbox_y=190, bbox_w=40, bbox_h=70, body_shape_score=0.90)
        rt._target_lost_frames = 1
        self.assertFalse(rt._allow_pull_target(t, stale_det=True))


    def test_large_jump_requires_two_stable_frames(self) -> None:
        rt = _make_runtime()
        from detector import Target
        rt.config["_runtime_detect_fov"] = 200
        rt._frame_cy = 225
        a = Target(centroid_x=400, centroid_y=230, area=200, bbox_x=380, bbox_y=190, bbox_w=40, bbox_h=70, body_shape_score=0.90)
        j1 = Target(centroid_x=470, centroid_y=220, area=220, bbox_x=450, bbox_y=185, bbox_w=44, bbox_h=72, body_shape_score=1.00)
        j2 = Target(centroid_x=472, centroid_y=221, area=222, bbox_x=452, bbox_y=186, bbox_w=43, bbox_h=71, body_shape_score=1.00)
        self.assertTrue(rt._allow_pull_target(a, stale_det=False))
        self.assertFalse(rt._allow_pull_target(j1, stale_det=False))
        self.assertTrue(rt._allow_pull_target(j2, stale_det=False))

# --- Dead code removal (MED7) -------------------------------------------


class DeadCodeRemovalTests(unittest.TestCase):
    def test_get_last_debug_lines_removed(self) -> None:
        self.assertFalse(hasattr(detector, "get_last_debug_lines"))

    def test_is_humanoid_contour_removed(self) -> None:
        self.assertFalse(hasattr(detector, "_is_humanoid_contour"))

    def test_last_debug_lines_global_removed(self) -> None:
        self.assertFalse(hasattr(detector, "_LAST_DEBUG_LINES"))

    def test_target_window_title_not_validated(self) -> None:
        from profiles import apply_profile
        from config_validation import validate_config
        cfg = validate_config(apply_profile({"profile": "apex_style_dry_run"}))
        self.assertNotIn("target_window_title", cfg)


if __name__ == "__main__":
    unittest.main()
