"""Integrated brutal proof: unified FOV + detector/motion on same runtime path."""

from __future__ import annotations

import copy
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import pytest

import profiles
from profiles import (
    PROFILE_APEX_STYLE_LIVE_TRACE,
    effective_detection_fov_radius,
    effective_fov_radius,
)
from targeting_runtime import TargetingRuntime, resolve_runtime_fov

REPO_ROOT = Path(__file__).resolve().parents[1]
GIF_DIR = REPO_ROOT / "artifacts" / "real_apex_test" / "_gif_frames"
INPUTS_DIR = REPO_ROOT / "artifacts" / "real_apex_test" / "_inputs"


def _live_cfg() -> dict:
    cfg = copy.deepcopy(profiles.PROFILE_DEFAULTS[PROFILE_APEX_STYLE_LIVE_TRACE])
    cfg.update(
        {
            "body_shape_min_score": 0.42,
            "target_stickiness_pixels": 70,
            "detection_motion_assist": True,
            "detection_motion_threshold": 9,
            "humanoid_min_height_pixels": 60,
            "new_lock_confirm_frames": 1,
            "unified_fov": True,
        }
    )
    return cfg


class FovSingleSourceTests(unittest.TestCase):
    def test_unified_profile_sets_runtime_keys_equal(self) -> None:
        cfg = _live_cfg()
        user, det = resolve_runtime_fov(cfg, ads_active=True)
        self.assertEqual(user, det)
        self.assertEqual(cfg["_runtime_fov"], user)
        self.assertEqual(cfg["_runtime_detect_fov"], float(det))
        self.assertEqual(int(cfg["_runtime_overlay_fov"]), int(user * 0.96))

    def test_runtime_loop_no_split_when_unified(self) -> None:
        rt = (REPO_ROOT / "runtime.py").read_text(encoding="utf-8")
        self.assertIn("user_fov = effective_fov_radius", rt)
        self.assertIn("if bool(cfg.get(\"unified_fov\", True)):", rt)
        self.assertIn("detect_fov = user_fov", rt)
        self.assertIn("update_fov(", rt)
        self.assertNotIn("set_fov_center(center_x", rt)

    def test_split_only_when_opt_out(self) -> None:
        cfg = _live_cfg()
        cfg["unified_fov"] = False
        cfg["detection_fov_margin_pixels"] = 30
        user, det = resolve_runtime_fov(cfg, ads_active=True)
        self.assertGreater(det, user)


class DebugRingOptInTests(unittest.TestCase):
    def test_debug_second_ring_off_by_default(self) -> None:
        from config_validation import validate_config
        from assist import load_config

        cfg = validate_config(load_config(REPO_ROOT / "config.json"))
        self.assertFalse(cfg["debug_show_detect_ring"])

    def test_draw_debug_one_ring_by_default(self) -> None:
        from tests.test_overlay_one_visible_ring import DebugRingDefaultsTests

        DebugRingDefaultsTests().test_draw_debug_default_draws_single_ring()


try:
    import tkinter as _tk  # noqa: F401

    _TK = True
except ImportError:
    _TK = False


@unittest.skipUnless(_TK, "tkinter required")
class OverlayAdsSpamTests(unittest.TestCase):
    def _win(self):
        from overlay_window import OverlayWindow

        win = OverlayWindow(1920, 1080, 140, fov_center_x=960, fov_center_y=540)
        c = MagicMock()
        c.type.return_value = "oval"
        rings: list[int] = []
        n = [10]

        def mk(*_a, **kw):
            i = n[0]
            n[0] += 1
            if kw.get("tags") == ("fov_ring",):
                rings.append(i)
            return i

        c.create_oval.side_effect = mk
        c.find_withtag.side_effect = lambda t: tuple(rings) if t == "fov_ring" else ()
        c.find_all.side_effect = lambda: tuple(rings)
        win._canvas = c
        win._fov_id = 10
        win._target_id = 99
        win._drawn_fov_radius = 140
        win._drawn_ring_color = "#446644"
        return win, c, rings

    def test_ads_spam_never_more_than_one_fov_ring_tag(self) -> None:
        win, canvas, _rings = self._win()
        seq = [
            (140, False),
            (185, True),
            (185, True),
            (140, False),
            (140, False),
            (185, True),
            (185, True),
            (140, False),
        ]
        for r, ads in seq:
            win.update_fov(r, ads, 960.0, 540.0)
            win._redraw()
            tagged = win.count_fov_ring_items()
            self.assertLessEqual(
                tagged,
                1,
                f"ADS step r={r} ads={ads}: {tagged} fov_ring tags",
            )
        large_ovals = [
            c
            for c in canvas.create_oval.call_args_list
            if c.kwargs.get("tags") == ("fov_ring",)
        ]
        self.assertGreater(len(large_ovals), 0)

    def test_target_dot_not_deleted_by_purge(self) -> None:
        win, canvas, _ = self._win()
        canvas.gettags.return_value = ("target_dot",)
        win.update_fov(185, True, 960.0, 540.0)
        win._redraw()
        deleted_ids = [c.args[0] for c in canvas.delete.call_args_list]
        self.assertNotIn(99, deleted_ids)


@pytest.mark.skipif(
    not GIF_DIR.exists() or not any(GIF_DIR.glob("frame_*.png")),
    reason="GIF frames missing",
)
class IntegratedTrackingWithFovTests(unittest.TestCase):
    def test_gif_tracking_passes_hip_and_ads_fov(self) -> None:
        from tests.test_functional_runtime_proof import _load_bgr

        cfg_hip = _live_cfg()
        cfg_hip["_ads_active"] = False
        user_hip, det_hip = resolve_runtime_fov(cfg_hip, ads_active=False)

        cfg_ads = _live_cfg()
        cfg_ads["_ads_active"] = True
        user_ads, det_ads = resolve_runtime_fov(cfg_ads, ads_active=True)
        self.assertGreater(user_ads, user_hip)
        self.assertEqual(user_ads, det_ads)

        rt = TargetingRuntime()
        failures: list[str] = []
        for fp in sorted(GIF_DIR.glob("frame_*.png"))[:20]:
            img = cv2.imread(str(fp))
            if img is None:
                continue
            h, w = img.shape[:2]
            for ads, ufov in ((False, user_hip), (True, user_ads)):
                c = copy.deepcopy(_live_cfg())
                c["_ads_active"] = ads
                c["fov_center_x"] = w / 2.0
                c["fov_center_y"] = h / 2.0
                resolve_runtime_fov(c, ads_active=ads)
                aim = rt.process_frame(img, c, time_sec=0.0)
                rt.reset()
                if aim.target and not aim.is_stale:
                    if not aim.inside_body_overlay or not aim.inside_body_pull:
                        failures.append(f"{fp.name} ads={ads}")
                    if int(c["_runtime_fov"]) != ufov:
                        failures.append(
                            f"{fp.name} fov mismatch {c['_runtime_fov']} vs {ufov}"
                        )
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
