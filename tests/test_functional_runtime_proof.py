"""Brutal functional proof: live-path behavior, not trace-only success.

Exercises ``TargetingRuntime`` (same lock + stale freeze as ``AssistRuntime``)
on real GIF frames and img1–img7 fixtures. Asserts overlay AND pull stay in
body bbox, stale skips ``observe_target``, presets reach live subsystems.
"""

from __future__ import annotations

import copy
import math
import re
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

import detector
import motion as motion_mod
import profiles
from detector import Target, is_upward_fragment_vs_locked
from motion import TargetTracker
from runtime_controller import RuntimeController
from target_lock import TargetLockState, apply_target_lock, detection_sticky_context
from targeting_runtime import TargetingRuntime

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUTS_DIR = REPO_ROOT / "artifacts" / "real_apex_test" / "_inputs"
GIF_DIR = REPO_ROOT / "artifacts" / "real_apex_test" / "_gif_frames"


def _tracking_preset() -> dict:
    """Tracking = Real Apex preset (from aba_gui.TUNING_PRESETS)."""
    text = (REPO_ROOT / "aba_gui.py").read_text(encoding="utf-8")
    m = re.search(
        r'"Tracking":\s*\{([^}]+(?:\{[^}]*\}[^}]*)*)\}',
        text,
        re.DOTALL,
    )
    assert m, "Tracking preset block missing from aba_gui.py"
    block = "{" + m.group(1) + "}"
    # Safe literal eval of dict literals only
    preset = {}
    for key, val in re.findall(r'"([^"]+)":\s*([^,\n]+)', block):
        val = val.strip()
        if val in ("True", "False"):
            preset[key] = val == "True"
        elif val.startswith('"'):
            preset[key] = val.strip('"')
        else:
            preset[key] = float(val) if "." in val else int(val)
    return preset


def _live_cfg() -> dict:
    cfg = copy.deepcopy(
        profiles.PROFILE_DEFAULTS[profiles.PROFILE_APEX_STYLE_LIVE_TRACE]
    )
    cfg.update(_tracking_preset())
    cfg["new_lock_confirm_frames"] = 1
    return cfg


def _load_bgr(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is not None:
        return img
    if path.suffix.lower() == ".webp":
        try:
            from PIL import Image

            rgb = np.array(Image.open(path).convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception:
            return None
    return None


def _gif_available() -> bool:
    return GIF_DIR.exists() and any(GIF_DIR.glob("frame_*.png"))


@pytest.fixture
def cfg() -> dict:
    return _live_cfg()


class FunctionalRuntimeProofTests(unittest.TestCase):
    def test_runtime_main_loop_wires_find_best_and_lock(self) -> None:
        src = (REPO_ROOT / "runtime.py").read_text(encoding="utf-8")
        self.assertIn("find_best_target(", src)
        self.assertIn("apply_target_lock(", src)
        self.assertIn("if stale:", src)
        self.assertIn("return self._last_motion", src)
        self.assertNotIn("find_best_target_legacy", src)

    def test_targeting_runtime_stale_skips_observe(self) -> None:
        rt = TargetingRuntime()
        cfg = _live_cfg()
        frame = np.zeros((400, 640, 3), dtype=np.uint8)
        # Lock a synthetic body in center
        t = Target(
            centroid_x=320.0,
            centroid_y=200.0,
            area=12800.0,
            bbox_x=280,
            bbox_y=120,
            bbox_w=80,
            bbox_h=160,
            confidence=0.9,
            body_shape_score=0.85,
            red_coverage=0.4,
            torso_score=0.7,
            limb_stack_score=0.6,
            part_count=3,
        )
        rt._lock_state.locked_target = t
        rt._lock_state.target_lost_frames = 2
        rt.tracker.observe_target(
            t.centroid_x, t.centroid_y, 0.0,
            bbox_x=t.bbox_x, bbox_y=t.bbox_y, bbox_w=t.bbox_w, bbox_h=t.bbox_h,
            aim_is_body_anchor=True,
        )
        before = rt.observe_target_call_count
        with unittest.mock.patch.object(
            detector, "find_best_target", return_value=detector.DetectionResult(None, [], 0.0)
        ):
            aim = rt.process_frame(frame, cfg, time_sec=1.0)
        self.assertTrue(aim.is_stale)
        self.assertEqual(before, rt.observe_target_call_count)
        self.assertFalse(aim.observe_called)

    def test_tracking_preset_hot_reload_reaches_tracker(self) -> None:
        preset = _tracking_preset()
        ctrl = RuntimeController(
            {"profile": profiles.PROFILE_APEX_STYLE_LIVE_TRACE},
            Path("/tmp/aba_func_test.json"),
        )
        rt = MagicMock()
        rt.running = True
        tracker = TargetTracker()
        rt._aim_tracker = tracker
        rt._pull = MagicMock()
        rt._detect_ctx = detector.DetectionContext()
        ctrl._runtime = rt
        with unittest.mock.patch.object(ctrl, "save_config"):
            merged = ctrl.apply_config_patch(preset, persist=False)
        self.assertAlmostEqual(float(merged["pull_strength"]), 0.95)
        self.assertAlmostEqual(float(merged["smoothing_tau_still"]), 0.030)
        self.assertAlmostEqual(float(merged["humanoid_min_height_pixels"]), 60.0)
        pull = rt._pull.update_tuning
        self.assertTrue(pull.called)
        self.assertAlmostEqual(
            pull.call_args.kwargs["pull_strength"], 0.95, places=3
        )

    def test_fragment_switch_blocked_in_lock(self) -> None:
        locked = Target(
            centroid_x=400.0,
            centroid_y=300.0,
            area=18000.0,
            bbox_x=360,
            bbox_y=180,
            bbox_w=90,
            bbox_h=200,
            confidence=0.9,
            body_shape_score=0.8,
            red_coverage=0.35,
            torso_score=0.75,
            limb_stack_score=0.7,
            part_count=4,
        )
        frag = Target(
            centroid_x=402.0,
            centroid_y=195.0,
            area=2750.0,
            bbox_x=375,
            bbox_y=120,
            bbox_w=50,
            bbox_h=55,
            confidence=0.88,
            body_shape_score=0.85,
            red_coverage=0.4,
            torso_score=0.8,
            limb_stack_score=0.5,
            part_count=2,
        )
        self.assertTrue(is_upward_fragment_vs_locked(locked, frag))
        state = TargetLockState()
        state.locked_target = locked
        cfg = _live_cfg()
        raw = detector.DetectionResult(frag, [frag], frag.confidence)
        out, stale = apply_target_lock(
            state, raw, center_y=360.0, cfg=cfg, fov_cx=400.0, fov_cy=360.0
        )
        self.assertIs(out.target, locked)
        self.assertTrue(stale or out.target is locked)


@pytest.mark.parametrize(
    "filename,expect_active",
    [
        ("img1_back_view.png", True),
        ("img2_shooting_26.png", True),
        ("img3_side_view.webp", True),
        ("img5_close_ads.webp", True),
        ("img6_seven_characters.png", True),
        ("img7_sky_dot_bug.png", False),
    ],
)
def test_real_image_targeting_runtime(
    filename: str, expect_active: bool, cfg: dict
) -> None:
    path = INPUTS_DIR / filename
    if not path.exists():
        pytest.skip(f"{path} missing")
    img = _load_bgr(path)
    assert img is not None
    h, w = img.shape[:2]
    cfg = copy.deepcopy(cfg)
    cfg["fov_center_x"] = w / 2.0
    cfg["fov_center_y"] = h / 2.0
    if filename.startswith("img6"):
        cfg["fov_radius_pixels"] = int(0.95 * max(w, h) / 2.0)
    rt = TargetingRuntime()
    aim = rt.process_frame(img, cfg, time_sec=0.0)
    assert aim.active == expect_active, (
        f"{filename}: active={aim.active} expected={expect_active}"
    )
    if expect_active and aim.target is not None:
        assert aim.inside_body_overlay, (
            f"{filename}: overlay outside motion body band "
            f"ov=({aim.overlay_x:.0f},{aim.overlay_y:.0f})"
        )
        assert aim.inside_body_pull, (
            f"{filename}: pull outside motion body band "
            f"pull=({aim.pull_x:.0f},{aim.pull_y:.0f})"
        )
        bb = rt.tracker._body_bbox
        if bb is not None:
            y_hi = bb[1] + bb[3] * motion_mod._body_y_hi_frac
            assert aim.overlay_y <= y_hi + 3.0, f"{filename}: overlay above chest band"
            assert aim.pull_y <= y_hi + 3.0, f"{filename}: pull above chest band"
        assert aim.target.bbox_y > h * 0.05, f"{filename}: bbox in sky band"


@unittest.skipUnless(_gif_available(), "GIF frames not in artifacts")
class GifFunctionalProofTests(unittest.TestCase):
    def test_full_gif_overlay_and_pull_inside_body(self) -> None:
        cfg = _live_cfg()
        rt = TargetingRuntime()
        ctx = rt._detect_ctx
        cfg["detection_motion_assist"] = True
        ctx.motion_assist = True
        ctx.motion_threshold = int(cfg["detection_motion_threshold"])
        failures: list[str] = []
        stale_rise: list[str] = []
        prev_oy: float | None = None
        frames = sorted(GIF_DIR.glob("frame_*.png"))
        t_sec = 0.0
        for fp in frames:
            img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]
            cfg["fov_center_x"] = w / 2.0
            cfg["fov_center_y"] = h / 2.0
            aim = rt.process_frame(img, cfg, time_sec=t_sec)
            t_sec += 1.0 / 30.0
            idx = fp.stem
            if aim.target is not None and not aim.is_stale:
                if not aim.inside_body_overlay:
                    failures.append(f"{idx}: overlay outside motion body band")
                if not aim.inside_body_pull:
                    failures.append(f"{idx}: pull outside motion body band")
                bb = rt.tracker._body_bbox
                if bb is not None:
                    y_hi = bb[1] + bb[3] * motion_mod._body_y_hi_frac
                    if aim.overlay_y > y_hi + 3.0:
                        failures.append(f"{idx}: overlay_y>{y_hi:.0f}")
            if aim.is_stale and prev_oy is not None and aim.overlay_y < prev_oy - 8.0:
                stale_rise.append(f"{idx}: overlay dropped {prev_oy:.0f}->{aim.overlay_y:.0f}")
            if aim.overlay_y > 0:
                prev_oy = aim.overlay_y
        self.assertEqual(failures, [], "\n  ".join(failures))
        self.assertEqual(stale_rise, [], "\n  ".join(stale_rise))

    def test_gif_stale_hold_does_not_increase_observe_calls(self) -> None:
        cfg = _live_cfg()
        rt = TargetingRuntime()
        frames = sorted(GIF_DIR.glob("frame_*.png"))[:45]
        obs_per_frame: list[tuple[str, bool, int]] = []
        t_sec = 0.0
        for fp in frames:
            img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            cfg["fov_center_x"] = w / 2.0
            cfg["fov_center_y"] = h / 2.0
            n_before = rt.observe_target_call_count
            aim = rt.process_frame(img, cfg, time_sec=t_sec)
            t_sec += 1.0 / 30.0
            delta = rt.observe_target_call_count - n_before
            obs_per_frame.append((fp.stem, aim.is_stale, delta))
            if aim.is_stale:
                self.assertEqual(delta, 0, f"{fp.stem}: observe during stale")
        stale_frames = sum(1 for _, st, _ in obs_per_frame if st)
        self.assertGreater(stale_frames, 0, "GIF replay produced no stale frames")


class SyntheticBreakTests(unittest.TestCase):
    """Try to break sky clamp / fragment guard with adversarial motion."""

    def test_fast_upward_velocity_does_not_climb_overlay(self) -> None:
        tr = TargetTracker()
        bx, by, bw, bh = 300, 200, 70, 150
        for i in range(12):
            tr._vx = 80.0
            tr._vy = -180.0
            m = tr.observe_target(
                335.0 + i * 2,
                260.0 - i * 0.5,
                i / 60.0,
                bbox_x=bx,
                bbox_y=by,
                bbox_w=bw,
                bbox_h=bh,
                aim_is_body_anchor=True,
            )
        y_hi = by + bh * motion_mod._body_y_hi_frac
        ox, oy = m.overlay_xy()
        self.assertLessEqual(oy, y_hi + 2.0)
        self.assertLessEqual(m.y, y_hi + 2.0)

    def test_side_motion_stays_in_bbox(self) -> None:
        tr = TargetTracker()
        bx, by, bw, bh = 200, 150, 60, 140
        for i in range(15):
            m = tr.observe_target(
                230.0 + i * 6,
                200.0,
                i / 60.0,
                bbox_x=bx,
                bbox_y=by,
                bbox_w=bw,
                bbox_h=bh,
                aim_is_body_anchor=True,
            )
        self.assertTrue(
            TargetTracker.point_inside_body_bbox(m.x, m.y, bx, by, bw, bh)
        )
        ox, oy = m.overlay_xy()
        self.assertTrue(
            TargetTracker.point_inside_body_bbox(ox, oy, bx, by, bw, bh)
        )


if __name__ == "__main__":
    unittest.main()
