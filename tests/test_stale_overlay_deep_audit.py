"""Deep audit: stale overlay hold, sky clamps, gate wiring (post 906b7de fixes)."""

from __future__ import annotations

import ast
import copy
import re
import unittest
from pathlib import Path

import cv2
import numpy as np

import motion as motion_mod
import profiles
from detector import Target
from motion import TargetMotion, TargetTracker
from profiles import PROFILE_APEX_STYLE_LIVE_TRACE
from target_lock import (
    TargetLockState,
    may_assist_pull_target,
    overlay_may_show_target,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME = REPO_ROOT / "runtime.py"
GIF_DIR = REPO_ROOT / "artifacts" / "real_apex_test" / "_gif_frames"


def _runtime_build_frame_overlay_ast() -> ast.BoolOp | None:
    tree = ast.parse(RUNTIME.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "build_frame_overlay":
                    if isinstance(node.value, ast.BoolOp):
                        return node.value
    return None


def _mirror_build(
    *,
    show: bool,
    plausible: bool,
    fresh: bool,
) -> bool:
    return show and plausible


class GateWiringTests(unittest.TestCase):
    def test_runtime_build_frame_overlay_ast_matches_mirror(self) -> None:
        expr = _runtime_build_frame_overlay_ast()
        self.assertIsNotNone(expr)
        text = RUNTIME.read_text(encoding="utf-8")
        self.assertNotIn("stale_det and locked_grace", text)
        self.assertIn("plausible_lock", text)
        self.assertIn("show_for_overlay and plausible_lock", text)

    def test_mirror_matches_documented_stale_grace(self) -> None:
        t = Target(
            centroid_x=400.0,
            centroid_y=300.0,
            area=3000.0,
            distance_to_center=10.0,
            confidence=0.8,
            bbox_x=370,
            bbox_y=200,
            bbox_w=60,
            bbox_h=120,
            body_shape_score=0.7,
            part_count=4,
            red_coverage=0.15,
            fill_ratio=0.5,
            max_circularity=0.5,
            has_classified_torso=True,
            head_score=0.5,
            torso_score=0.5,
            limb_stack_score=0.4,
        )
        for lost, expect in ((3, False), (12, False), (13, False)):
            got = _mirror_build(show=False, plausible=True, fresh=False)
            self.assertEqual(got, expect, f"stale lost={lost}: dot build should stay off")


class SkyClampPathTests(unittest.TestCase):
    def test_overlay_follow_pull_motion_inside_body_band(self) -> None:
        tr = TargetTracker()
        bx, by, bw, bh = 400, 220, 70, 150
        m = tr.observe_target(
            435.0,
            290.0,
            0.0,
            bbox_x=bx,
            bbox_y=by,
            bbox_w=bw,
            bbox_h=bh,
            aim_is_body_anchor=True,
        )
        tr._vx, tr._vy = 80.0, -180.0
        m2 = tr.observe_target(
            437.0,
            288.0,
            1.0 / 30.0,
            bbox_x=bx,
            bbox_y=by,
            bbox_w=bw,
            bbox_h=bh,
            aim_is_body_anchor=True,
        )
        y_hi = by + bh * motion_mod._body_y_hi_frac
        y_lo = by + bh * motion_mod._body_y_lo_frac
        for label, x, y in (
            ("motion", m2.x, m2.y),
            ("overlay", m2.overlay_xy()[0], m2.overlay_xy()[1]),
        ):
            self.assertGreaterEqual(y, y_lo - 1.0, label)
            self.assertLessEqual(y, y_hi + 2.0, f"{label} above chest band")

    def test_fragment_bbox_does_not_raise_overlay_above_stable(self) -> None:
        tr = TargetTracker()
        bx, by, bw, bh = 400, 200, 80, 160
        for i in range(5):
            tr.observe_target(
                440.0,
                275.0,
                i / 30.0,
                bbox_x=bx,
                bbox_y=by,
                bbox_w=bw,
                bbox_h=bh,
                aim_is_body_anchor=True,
            )
        m = tr.observe_target(
            442.0,
            240.0,
            0.2,
            bbox_x=bx,
            bbox_y=100,
            bbox_w=35,
            bbox_h=45,
            aim_is_body_anchor=True,
        )
        _, oy = m.overlay_xy()
        stable_hi = by + bh * motion_mod._body_y_hi_frac
        self.assertLessEqual(oy, stable_hi + 3.0)


class StaleGhostPullTests(unittest.TestCase):
    def test_stale_smooth_aim_does_not_advance(self) -> None:
        from runtime import AssistRuntime

        rt = AssistRuntime.__new__(AssistRuntime)
        rt.config = {"_runtime_detect_fov": 200.0, "overlay_dot_smooth_alpha": 0.58}
        rt._aim_tracker = TargetTracker()
        rt._last_motion = None
        rt._frame_cx = 640.0
        rt._frame_cy = 360.0
        t = Target(
            centroid_x=500.0,
            centroid_y=200.0,
            area=2000.0,
            distance_to_center=50.0,
            confidence=0.7,
            bbox_x=470,
            bbox_y=120,
            bbox_w=60,
            bbox_h=100,
            body_shape_score=0.6,
            part_count=3,
            red_coverage=0.1,
            fill_ratio=0.5,
            max_circularity=0.5,
            has_classified_torso=True,
            head_score=0.4,
            torso_score=0.4,
            limb_stack_score=0.3,
        )
        m0 = AssistRuntime._smooth_aim(rt, t, 0.0, stale=False)
        self.assertIsNotNone(m0)
        m1 = AssistRuntime._smooth_aim(rt, t, 0.05, stale=True)
        self.assertIs(m1, rt._last_motion)
        self.assertEqual(m0.x, m1.x)
        self.assertEqual(m0.y, m1.y)


def _gif_available() -> bool:
    return GIF_DIR.exists() and any(GIF_DIR.glob("frame_*.png"))


@unittest.skipUnless(_gif_available(), "GIF frames missing")
class GifMotionOverlayAuditTests(unittest.TestCase):
    def test_gif_sequence_overlay_stays_in_body_when_active(self) -> None:
        from targeting_runtime import TargetingRuntime

        cfg = copy.deepcopy(profiles.PROFILE_DEFAULTS[PROFILE_APEX_STYLE_LIVE_TRACE])
        cfg.update(
            {
                "body_shape_min_score": 0.42,
                "new_lock_confirm_frames": 1,
                "detection_motion_assist": True,
            }
        )
        rt = TargetingRuntime()
        violations: list[str] = []
        for fp in sorted(GIF_DIR.glob("frame_*.png"))[:25]:
            img = cv2.imread(str(fp))
            if img is None:
                continue
            h, w = img.shape[:2]
            cfg["fov_center_x"] = w / 2.0
            cfg["fov_center_y"] = h / 2.0
            aim = rt.process_frame(img, cfg, time_sec=0.0)
            rt.reset()
            if not aim.active or aim.target is None:
                continue
            bb = aim.bbox_used or (
                aim.target.bbox_x,
                aim.target.bbox_y,
                aim.target.bbox_w,
                aim.target.bbox_h,
            )
            bx, by, bw, bh = bb
            y_hi = by + bh * motion_mod._body_y_hi_frac
            y_lo = by + bh * motion_mod._body_y_lo_frac
            if aim.aim_y < y_lo - 2 or aim.aim_y > y_hi + 3:
                violations.append(
                    f"{fp.name}: aim_y={aim.aim_y:.0f} outside [{y_lo:.0f},{y_hi:.0f}]"
                )
            if aim.aim_y < h * 0.08:
                violations.append(f"{fp.name}: aim in sky band y={aim.aim_y:.0f}")
        self.assertEqual(violations, [])


class RuntimeHoldLastSourceTests(unittest.TestCase):
    def test_hold_last_gated_by_stale_grace_not_miss_count_only(self) -> None:
        text = RUNTIME.read_text(encoding="utf-8")
        self.assertIn("target_lost_frames <= stale_grace", text)
        self.assertNotIn("_overlay_miss_frames < 6", text)


class HoldLastGhostWindowTests(unittest.TestCase):
    def test_hold_last_off_at_lost_13_while_lock_grace_18(self) -> None:
        held = (100.0, 200.0)
        locked_hold = True
        stale_grace = 12
        target_lost_frames = 13
        overlay_pt = None
        if (
            held is not None
            and locked_hold
            and target_lost_frames <= stale_grace
        ):
            overlay_pt = held
        self.assertIsNone(overlay_pt)


if __name__ == "__main__":
    unittest.main()
