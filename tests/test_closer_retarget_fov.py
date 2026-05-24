"""Frame-45 class bug: closer in-FOV dummy must beat stale rim fragment lock."""

from __future__ import annotations

import copy
import unittest
from pathlib import Path

import cv2

import detector
import profiles
from detector import DETECTION_MODE_APEX
from profiles import (
    PROFILE_APEX_STYLE_LIVE_TRACE,
    effective_detection_fov_radius,
    effective_overlay_fov_radius,
)
from target_lock import TargetLockState, apply_target_lock, detection_sticky_context
from targeting_runtime import TargetingRuntime, resolve_runtime_fov

GIF = Path(__file__).resolve().parents[1] / "artifacts/real_apex_test/_gif_frames_all"


def _cfg() -> dict:
    c = copy.deepcopy(profiles.PROFILE_DEFAULTS[PROFILE_APEX_STYLE_LIVE_TRACE])
    c["_ads_active"] = True
    c["new_lock_confirm_frames"] = 1
    return c


@unittest.skipUnless(
    (GIF / "frame_0045.png").exists(),
    "need extracted GIF frame_0045",
)
class Frame45CloserRetargetTests(unittest.TestCase):
    def test_center_dummy_accepted_after_close_fov_gate(self) -> None:
        img = cv2.imread(str(GIF / "frame_0045.png"))
        self.assertIsNotNone(img)
        h, w = img.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        cfg = _cfg()
        det_fov = effective_detection_fov_radius(cfg, ads_active=True)
        ctx = detector.DetectionContext()
        cands, _, _ = detector.enumerate_candidates(
            img,
            cfg["hsv_ranges"],
            det_fov,
            float(cfg["min_target_area_pixels"]),
            cx,
            cy,
            detection_mode=DETECTION_MODE_APEX,
            context=ctx,
            body_shape_min_score=0.42,
        )
        accepted = [c for c in cands if c.accepted]
        near = [
            c
            for c in accepted
            if c.distance_to_center < det_fov * 0.35 and c.bbox_h >= 70
        ]
        self.assertGreaterEqual(
            len(near),
            1,
            "close in-FOV humanoid should be accepted (was no_torso)",
        )

    def test_retarget_beats_stale_rim_lock_on_frame_45_sequence(self) -> None:
        """Full runtime replay: hand off to center dummy on frame 45 (1:1 with live)."""
        cfg = _cfg()
        rt = TargetingRuntime()
        h = w = 0
        for fi in range(46):
            p = GIF / f"frame_{fi:04d}.png"
            img = cv2.imread(str(p))
            if img is None:
                continue
            h, w = img.shape[:2]
            cfg["fov_center_x"] = w / 2.0
            cfg["fov_center_y"] = h / 2.0
            aim = rt.process_frame(img, cfg, time_sec=fi / 30.0)

        t = aim.target or rt.lock_state.locked_target
        self.assertIsNotNone(t, "frame 45 should hold center dummy lock")
        assert t is not None
        dist = ((t.centroid_x - w / 2) ** 2 + (t.centroid_y - h / 2) ** 2) ** 0.5
        ov = effective_overlay_fov_radius(cfg)
        self.assertLess(
            dist,
            ov * 0.55,
            f"frame 45 should lock center dummy not rim fragment, dist={dist:.0f}",
        )
        self.assertGreaterEqual(t.bbox_h, 70, "should be full-size close dummy")
        self.assertTrue(aim.active, "overlay should be LIVE on fresh center detect")

    def test_close_dummy_bbox_not_shifted_above_body(self) -> None:
        """Debug bbox must sit on the body column, not a HUD fringe above it."""
        img = cv2.imread(str(GIF / "frame_0045.png"))
        self.assertIsNotNone(img)
        h, w = img.shape[:2]
        cfg = _cfg()
        resolve_runtime_fov(cfg, ads_active=True)
        det = int(cfg["_runtime_detect_fov"])
        ctx = detector.DetectionContext()
        cands, _, _ = detector.enumerate_candidates(
            img,
            cfg["hsv_ranges"],
            det,
            float(cfg["min_target_area_pixels"]),
            w / 2,
            h / 2,
            detection_mode=DETECTION_MODE_APEX,
            context=ctx,
            body_shape_min_score=0.42,
        )
        center = next(
            c
            for c in cands
            if c.accepted and c.distance_to_center < 80 and c.bbox_h >= 48
        )
        foot_y = center.bbox_y + center.bbox_h
        self.assertGreater(foot_y, h * 0.28, f"bbox foot too high: {foot_y}")
        self.assertLess(center.bbox_y, h * 0.42, f"bbox top too low on screen: {center.bbox_y}")
        self.assertGreaterEqual(center.bbox_h, 48)

    def test_sticky_pool_retarget_on_frame_45(self) -> None:
        """Rim fragment in sticky pool must lose to closer center humanoid."""
        cfg = _cfg()
        rt = TargetingRuntime()
        for fi in range(45):
            p = GIF / f"frame_{fi:04d}.png"
            im = cv2.imread(str(p))
            self.assertIsNotNone(im)
            hh, ww = im.shape[:2]
            cfg["fov_center_x"] = ww / 2.0
            cfg["fov_center_y"] = hh / 2.0
            rt.process_frame(im, cfg, time_sec=fi / 30.0)

        img = cv2.imread(str(GIF / "frame_0045.png"))
        self.assertIsNotNone(img)
        h, w = img.shape[:2]
        cfg["fov_center_x"] = w / 2.0
        cfg["fov_center_y"] = h / 2.0
        state = rt.lock_state
        ctx = rt._detect_ctx
        resolve_runtime_fov(cfg, ads_active=True)
        det = int(cfg["_runtime_detect_fov"])
        ov = float(cfg["_runtime_overlay_fov"])
        sticky, locked, _ = detection_sticky_context(state, cfg)
        self.assertTrue(locked)
        raw = detector.find_best_target(
            img,
            cfg["hsv_ranges"],
            det,
            float(cfg["min_target_area_pixels"]),
            w / 2,
            h / 2,
            detection_mode=DETECTION_MODE_APEX,
            context=ctx,
            sticky_target=sticky,
            currently_locked=locked,
            min_height_px=float(cfg["humanoid_min_height_pixels"]),
            display_fov_radius=ov,
            stickiness_pixels=float(cfg["target_stickiness_pixels"]),
        )
        self.assertTrue(raw.active)
        self.assertIsNotNone(raw.target)
        t = raw.target
        assert t is not None
        self.assertLess(t.distance_to_center, 80.0)
        self.assertGreaterEqual(t.bbox_h, 70)
        self.assertTrue(
            any(
                "closer_retarget" in ln or "sticky_identity" in ln
                for ln in raw.debug_lines
            ),
            raw.debug_lines,
        )


if __name__ == "__main__":
    unittest.main()
