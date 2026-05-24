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
from targeting_runtime import resolve_runtime_fov

GIF = Path(__file__).resolve().parents[1] / "artifacts/real_apex_test/_gif_frames_all"


def _cfg() -> dict:
    c = copy.deepcopy(profiles.PROFILE_DEFAULTS[PROFILE_APEX_STYLE_LIVE_TRACE])
    c["_ads_active"] = True
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
        """Stale rim lock (~frame 26) must hand off to center dummy on frame 45."""
        cfg = _cfg()
        state = TargetLockState()
        ctx = detector.DetectionContext()
        h = w = 0
        for fi in range(27):
            p = GIF / f"frame_{fi:04d}.png"
            img = cv2.imread(str(p))
            self.assertIsNotNone(img)
            h, w = img.shape[:2]
            resolve_runtime_fov(cfg, ads_active=True)
            det = int(cfg["_runtime_detect_fov"])
            ov = float(cfg["_runtime_overlay_fov"])
            sticky, locked, _ = detection_sticky_context(state, cfg)
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
            apply_target_lock(
                state,
                raw,
                center_y=h / 2,
                cfg=cfg,
                frame_size=(w, h),
                fov_cx=w / 2,
                fov_cy=h / 2,
            )

        rim = state.locked_target
        self.assertIsNotNone(rim)
        assert rim is not None
        self.assertGreater(rim.distance_to_center, 90.0)
        self.assertLess(rim.bbox_h, 60)

        img45 = cv2.imread(str(GIF / "frame_0045.png"))
        self.assertIsNotNone(img45)
        h, w = img45.shape[:2]
        resolve_runtime_fov(cfg, ads_active=True)
        det = int(cfg["_runtime_detect_fov"])
        ov = float(cfg["_runtime_overlay_fov"])
        sticky, locked, _ = detection_sticky_context(state, cfg)
        raw = detector.find_best_target(
            img45,
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
        apply_target_lock(
            state,
            raw,
            center_y=h / 2,
            cfg=cfg,
            frame_size=(w, h),
            fov_cx=w / 2,
            fov_cy=h / 2,
        )

        t = state.locked_target
        self.assertIsNotNone(t)
        assert t is not None
        dist = ((t.centroid_x - w / 2) ** 2 + (t.centroid_y - h / 2) ** 2) ** 0.5
        ov = effective_overlay_fov_radius(cfg)
        self.assertLess(
            dist,
            ov * 0.55,
            f"frame 45 should lock center dummy not rim fragment, dist={dist:.0f}",
        )
        self.assertGreaterEqual(t.bbox_h, 70, "should be full-size close dummy")

    def test_sticky_pool_retarget_on_frame_45(self) -> None:
        """Rim fragment in sticky pool must lose to closer center humanoid."""
        img = cv2.imread(str(GIF / "frame_0045.png"))
        self.assertIsNotNone(img)
        h, w = img.shape[:2]
        cfg = _cfg()
        state = TargetLockState()
        ctx = detector.DetectionContext()
        for fi in range(26):
            p = GIF / f"frame_{fi:04d}.png"
            im = cv2.imread(str(p))
            self.assertIsNotNone(im)
            hh, ww = im.shape[:2]
            resolve_runtime_fov(cfg, ads_active=True)
            det = int(cfg["_runtime_detect_fov"])
            ov = float(cfg["_runtime_overlay_fov"])
            sticky, locked, _ = detection_sticky_context(state, cfg)
            raw = detector.find_best_target(
                im,
                cfg["hsv_ranges"],
                det,
                float(cfg["min_target_area_pixels"]),
                ww / 2,
                hh / 2,
                detection_mode=DETECTION_MODE_APEX,
                context=ctx,
                sticky_target=sticky,
                currently_locked=locked,
                min_height_px=float(cfg["humanoid_min_height_pixels"]),
                display_fov_radius=ov,
                stickiness_pixels=float(cfg["target_stickiness_pixels"]),
            )
            apply_target_lock(state, raw, center_y=hh / 2, cfg=cfg, frame_size=(ww, hh))
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
            any("closer_retarget" in ln for ln in raw.debug_lines),
            raw.debug_lines,
        )


if __name__ == "__main__":
    unittest.main()
