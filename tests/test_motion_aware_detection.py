"""Motion-aware detection: stateful DetectionContext + motion-overlap scoring.

These tests pin the new behavior introduced to help Apex-style enemies whose
armor color blends into the terrain: temporal frame-difference signals presence
even when the static shape mask is weak/noisy.
"""

from __future__ import annotations

import unittest

import cv2
import numpy as np

import detector


CX = 640
CY = 360
H = 720
W = 1280
FOV = 200
MIN_AREA = 40.0


def _grass_frame(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    f = np.full((H, W, 3), (62, 110, 70), dtype=np.uint8)
    noise = rng.integers(-8, 8, (H, W, 3), dtype=np.int16)
    return np.clip(f.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _draw_humanoid(
    frame: np.ndarray,
    cx: int,
    foot_y: int,
    *,
    armor: tuple[int, int, int],
    helmet: tuple[int, int, int],
) -> None:
    """Apex-style block humanoid: helmet, chest, arms, legs with internal gaps."""
    cv2.rectangle(frame, (cx - 13, foot_y - 110), (cx + 13, foot_y - 80), helmet, -1)
    cv2.rectangle(frame, (cx - 7, foot_y - 108), (cx + 7, foot_y - 86), (40, 30, 20), -1)
    cv2.rectangle(frame, (cx - 22, foot_y - 75), (cx + 22, foot_y - 32), armor, -1)
    cv2.rectangle(frame, (cx - 7, foot_y - 72), (cx + 7, foot_y - 38), (50, 50, 50), -1)
    cv2.rectangle(frame, (cx - 27, foot_y - 68), (cx - 20, foot_y - 40), armor, -1)
    cv2.rectangle(frame, (cx + 20, foot_y - 68), (cx + 27, foot_y - 40), armor, -1)
    cv2.rectangle(frame, (cx - 14, foot_y - 28), (cx - 3, foot_y + 20), armor, -1)
    cv2.rectangle(frame, (cx + 3, foot_y - 28), (cx + 14, foot_y + 20), armor, -1)


class DetectionContextTests(unittest.TestCase):
    def test_reset_clears_state(self) -> None:
        ctx = detector.DetectionContext()
        f = _grass_frame()
        detector.build_detection_mask(f, None, detection_mode="shape", context=ctx)
        self.assertIsNotNone(ctx.prev_gray)
        ctx.reset()
        self.assertIsNone(ctx.prev_gray)
        self.assertIsNone(ctx.last_motion_mask)

    def test_motion_mask_populated_after_two_frames(self) -> None:
        ctx = detector.DetectionContext(motion_assist=True, motion_threshold=8)
        f1 = _grass_frame(seed=1)
        _draw_humanoid(f1, CX - 20, CY + 50, armor=(40, 200, 230), helmet=(60, 140, 200))
        f2 = _grass_frame(seed=1)
        _draw_humanoid(f2, CX + 0, CY + 50, armor=(40, 200, 230), helmet=(60, 140, 200))
        detector.build_detection_mask(f1, None, detection_mode="shape", context=ctx)
        self.assertIsNone(ctx.last_motion_mask, "first frame: no prev → no motion mask")
        detector.build_detection_mask(f2, None, detection_mode="shape", context=ctx)
        self.assertIsNotNone(ctx.last_motion_mask)
        self.assertGreater(int((ctx.last_motion_mask > 0).sum()), 0)

    def test_motion_coverage_ratio_inside_bbox(self) -> None:
        ctx = detector.DetectionContext(motion_assist=True, motion_threshold=8)
        f1 = _grass_frame(seed=2)
        f2 = _grass_frame(seed=2)
        _draw_humanoid(f2, CX, CY + 50, armor=(40, 200, 230), helmet=(60, 140, 200))
        detector.build_detection_mask(f1, None, detection_mode="shape", context=ctx)
        detector.build_detection_mask(f2, None, detection_mode="shape", context=ctx)
        body_overlap = ctx.motion_coverage_ratio(CX - 30, CY - 80, 60, 140)
        empty_overlap = ctx.motion_coverage_ratio(50, 50, 60, 140)
        self.assertGreater(body_overlap, empty_overlap)
        self.assertGreater(body_overlap, 0.05)


class MotionValidationScoringTests(unittest.TestCase):
    """Motion overlap converts a marginal single-blob candidate into a target."""

    def test_motion_bonus_lifts_low_contrast_humanoid(self) -> None:
        """Motion context boosts confidence on a moving low-contrast target.

        When the same body sits still in the same place across both frames the
        motion mask is empty and behaves like the no-context path. When the
        body moves the motion-coverage bonus must be > 0 for the moved bbox.
        """
        armor = (40, 200, 230)
        helmet = (60, 140, 200)
        f1 = _grass_frame(seed=3)
        _draw_humanoid(f1, CX, CY + 50, armor=armor, helmet=helmet)
        f2 = _grass_frame(seed=3)
        _draw_humanoid(f2, CX + 18, CY + 50, armor=armor, helmet=helmet)
        ctx = detector.DetectionContext(motion_assist=True, motion_threshold=8)
        detector.find_best_target(
            f1, None, FOV, MIN_AREA, float(CX), float(CY), detection_mode="shape", context=ctx
        )
        detector.find_best_target(
            f2, None, FOV, MIN_AREA, float(CX), float(CY), detection_mode="shape", context=ctx
        )
        # Body has moved → motion-coverage at the new bbox is positive.
        overlap = ctx.motion_coverage_ratio(CX - 10, CY - 60, 60, 140)
        self.assertGreater(
            overlap, 0.03, "motion overlap must register a body that just moved"
        )

    def test_static_solid_blob_not_promoted(self) -> None:
        # A static red blob (no inter-frame movement) must NOT be detected even
        # when a context is supplied — motion bonus requires real movement.
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        cv2.rectangle(frame, (CX - 25, CY - 60), (CX + 25, CY + 60), (0, 0, 255), -1)
        # Blur to mimic the merged-blob synthetic scenario.
        frame = cv2.GaussianBlur(frame, (21, 21), 0)
        ctx = detector.DetectionContext(motion_assist=True, motion_threshold=8)
        # Feed the same frame twice → motion mask stays empty.
        detector.find_best_target(
            frame, None, FOV, MIN_AREA, float(CX), float(CY), detection_mode="shape", context=ctx
        )
        r = detector.find_best_target(
            frame, None, FOV, MIN_AREA, float(CX), float(CY), detection_mode="shape", context=ctx
        )
        self.assertFalse(r.active, "static blob must not be promoted by motion path")


if __name__ == "__main__":
    unittest.main()
