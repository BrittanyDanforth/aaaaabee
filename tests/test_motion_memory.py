"""DetectionContext motion-validated memory — once-validated targets keep their
motion bonus for ``motion_memory_frames`` after they stop moving.

Without the memory a body that strafes for a few frames and then stops loses
its motion bonus next frame, drops below the confidence floor, and triggers a
re-lock flicker. With the memory the validation decays linearly and the body
stays locked for a brief grace window.
"""

from __future__ import annotations

import unittest

import numpy as np

import detector


class MotionMemoryTests(unittest.TestCase):
    def test_credit_decays_to_zero_after_memory_window(self) -> None:
        ctx = detector.DetectionContext(motion_memory_frames=5)
        ctx.note_motion_validated(100, 100, 60, 140)
        first = ctx.motion_coverage_with_memory(100, 100, 60, 140)
        self.assertGreater(first, 0.0, "fresh memory must produce positive virtual coverage")
        # Tick the credit by simulating prev_gray updates.
        gray = np.zeros((480, 640), dtype=np.uint8)
        for _ in range(5):
            ctx.update_prev(gray)
        after = ctx.motion_coverage_with_memory(100, 100, 60, 140)
        self.assertEqual(after, 0.0, "credit must expire after motion_memory_frames")

    def test_memory_does_not_help_distant_bbox(self) -> None:
        ctx = detector.DetectionContext(motion_memory_frames=8)
        ctx.note_motion_validated(100, 100, 60, 140)
        # Bbox far from validated centre must not inherit the bonus.
        far = ctx.motion_coverage_with_memory(500, 100, 60, 140)
        self.assertEqual(far, 0.0)

    def test_live_coverage_takes_priority_over_memory(self) -> None:
        ctx = detector.DetectionContext()
        # Fake a motion mask that fully covers the queried bbox.
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[100:200, 100:200] = 255
        ctx.last_motion_mask = mask
        live = ctx.motion_coverage_with_memory(100, 100, 100, 100)
        self.assertGreater(live, 0.5, "live coverage must be returned when available")

    def test_reset_clears_memory(self) -> None:
        ctx = detector.DetectionContext(motion_memory_frames=8)
        ctx.note_motion_validated(100, 100, 60, 140)
        ctx.reset()
        self.assertEqual(
            ctx.motion_coverage_with_memory(100, 100, 60, 140),
            0.0,
            "reset must drop validation memory",
        )

    def test_static_target_after_motion_keeps_validation_credit(self) -> None:
        """End-to-end: a body that moves for one frame then stops should retain
        a positive overlap from the memory channel for the next several frames."""
        ctx = detector.DetectionContext(motion_memory_frames=10)
        H, W = 480, 640
        # First frame: empty motion mask, but mark the bbox as validated as if
        # the scoring layer had observed live motion overlap.
        ctx.note_motion_validated(220, 180, 60, 140)
        # Tick a couple of frames to simulate the body sitting still.
        gray = np.zeros((H, W), dtype=np.uint8)
        ctx.update_prev(gray)
        ctx.update_prev(gray)
        # Memory should still be active and produce a positive overlap.
        v = ctx.motion_coverage_with_memory(220, 180, 60, 140)
        self.assertGreater(
            v, detector.DetectionContext.motion_validate_threshold * 0.5,
            "stationary body within memory window must report non-trivial overlap",
        )


if __name__ == "__main__":
    unittest.main()
