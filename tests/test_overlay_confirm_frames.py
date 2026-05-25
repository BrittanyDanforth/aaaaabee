"""Overlay dot needs 2 consecutive humanoid frames — no 1-frame ADS flash."""

from __future__ import annotations

import unittest

from detector import Target
from target_lock import OVERLAY_CONFIRM_FRAMES, TargetLockState, overlay_may_show_target


def _humanoid() -> Target:
    return Target(
        centroid_x=400.0,
        centroid_y=300.0,
        area=4000.0,
        distance_to_center=20.0,
        confidence=0.85,
        bbox_x=370,
        bbox_y=200,
        bbox_w=60,
        bbox_h=120,
        body_shape_score=0.72,
        part_count=4,
        red_coverage=0.12,
        fill_ratio=0.45,
        max_circularity=0.48,
        has_classified_torso=True,
        head_score=0.5,
        torso_score=0.55,
    )


class OverlayConfirmFramesTests(unittest.TestCase):
    def test_first_frame_hidden_with_lock_state(self) -> None:
        state = TargetLockState()
        t = _humanoid()
        self.assertFalse(
            overlay_may_show_target(
                t,
                detection_fresh=True,
                center_y=360.0,
                lock_state=state,
            )
        )
        self.assertEqual(state.overlay_confirm_frames, 1)

    def test_second_frame_shows(self) -> None:
        state = TargetLockState()
        t = _humanoid()
        for _ in range(OVERLAY_CONFIRM_FRAMES - 1):
            overlay_may_show_target(
                t, detection_fresh=True, center_y=360.0, lock_state=state
            )
        self.assertTrue(
            overlay_may_show_target(
                t, detection_fresh=True, center_y=360.0, lock_state=state
            )
        )

    def test_break_resets_counter(self) -> None:
        state = TargetLockState()
        t = _humanoid()
        overlay_may_show_target(
            t, detection_fresh=True, center_y=360.0, lock_state=state
        )
        overlay_may_show_target(
            None, detection_fresh=False, center_y=360.0, lock_state=state
        )
        self.assertEqual(state.overlay_confirm_frames, 0)


if __name__ == "__main__":
    unittest.main()
