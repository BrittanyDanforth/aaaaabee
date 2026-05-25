"""Overlay dot must not show on crate-like false positives."""

from __future__ import annotations

import unittest

from detector import Target
from target_lock import overlay_may_show_target


def _crate_like() -> Target:
    return Target(
        centroid_x=400.0,
        centroid_y=300.0,
        area=8000.0,
        distance_to_center=40.0,
        confidence=0.8,
        bbox_x=300,
        bbox_y=250,
        bbox_w=200,
        bbox_h=100,
        body_shape_score=0.62,
        part_count=1,
        red_coverage=0.10,
        fill_ratio=0.92,
        max_circularity=0.70,
        has_classified_torso=False,
        head_score=0.15,
        torso_score=0.10,
    )


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


class OverlayHumanoidGateTests(unittest.TestCase):
    def test_crate_like_blocked(self) -> None:
        self.assertFalse(
            overlay_may_show_target(
                _crate_like(),
                detection_fresh=True,
                center_y=360.0,
            )
        )

    def test_humanoid_allowed_when_fresh(self) -> None:
        self.assertTrue(
            overlay_may_show_target(
                _humanoid(),
                detection_fresh=True,
                center_y=360.0,
            )
        )

    def test_not_fresh_never_shows(self) -> None:
        self.assertFalse(
            overlay_may_show_target(
                _humanoid(),
                detection_fresh=False,
                center_y=360.0,
            )
        )


if __name__ == "__main__":
    unittest.main()
