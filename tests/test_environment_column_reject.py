"""Reject firing-range tower / tall sky props — not stale overlay targets."""

from __future__ import annotations

import unittest

from detector import Target, bbox_top_in_sky_band, target_is_environment_column
from target_lock import lock_target_is_plausible


def _tower_target() -> Target:
    """Matches gif frames 6–13: tall bbox, top in sky, low red."""
    return Target(
        centroid_x=440.0,
        centroid_y=160.0,
        area=230 * 70,
        distance_to_center=30.0,
        confidence=0.75,
        bbox_x=400,
        bbox_y=45,
        bbox_w=70,
        bbox_h=230,
        body_shape_score=0.72,
        part_count=6,
        red_coverage=0.03,
        fill_ratio=0.45,
        max_circularity=0.4,
        has_classified_torso=False,
        head_score=0.3,
        torso_score=0.25,
        limb_stack_score=0.2,
    )


class EnvironmentColumnTests(unittest.TestCase):
    def test_bbox_top_in_sky_band(self) -> None:
        self.assertTrue(bbox_top_in_sky_band(45.0, 213.5))
        self.assertFalse(bbox_top_in_sky_band(180.0, 213.5))

    def test_tower_is_environment_column(self) -> None:
        t = _tower_target()
        self.assertTrue(
            target_is_environment_column(
                t, center_y=213.5, frame_w=468, frame_h=427
            )
        )

    def test_tower_not_plausible_lock(self) -> None:
        t = _tower_target()
        self.assertFalse(
            lock_target_is_plausible(
                t,
                center_y=213.5,
                frame_w=468,
                frame_h=427,
                fov_cx=234.0,
                fov_cy=213.5,
            )
        )


if __name__ == "__main__":
    unittest.main()
