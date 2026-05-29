"""YOLO direct overlay must use target aim, not CV chest motion."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from detector import Target
from runtime import AssistRuntime


class YoloDirectOverlayTests(unittest.TestCase):
    def test_yolo_overlay_point_uses_target_centroid(self) -> None:
        target = Target(
            220.0,
            195.0,
            5000.0,
            15.0,
            0.9,
            bbox_x=190,
            bbox_y=120,
            bbox_w=60,
            bbox_h=100,
        )
        cap = MagicMock()
        cap.offset_x = 0
        cap.offset_y = 0
        cap.width = 416
        cap.height = 416
        pt = AssistRuntime._yolo_overlay_point_from_target(
            target,
            cap,
            center_x=208.0,
            center_y=208.0,
            detect_fov=208.0,
            display_fov=200.0,
        )
        self.assertIsNotNone(pt)
        assert pt is not None
        self.assertAlmostEqual(pt[0], 220.0, delta=2.0)
        self.assertAlmostEqual(pt[1], 195.0, delta=2.0)


if __name__ == "__main__":
    unittest.main()
