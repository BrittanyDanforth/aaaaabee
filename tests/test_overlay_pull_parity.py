"""Pull target and visible dot must share the same ring-clamped frame point."""

from __future__ import annotations

import math
import unittest

from motion import TargetMotion, TargetTracker
from runtime import AssistRuntime


class OverlayPullParityTests(unittest.TestCase):
    def test_runtime_pull_uses_clamped_not_raw_overlay(self) -> None:
        """Ring clamp may differ from overlay_xy; pull must use clamped frame point."""
        tr = TargetTracker()
        tr.configure_fov_clamp(200.0, 200.0, 100.0)
        m = tr.observe_target(
            320.0, 200.0, 0.0,
            bbox_x=290, bbox_y=100, bbox_w=60, bbox_h=120,
            aim_is_body_anchor=True,
        )
        ox, oy = m.overlay_xy()

        class _Reg:
            offset_x = 0
            offset_y = 0

        fo = AssistRuntime._frame_overlay_point(
            m,
            _Reg(),
            center_x=200.0,
            center_y=200.0,
            detect_fov=150.0,
            display_fov=100.0,
        )
        assert fo is not None
        dist_raw = math.hypot(ox - 200.0, oy - 200.0)
        dist_pull = math.hypot(fo[0] - 200.0, fo[1] - 200.0)
        if dist_raw > 100.0 * 0.96:
            self.assertLess(dist_pull, dist_raw)
            self.assertAlmostEqual(dist_pull, 100.0 * 0.96, delta=2.0)
        else:
            self.assertAlmostEqual(fo[0], ox, delta=0.5)
            self.assertAlmostEqual(fo[1], oy, delta=0.5)

    def test_single_ring_margin_not_double(self) -> None:
        """After one 0.96 clamp, distance from center equals limit (not 0.92×)."""
        tr = TargetTracker()
        tr.configure_fov_clamp(200.0, 200.0, 100.0)
        m = tr.observe_target(
            320.0, 200.0, 0.0,
            bbox_x=290, bbox_y=100, bbox_w=60, bbox_h=120,
            aim_is_body_anchor=True,
        )

        class _Reg:
            offset_x = 0
            offset_y = 0

        fo = AssistRuntime._frame_overlay_point(
            m,
            _Reg(),
            center_x=200.0,
            center_y=200.0,
            detect_fov=150.0,
            display_fov=100.0,
        )
        assert fo is not None
        dist = math.hypot(fo[0] - 200.0, fo[1] - 200.0)
        self.assertAlmostEqual(dist, 100.0 * 0.96, delta=2.0)
        self.assertGreater(dist, 100.0 * 0.92)


if __name__ == "__main__":
    unittest.main()
