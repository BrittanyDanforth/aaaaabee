"""Overlay dot and pull share the same aim anchor — pull centers on the dot."""

from __future__ import annotations

import math
import unittest

from motion import TargetMotion, TargetTracker, clamp_aim_to_display_fov


class OverlayDotPipelineTests(unittest.TestCase):
    def test_deadband_keeps_pull_and_overlay_aligned(self) -> None:
        tr = TargetTracker()
        tr.configure_smoothing_tau(0.05, 0.02)
        t = 0.0
        m0 = tr.observe_target(
            500.0, 400.0, t, bbox_x=470, bbox_y=330, bbox_w=60, bbox_h=140,
            aim_is_body_anchor=True,
        )
        dt = 1.0 / 60.0
        max_sep = 0.0
        for i in range(1, 25):
            jitter_x = 500.0 + (1.0 if i % 2 == 0 else -1.0)
            jitter_y = 400.0 + (1.0 if i % 3 == 0 else -1.0)
            m = tr.observe_target(
                jitter_x,
                jitter_y,
                i * dt,
                bbox_x=470,
                bbox_y=330,
                bbox_w=60,
                bbox_h=140,
                aim_is_body_anchor=True,
            )
            ox, oy = m.overlay_xy()
            sep = math.hypot(m.x - ox, m.y - oy)
            max_sep = max(max_sep, sep)
        self.assertLess(
            max_sep,
            0.01,
            "pull anchor must match overlay dot anchor (crosshair → dot)",
        )
        ox_max = max(abs(p[0] - 500.0) for p in [m0.overlay_xy()])
        self.assertLess(ox_max, 0.75, "overlay stays pinned on stationary jitter")

    def test_overlay_xy_defaults_to_pull_when_unset(self) -> None:
        m = TargetMotion(10.0, 20.0, 1.0, 2.0)
        self.assertEqual(m.overlay_xy(), (10.0, 20.0))

    def test_shared_fov_clamp_matches_runtime_ring(self) -> None:
        ax, ay = clamp_aim_to_display_fov(
            900.0, 400.0, 640.0, 400.0, detect_fov=200.0, display_fov=180.0,
        )
        dist = math.hypot(ax - 640.0, ay - 400.0)
        self.assertAlmostEqual(dist, min(200.0, 180.0) * 0.96, delta=0.5)


if __name__ == "__main__":
    unittest.main()
