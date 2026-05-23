"""End-to-end overlay dot vs pull anchor — no shared deadband jitter."""

from __future__ import annotations

import math
import unittest

from motion import TargetMotion, TargetTracker


class OverlayDotPipelineTests(unittest.TestCase):
    def test_deadband_freezes_overlay_but_pull_creeps(self) -> None:
        tr = TargetTracker()
        tr.configure_smoothing_tau(0.05, 0.02)
        t = 0.0
        m0 = tr.observe_target(
            500.0, 400.0, t, bbox_x=470, bbox_y=330, bbox_w=60, bbox_h=140,
            aim_is_body_anchor=True,
        )
        dt = 1.0 / 60.0
        overlay_positions: list[tuple[float, float]] = []
        pull_positions: list[tuple[float, float]] = []
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
            overlay_positions.append((ox, oy))
            pull_positions.append((m.x, m.y))

        ox_max = max(abs(p[0] - m0.overlay_xy()[0]) for p in overlay_positions)
        oy_max = max(abs(p[1] - m0.overlay_xy()[1]) for p in overlay_positions)
        pull_drift = max(
            math.hypot(px - m0.x, py - m0.y) for px, py in pull_positions
        )
        self.assertLess(ox_max, 0.6, "overlay dot anchor should stay pinned in deadband")
        self.assertLess(oy_max, 0.6, "overlay dot anchor should stay pinned in deadband")
        self.assertGreater(pull_drift, 0.05, "pull anchor should still creep in deadband")

    def test_overlay_xy_defaults_to_pull_when_unset(self) -> None:
        m = TargetMotion(10.0, 20.0, 1.0, 2.0)
        self.assertEqual(m.overlay_xy(), (10.0, 20.0))

    def test_overlay_xy_uses_split_anchor(self) -> None:
        m = TargetMotion(11.0, 22.0, 0.0, 0.0, overlay_x=100.0, overlay_y=200.0)
        self.assertEqual(m.overlay_xy(), (100.0, 200.0))
        self.assertEqual((m.x, m.y), (11.0, 22.0))


if __name__ == "__main__":
    unittest.main()
