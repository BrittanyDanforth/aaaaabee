"""End-to-end overlay dot vs pull anchor — no shared deadband jitter."""

from __future__ import annotations

import math
import unittest

from motion import TargetMotion, TargetTracker


class OverlayDotPipelineTests(unittest.TestCase):
    def test_overlay_follows_strafe_while_pull_uses_deadband(self) -> None:
        """Overlay has its own follow path — must glide on strafe, not teleport."""
        tr = TargetTracker()
        tr.configure_smoothing_tau(0.05, 0.02)
        dt = 1.0 / 60.0
        tr.observe_target(
            500.0, 400.0, 0.0,
            bbox_x=470, bbox_y=330, bbox_w=60, bbox_h=140,
            aim_is_body_anchor=True,
        )
        prev_ox, prev_oy = 500.0, 400.0
        max_step = 0.0
        for i in range(1, 40):
            x = 500.0 + i * 2.5
            m = tr.observe_target(
                x, 400.0, i * dt,
                bbox_x=470, bbox_y=330, bbox_w=60, bbox_h=140,
                aim_is_body_anchor=True,
            )
            ox, oy = m.overlay_xy()
            max_step = max(max_step, math.hypot(ox - prev_ox, oy - prev_oy))
            prev_ox, prev_oy = ox, oy
        self.assertGreater(max_step, 0.4, "overlay must move each frame on strafe")
        self.assertLess(max_step, 12.0, "overlay must not teleport per frame")
        self.assertGreater(ox, 515.0, "overlay should track toward strafing target")

    def test_stationary_jitter_overlay_stays_tight(self) -> None:
        tr = TargetTracker()
        tr.configure_smoothing_tau(0.05, 0.02)
        m0 = tr.observe_target(
            500.0, 400.0, 0.0,
            bbox_x=470, bbox_y=330, bbox_w=60, bbox_h=140,
            aim_is_body_anchor=True,
        )
        dt = 1.0 / 60.0
        overlay_positions: list[tuple[float, float]] = []
        for i in range(1, 25):
            jitter_x = 500.0 + (1.0 if i % 2 == 0 else -1.0)
            jitter_y = 400.0 + (1.0 if i % 3 == 0 else -1.0)
            m = tr.observe_target(
                jitter_x, jitter_y, i * dt,
                bbox_x=470, bbox_y=330, bbox_w=60, bbox_h=140,
                aim_is_body_anchor=True,
            )
            overlay_positions.append(m.overlay_xy())
        ox0, oy0 = m0.overlay_xy()
        ox_max = max(abs(p[0] - ox0) for p in overlay_positions)
        oy_max = max(abs(p[1] - oy0) for p in overlay_positions)
        self.assertLess(ox_max, 4.0, "stationary overlay follow should damp 1px noise")
        self.assertLess(oy_max, 4.0)

    def test_overlay_xy_defaults_to_pull_when_unset(self) -> None:
        m = TargetMotion(10.0, 20.0, 1.0, 2.0)
        self.assertEqual(m.overlay_xy(), (10.0, 20.0))

    def test_overlay_xy_uses_split_anchor(self) -> None:
        m = TargetMotion(11.0, 22.0, 0.0, 0.0, overlay_x=100.0, overlay_y=200.0)
        self.assertEqual(m.overlay_xy(), (100.0, 200.0))
        self.assertEqual((m.x, m.y), (11.0, 22.0))


if __name__ == "__main__":
    unittest.main()
