"""Red dot drag: frame follow uses chest-clamped aim, not deadband 2px cap."""

from __future__ import annotations

import unittest

from motion import TargetTracker


class OverlayFollowWiringTests(unittest.TestCase):
    def test_deadband_cap_does_not_pin_overlay_follow(self) -> None:
        """Overlay follow uses pre-cap measurement; pull path keeps 2px cap."""
        tr = TargetTracker()
        tr.configure_smoothing_tau(0.08, 0.03)
        dt = 1.0 / 60.0
        for i in range(25):
            tr.observe_target(
                500.0 + (2.5 if i % 2 == 0 else -2.5),
                400.0,
                i * dt,
                bbox_x=470,
                bbox_y=280,
                bbox_w=60,
                bbox_h=120,
                aim_is_body_anchor=True,
            )
        m0 = tr.observe_target(
            500.0, 400.0, 25 * dt,
            bbox_x=470, bbox_y=280, bbox_w=60, bbox_h=120,
            aim_is_body_anchor=True,
        )
        m1 = tr.observe_target(
            512.0, 400.0, 26 * dt,
            bbox_x=470, bbox_y=280, bbox_w=60, bbox_h=120,
            aim_is_body_anchor=True,
        )
        ox0, _ = m0.overlay_xy()
        ox1, _ = m1.overlay_xy()
        overlay_step = abs(ox1 - ox0)
        pull_step = abs(m1.x - m0.x)
        self.assertGreater(overlay_step, pull_step * 0.5)

    def test_overlay_dot_alpha_changes_follow_speed(self) -> None:
        tr_slow = TargetTracker()
        tr_slow.configure_overlay_dot_alpha(0.15)
        tr_fast = TargetTracker()
        tr_fast.configure_overlay_dot_alpha(0.85)
        dt = 1.0 / 60.0
        for tr in (tr_slow, tr_fast):
            tr.observe_target(
                400.0, 300.0, 0.0,
                bbox_x=370, bbox_y=230, bbox_w=60, bbox_h=130,
                aim_is_body_anchor=True,
            )
        for i in range(1, 20):
            ax = 400.0 + i * 3.0
            m_slow = tr_slow.observe_target(
                ax, 300.0, i * dt,
                bbox_x=int(ax - 30), bbox_y=230, bbox_w=60, bbox_h=130,
                aim_is_body_anchor=True,
            )
            m_fast = tr_fast.observe_target(
                ax, 300.0, i * dt,
                bbox_x=int(ax - 30), bbox_y=230, bbox_w=60, bbox_h=130,
                aim_is_body_anchor=True,
            )
        slow_x, _ = m_slow.overlay_xy()
        fast_x, _ = m_fast.overlay_xy()
        self.assertGreater(fast_x, slow_x)


if __name__ == "__main__":
    unittest.main()
