"""Motion smoothing: deadband + production overlay follow (_advance_overlay_follow)."""

from __future__ import annotations

import math
import unittest

from motion import TargetTracker


class StationaryDeadbandTests(unittest.TestCase):
    def test_stationary_target_jitter_within_2px_is_absorbed(self) -> None:
        tracker = TargetTracker()
        m0 = tracker.observe(500.0, 400.0, 0.0)
        positions = []
        dt = 1.0 / 60.0
        for i in range(1, 30):
            jitter_x = 500.0 + (1.0 if i % 2 == 0 else -1.0)
            jitter_y = 400.0 + (1.0 if i % 3 == 0 else -1.0)
            m = tracker.observe(jitter_x, jitter_y, i * dt)
            positions.append(m.overlay_xy())
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        self.assertLess(max(abs(x - 500.0) for x in xs), 1.0)
        self.assertLess(max(abs(y - 400.0) for y in ys), 1.0)

    def test_slow_moving_target_still_tracks(self) -> None:
        tracker = TargetTracker()
        tracker.observe(500.0, 400.0, 0.0)
        dt = 1.0 / 60.0
        for i in range(1, 20):
            x = 500.0 + i * 2.0
            tracker.observe(x, 400.0, i * dt)
        m = tracker._last
        assert m is not None
        self.assertGreater(m.x, 510.0, "smoother must follow a 120 px/s target")


class OverlayFollowBehaviorTests(unittest.TestCase):
    """Production overlay drag via observe_target → _advance_overlay_follow (not smooth_overlay_point)."""

    def _observe(
        self,
        tr: TargetTracker,
        x: float,
        y: float,
        t: float,
        *,
        bh: int = 140,
    ):
        return tr.observe_target(
            x,
            y,
            t,
            bbox_x=int(x) - 30,
            bbox_y=int(y) - 70,
            bbox_w=60,
            bbox_h=bh,
            aim_is_body_anchor=True,
        )

    def test_overlay_follow_dampens_single_frame_jump(self) -> None:
        tr = TargetTracker()
        tr.configure_overlay_dot_alpha(0.45)
        dt = 1.0 / 60.0
        self._observe(tr, 100.0, 100.0, 0.0)
        m1 = self._observe(tr, 150.0, 100.0, dt)
        ox, _ = m1.overlay_xy()
        self.assertGreater(ox, 102.0)
        self.assertLess(ox, 145.0)

    def test_reset_overlay_smoothing_clears_follow_state(self) -> None:
        tr = TargetTracker()
        self._observe(tr, 100.0, 100.0, 0.0)
        self.assertIsNotNone(tr._overlay_follow_x)
        self.assertIsNotNone(tr._overlay_follow_y)
        tr.reset_overlay_smoothing()
        self.assertIsNone(tr._overlay_follow_x)
        self.assertIsNone(tr._overlay_follow_y)

    def test_follow_caps_upward_teleport_per_body_height(self) -> None:
        tr = TargetTracker()
        tr.configure_fov_clamp(400.0, 500.0, 400.0)
        dt = 1.0 / 60.0
        bh = 140
        m0 = self._observe(tr, 400.0, 500.0, 0.0, bh=bh)
        m1 = self._observe(tr, 400.0, 420.0, dt, bh=bh)
        _, oy0 = m0.overlay_xy()
        _, oy1 = m1.overlay_xy()
        step = abs(oy1 - oy0)
        cap = max(36.0, bh * 0.55)
        self.assertLess(step, cap + 3.0)

    def test_follow_allows_lateral_strafe(self) -> None:
        tr = TargetTracker()
        dt = 1.0 / 60.0
        m0 = self._observe(tr, 400.0, 500.0, 0.0, bh=120)
        m1 = self._observe(tr, 430.0, 502.0, dt, bh=120)
        ox0, _ = m0.overlay_xy()
        ox1, _ = m1.overlay_xy()
        self.assertGreater(ox1 - ox0, 3.0)


class LegacySmoothOverlayPointTests(unittest.TestCase):
    """Legacy API kept for compatibility; not called by AssistRuntime."""

    def test_legacy_helper_still_exists_for_compat(self) -> None:
        tr = TargetTracker()
        out = tr.smooth_overlay_point(100.0, 100.0)
        self.assertEqual(out, (100.0, 100.0))

    def test_legacy_passes_through_non_finite(self) -> None:
        tr = TargetTracker()
        out = tr.smooth_overlay_point(float("nan"), 100.0)
        self.assertTrue(math.isnan(out[0]))


if __name__ == "__main__":
    unittest.main()
