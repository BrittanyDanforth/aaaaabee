"""Motion smoothing pins: stationary-target deadband + overlay EMA helper.

Together these prevent the 'glitchy / swimming dot' on stationary or
slowly-tracking enemies reported by the user in real Apex gameplay.
"""

from __future__ import annotations

import math
import unittest

from motion import TargetTracker


class StationaryDeadbandTests(unittest.TestCase):
    def test_stationary_target_jitter_within_2px_is_absorbed(self) -> None:
        """When the smoothed velocity is near zero and a measurement lands
        within 2 px of the smoothed position, the smoother should NOT update.
        Without this, the user sees the dot 'swim' on a still enemy as the
        detector centroid jitters 1-3 px every frame."""
        tracker = TargetTracker()
        # Seed the tracker at (500, 400).
        m0 = tracker.observe(500.0, 400.0, 0.0)
        # Several frames of small jitter ±1 px around (500, 400) at 60 FPS.
        positions = []
        dt = 1.0 / 60.0
        for i in range(1, 30):
            jitter_x = 500.0 + (1.0 if i % 2 == 0 else -1.0)
            jitter_y = 400.0 + (1.0 if i % 3 == 0 else -1.0)
            m = tracker.observe(jitter_x, jitter_y, i * dt)
            positions.append(m.overlay_xy())
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        # Pull deadband pins; overlay on bare observe uses the same smooth anchor.
        self.assertLess(max(abs(x - 500.0) for x in xs), 1.0)
        self.assertLess(max(abs(y - 400.0) for y in ys), 1.0)

    def test_slow_moving_target_still_tracks(self) -> None:
        """A target moving > 50 px/s must escape the deadband — the smoother
        must follow it, not freeze."""
        tracker = TargetTracker()
        tracker.observe(500.0, 400.0, 0.0)
        dt = 1.0 / 60.0
        # 120 px/s lateral motion: 2 px per frame.
        last_x = 500.0
        for i in range(1, 20):
            x = 500.0 + i * 2.0
            tracker.observe(x, 400.0, i * dt)
            last_x = x
        m = tracker._last
        assert m is not None
        # Smoothed x should track meaningfully toward the moving measurement.
        self.assertGreater(m.x, 510.0, "smoother must follow a 120 px/s target")


class OverlayEMATests(unittest.TestCase):
    def test_overlay_smoother_dampens_pop(self) -> None:
        """The post-FOV-clamp EMA must reduce a one-frame jump."""
        tracker = TargetTracker()
        first = tracker.smooth_overlay_point(100.0, 100.0)
        self.assertEqual(first, (100.0, 100.0), "first sample should pass through")
        # Apply a 50 px jump.
        second = tracker.smooth_overlay_point(150.0, 100.0)
        # Step-cap + EMA drag — must move toward 150 but not snap.
        self.assertGreater(second[0], 102.0)
        self.assertLess(second[0], 130.0)

    def test_overlay_smoother_reset_clears_state(self) -> None:
        tracker = TargetTracker()
        tracker.smooth_overlay_point(100.0, 100.0)
        tracker.reset_overlay_smoothing()
        # After reset the next call should re-seed at the new point.
        out = tracker.smooth_overlay_point(500.0, 500.0)
        self.assertEqual(out, (500.0, 500.0))

    def test_overlay_smoother_passes_through_non_finite(self) -> None:
        tracker = TargetTracker()
        out = tracker.smooth_overlay_point(float("nan"), 100.0)
        # Smoother must return the input as-is when not finite so the
        # downstream Tk renderer can guard before drawing.
        self.assertTrue(math.isnan(out[0]))


class OverlayDisplayCapTests(unittest.TestCase):
    def test_drag_smooth_blocks_sky_teleport(self) -> None:
        tracker = TargetTracker()
        tracker.smooth_overlay_point(400.0, 500.0, alpha=0.4, bbox_h=140, dt=1.0 / 60.0)
        _, y = tracker.smooth_overlay_point(
            400.0, 420.0, alpha=0.4, bbox_h=140, dt=1.0 / 60.0
        )
        self.assertGreater(y, 500.0 - 8.0, "upward drag step must be capped per body height")

    def test_drag_smooth_allows_lateral_strafe(self) -> None:
        tracker = TargetTracker()
        tracker.smooth_overlay_point(400.0, 500.0, alpha=0.4, bbox_h=120, dt=1.0 / 60.0)
        x, _ = tracker.smooth_overlay_point(
            430.0, 502.0, alpha=0.4, bbox_h=120, dt=1.0 / 60.0
        )
        self.assertGreater(x, 403.0)


if __name__ == "__main__":
    unittest.main()
