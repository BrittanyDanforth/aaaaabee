"""End-to-end checks: overlay drag is one buffer, wired in runtime, stable motion."""

from __future__ import annotations

import math
import unittest
from pathlib import Path

from motion import TargetTracker


class OverlayDragIntegrationTests(unittest.TestCase):
    def test_runtime_wires_unified_drag_with_bbox_and_dt(self) -> None:
        text = Path(__file__).resolve().parents[1].joinpath("runtime.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("set_monitor_overlay_point", text)
        self.assertIn("overlay_motion.overlay_xy()", text)
        motion_src = Path(__file__).resolve().parents[1].joinpath("motion.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("_advance_overlay_follow", motion_src)
        self.assertIn("configure_overlay_dot_alpha", motion_src)
        self.assertIn("sync_overlay_follow_frame", text)
        self.assertIn("motion.overlay_xy()", text)
        self.assertNotIn("cap_frame_display_step", text)

    def test_sixty_frame_jitter_has_bounded_step(self) -> None:
        """Detector noise must not recreate the dot — per-frame travel stays capped."""
        tracker = TargetTracker()
        tracker.smooth_overlay_point(500.0, 400.0, alpha=0.4, bbox_h=120, dt=1.0 / 60.0)
        prev = tracker.peek_overlay_smooth()
        assert prev is not None
        max_step = 0.0
        for i in range(1, 61):
            jx = 500.0 + (3.0 if i % 2 == 0 else -3.0)
            jy = 400.0 + (2.0 if i % 3 == 0 else -2.0)
            x, y = tracker.smooth_overlay_point(
                jx, jy, alpha=0.4, bbox_h=120, dt=1.0 / 60.0
            )
            step = math.hypot(x - prev[0], y - prev[1])
            max_step = max(max_step, step)
            prev = (x, y)
        self.assertLess(
            max_step,
            28.0,
            "monitor drag must cap per-frame travel (no teleport/recreate)",
        )

    def test_ring_reclamp_syncs_follow_state(self) -> None:
        tracker = TargetTracker()
        tracker.observe_target(
            100.0, 100.0, 0.0,
            bbox_x=70, bbox_y=50, bbox_w=60, bbox_h=100,
            aim_is_body_anchor=True,
        )
        tracker.sync_overlay_follow_frame(120.0, 100.0)
        m = tracker.observe_target(
            125.0, 100.0, 1.0 / 60.0,
            bbox_x=70, bbox_y=50, bbox_w=60, bbox_h=100,
            aim_is_body_anchor=True,
        )
        ox, _ = m.overlay_xy()
        self.assertGreater(ox, 118.0)
        self.assertLess(ox, 125.0)

    def test_hold_last_smooth_point_api(self) -> None:
        tracker = TargetTracker()
        self.assertIsNone(tracker.peek_overlay_smooth())
        tracker.smooth_overlay_point(10.0, 20.0, alpha=0.4, bbox_h=80, dt=1.0 / 60.0)
        pt = tracker.peek_overlay_smooth()
        assert pt is not None
        self.assertAlmostEqual(pt[0], 10.0)
        tracker.reset_overlay_smoothing()
        self.assertIsNone(tracker.peek_overlay_smooth())


if __name__ == "__main__":
    unittest.main()
