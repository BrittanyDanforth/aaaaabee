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
        self.assertIn("_frame_overlay_point", text)
        self.assertIn("monitor_overlay", text)
        motion_src = Path(__file__).resolve().parents[1].joinpath("motion.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("_advance_overlay_follow", motion_src)
        self.assertIn("configure_overlay_dot_alpha", motion_src)
        self.assertIn("sync_overlay_follow_frame", text)
        self.assertIn("motion.overlay_xy()", text)
        self.assertNotIn("cap_frame_display_step", text)

    def test_sixty_frame_jitter_has_bounded_step(self) -> None:
        """Production overlay follow caps per-frame travel on detector noise."""
        tracker = TargetTracker()
        dt = 1.0 / 60.0
        m0 = tracker.observe_target(
            500.0, 400.0, 0.0,
            bbox_x=470, bbox_y=280, bbox_w=60, bbox_h=120,
            aim_is_body_anchor=True,
        )
        prev = m0.overlay_xy()
        max_step = 0.0
        for i in range(1, 61):
            jx = 500.0 + (3.0 if i % 2 == 0 else -3.0)
            jy = 400.0 + (2.0 if i % 3 == 0 else -2.0)
            m = tracker.observe_target(
                jx, jy, i * dt,
                bbox_x=470, bbox_y=280, bbox_w=60, bbox_h=120,
                aim_is_body_anchor=True,
            )
            ox, oy = m.overlay_xy()
            step = math.hypot(ox - prev[0], oy - prev[1])
            max_step = max(max_step, step)
            prev = (ox, oy)
        self.assertLess(max_step, 28.0)

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
        tracker.set_monitor_overlay_point(10.0, 20.0)
        pt = tracker.peek_overlay_smooth()
        assert pt is not None
        self.assertAlmostEqual(pt[0], 10.0)
        tracker.reset_overlay_smoothing()
        self.assertIsNone(tracker.peek_overlay_smooth())

    def test_frame_overlay_point_clamps_to_display_ring(self) -> None:
        from runtime import AssistRuntime

        tr = TargetTracker()
        m = tr.observe_target(
            550.0, 300.0, 0.0,
            bbox_x=520, bbox_y=230, bbox_w=60, bbox_h=130,
            aim_is_body_anchor=True,
        )
        class _Reg:
            offset_x = 100
            offset_y = 50

        pt = AssistRuntime._frame_overlay_point(
            m,
            _Reg(),
            center_x=500.0,
            center_y=400.0,
            detect_fov=200.0,
            display_fov=140.0,
        )
        assert pt is not None
        fx_c = 500.0 - 100.0
        fy_c = 400.0 - 50.0
        dist = math.hypot(pt[0] - fx_c, pt[1] - fy_c)
        self.assertAlmostEqual(dist, 140.0 * 0.96, delta=3.0)


class OverlayPullUpwardDivergenceTests(unittest.TestCase):
    """Regression: motion.overlay_xy().y must not drift more than 3 px ABOVE
    motion.x/y inside the body bbox (the user-audit 'red dot creeps into the
    air while pull stays on the chest' symptom)."""

    def test_overlay_y_capped_above_pull_in_body_bbox(self) -> None:
        tr = TargetTracker()
        bx, by, bw, bh = 470, 280, 60, 120
        # Seed the smoother near the bottom chest band so pull settles low.
        for i in range(10):
            tr.observe_target(
                500.0, 395.0, i / 60.0,
                bbox_x=bx, bbox_y=by, bbox_w=bw, bbox_h=bh,
                aim_is_body_anchor=True,
            )
        # Now feed a measurement near the top of the chest band — the
        # overlay path will race toward it via _advance_overlay_follow
        # while pull continues to lag at the bottom edge.
        for i in range(10, 25):
            m = tr.observe_target(
                500.0, 315.0, i / 60.0,
                bbox_x=bx, bbox_y=by, bbox_w=bw, bbox_h=bh,
                aim_is_body_anchor=True,
            )
            ox, oy = m.overlay_xy()
            self.assertGreaterEqual(
                oy, m.y - 3.0,
                f"frame {i}: overlay_y={oy} pull_y={m.y} -- overlay drifted "
                f">3 px above pull (sky-drift symptom)",
            )

    def test_downward_divergence_not_clamped(self) -> None:
        tr = TargetTracker()
        bx, by, bw, bh = 470, 280, 60, 120
        # Seed at top of band so pull settles high.
        for i in range(10):
            tr.observe_target(
                500.0, 315.0, i / 60.0,
                bbox_x=bx, bbox_y=by, bbox_w=bw, bbox_h=bh,
                aim_is_body_anchor=True,
            )
        # Then move detector to bottom-band; overlay will race down
        # below pull.  This direction should NOT be capped — only
        # the upward (sky) direction matters for the user-audit fix.
        m = tr.observe_target(
            500.0, 395.0, 10 / 60.0,
            bbox_x=bx, bbox_y=by, bbox_w=bw, bbox_h=bh,
            aim_is_body_anchor=True,
        )
        # Just check it doesn't error and stays inside the bbox.
        ox, oy = m.overlay_xy()
        self.assertGreaterEqual(oy, by)
        self.assertLessEqual(oy, by + bh)


if __name__ == "__main__":
    unittest.main()
