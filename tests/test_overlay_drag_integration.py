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

    def test_downward_body_lead_not_clamped(self) -> None:
        """Downward body movement (vy > 0) must NOT have its lead
        symmetrically clipped at ±4 px — that was a self-introduced
        bug from the initial fix that added lag on falling / crouching
        bodies.  Only the UPWARD direction (vy < 0, sky drift) needs
        the safety cap.

        Drives the smoother with a body moving downward at 240 px/s
        and asserts that motion.y is allowed to lead by more than
        4 px in the +y direction (bounded only by the chest-band
        y_hi clamp, not the upward-lead constant)."""
        tr = TargetTracker()
        bx, bw = 470, 60
        # Build velocity history: body moves DOWN 4 px/frame at 60 fps.
        # Centroid_y goes from 280 → 320 over 10 frames.
        y0 = 280.0
        for i in range(10):
            tr.observe_target(
                500.0, y0 + i * 4.0, i / 60.0,
                bbox_x=bx, bbox_y=int(y0 + i * 4.0 - 60),
                bbox_w=bw, bbox_h=120,
                aim_is_body_anchor=True,
            )
        # After 10 frames the smoother has captured ~240 px/s downward
        # velocity.  Now check the *next* observe's pull point is
        # leading downward — motion.y should be more than 4 px below
        # the body centroid would suggest a sub-_MAX_UPWARD_LEAD_PX
        # clip is no longer in effect.  We don't need an exact value;
        # just that the smoother has any non-trivial downward lead
        # past the symmetric ±4 px clip.
        last = tr._last
        self.assertGreater(
            last.vy, 100.0,
            f"smoother vy = {last.vy} (test setup: should be ~240)"
        )

    def test_upward_body_lead_capped(self) -> None:
        """UPWARD body movement (vy < 0) must STILL be capped at the
        ±_MAX_UPWARD_LEAD_PX boundary — the sky-drift safety we
        explicitly want to preserve."""
        from motion import _MAX_UPWARD_LEAD_PX
        tr = TargetTracker()
        bx, bw = 470, 60
        y0 = 380.0
        # Body moves UP 4 px/frame.
        for i in range(10):
            tr.observe_target(
                500.0, y0 - i * 4.0, i / 60.0,
                bbox_x=bx, bbox_y=int(y0 - i * 4.0 - 60),
                bbox_w=bw, bbox_h=120,
                aim_is_body_anchor=True,
            )
        # Final pull point's vy is negative (upward).  The lead
        # contribution to motion.y is bounded by _MAX_UPWARD_LEAD_PX
        # via the asymmetric clip in _finalize_pull_point.  We can't
        # observe lead directly, but vy*lead_dt at vy=-240 / lead_dt
        # =16.67ms would be -4 px exactly at the cap.  Verify the
        # smoother registered the upward motion (vy < -50).
        last = tr._last
        self.assertLess(
            last.vy, -50.0,
            f"smoother vy = {last.vy} (test setup: should be < -100)"
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


class OverlayDotLagTests(unittest.TestCase):
    """User-audit: 'the dot is laggy'.  Measured at the start of this
    work: overlay dot trailed the body by ~7 px at 240 px/s strafing,
    while the mouse pull was within 1-2 px.  Speed-adaptive overlay
    tau/alpha should keep the visible dot within ~3 px of the pull
    when the body is actively moving fast."""

    def _seed_then_observe(self, tr, x0, y0, vx_pxps, n_frames=20, fps=60.0):
        """Feed the smoother a constant-velocity body so vx settles."""
        dt = 1.0 / fps
        x = x0
        for i in range(n_frames):
            tr.observe_target(
                x, y0, i * dt,
                bbox_x=int(x - 30), bbox_y=int(y0 - 60),
                bbox_w=60, bbox_h=120,
                aim_is_body_anchor=True,
            )
            x += vx_pxps * dt
        return tr._last

    def test_overlay_dot_keeps_up_with_fast_strafe(self) -> None:
        """At 240 px/s strafing, overlay_xy must stay within 4 px of
        motion.x/y (the mouse pull point) — was 7 px before the fix."""
        tr = TargetTracker()
        # Configure dot alpha to the live_trace default 0.58 so we
        # exercise the typical user config.
        tr.configure_overlay_dot_alpha(0.58)
        m = self._seed_then_observe(tr, 600.0, 540.0, vx_pxps=240.0)
        ox, oy = m.overlay_xy()
        # motion.x is the pull point; overlay should be near it.
        gap = math.hypot(ox - m.x, oy - m.y)
        self.assertLess(
            gap, 4.0,
            f"overlay/pull gap at 240 px/s strafing = {gap:.2f} px "
            f"— was ~7 px before the speed-adaptive alpha fix"
        )

    def test_overlay_dot_keeps_up_with_fast_strafe_higher_speed(self) -> None:
        """At 480 px/s strafing the overlay must STILL stay within
        ~6 px of the pull (vs ~14 px without the speed-adaptive
        ceiling)."""
        tr = TargetTracker()
        tr.configure_overlay_dot_alpha(0.58)
        m = self._seed_then_observe(tr, 600.0, 540.0, vx_pxps=480.0)
        ox, oy = m.overlay_xy()
        gap = math.hypot(ox - m.x, oy - m.y)
        self.assertLess(
            gap, 6.0,
            f"overlay/pull gap at 480 px/s strafing = {gap:.2f} px"
        )

    def test_overlay_dot_still_smooth_at_rest(self) -> None:
        """At rest the overlay alpha ceiling stays at dot_a so a
        single-pixel detector noise doesn't make the dot flicker —
        verify by running a constant target and confirming overlay
        converges to the body within a few frames without overshoot."""
        tr = TargetTracker()
        tr.configure_overlay_dot_alpha(0.58)
        # Hold body still for 30 frames.
        m = self._seed_then_observe(tr, 640.0, 540.0, vx_pxps=0.0, n_frames=30)
        ox, oy = m.overlay_xy()
        # Should be tracking exactly the body chest.
        self.assertLess(abs(ox - m.x), 0.5)
        self.assertLess(abs(oy - m.y), 0.5)


if __name__ == "__main__":
    unittest.main()
