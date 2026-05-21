"""Moving-target pull follow — prove low lag and gate does not over-block."""

from __future__ import annotations

import math
import unittest

from detector import Target
from motion import TargetTracker
from mouse_gate import MouseGateContext, evaluate_mouse_gate
from pull import PullController, PullTuning
from pull_trace import PullTraceFrame, format_trace_line


def _tuning(**kw) -> PullTuning:
    base = dict(
        max_speed=12.0,
        pull_strength=0.5,
        deadzone=2.0,
        velocity_smoothing=0.5,
        smoothing_curve="ease_out",
        magnetism_radius=70.0,
        magnetism_min_scale=0.35,
        fov_radius=140.0,
        fov_edge_min_scale=0.5,
        prediction_enabled=False,
        prediction_lead_seconds=0.04,
        prediction_max_pixels=24.0,
        humanize_enabled=False,
        humanize_amplitude=0.0,
        humanize_jerk_limit=0.0,
        aim_pre_smoothed=True,
    )
    base.update(kw)
    return PullTuning(**base)


def _simulate_lag(
    fps: float,
    *,
    amp: float = 80.0,
    freq_hz: float = 1.2,
    steps: int = 120,
) -> tuple[float, float, int]:
    ctrl = PullController(_tuning())
    cx, cy = 640.0, 360.0
    ch = [cx, cy]
    aim = TargetTracker()
    dt = 1.0 / fps
    errs: list[float] = []
    for i in range(steps):
        t = i * dt
        raw_x = cx + math.sin(t * freq_hz * 2.0 * math.pi) * amp
        m = aim.observe_target(
            raw_x, cy, t,
            bbox_x=int(raw_x) - 20, bbox_y=200, bbox_w=40, bbox_h=100,
        )
        tgt = Target(
            m.x, m.y, 400.0, 10.0, 0.9,
            bbox_x=int(raw_x) - 20, bbox_y=200, bbox_w=40, bbox_h=100,
        )
        pr = ctrl.compute_delta(tgt, ch[0], ch[1], time_sec=t)
        ch[0] += pr.dx
        ch[1] += pr.dy
        errs.append(math.hypot(m.x - ch[0], m.y - ch[1]))
    over30 = sum(1 for e in errs if e > 30.0)
    tail = errs[-30:] if len(errs) >= 30 else errs
    return max(tail), sum(tail) / len(tail), over30


class PullFollowTests(unittest.TestCase):
    def test_30fps_max_lag_under_30px_hard_motion(self) -> None:
        mx, avg, over30 = _simulate_lag(30.0)
        self.assertLess(mx, 32.0, f"30fps max lag {mx:.1f}px too high")
        self.assertLess(avg, 22.0)
        self.assertLessEqual(over30, 10)

    def test_60fps_max_lag_under_22px(self) -> None:
        mx, avg, _ = _simulate_lag(60.0)
        self.assertLess(mx, 22.0)
        self.assertLess(avg, 14.0)

    def test_30_and_60_fps_similar_time_lag(self) -> None:
        _, avg30, _ = _simulate_lag(30.0)
        _, avg60, _ = _simulate_lag(60.0)
        self.assertLess(avg30, avg60 * 2.2)

    def test_pre_smoothed_no_tracker_calls(self) -> None:
        ctrl = PullController(_tuning())
        tgt = Target(300, 220, 400, 10, 0.9, bbox_x=280, bbox_y=100, bbox_w=40, bbox_h=120)
        with unittest.mock.patch.object(ctrl._tracker, "observe_target") as obs:
            ctrl.compute_delta(tgt, 200, 200, time_sec=1.0)
        obs.assert_not_called()

    def test_gate_allows_dt_scaled_pull(self) -> None:
        cfg = {"allow_live_mouse": True, "dry_run": False}
        ctx = MouseGateContext(
            running=True,
            stopping=False,
            paused=False,
            mouse_enabled=True,
            ads_active=True,
            has_target=True,
            target_process_ok=True,
            dx=28,
            dy=0,
            max_pull_per_frame=12.0,
            pull_budget_scale=3.5,
        )
        r = evaluate_mouse_gate(cfg, ctx)
        self.assertTrue(r.allowed, r.reason)

    def test_gate_blocks_absurd_flick(self) -> None:
        cfg = {"allow_live_mouse": True, "dry_run": False}
        ctx = MouseGateContext(
            running=True,
            stopping=False,
            paused=False,
            mouse_enabled=True,
            ads_active=True,
            has_target=True,
            target_process_ok=True,
            dx=80,
            dy=0,
            max_pull_per_frame=12.0,
            pull_budget_scale=3.5,
        )
        r = evaluate_mouse_gate(cfg, ctx)
        self.assertFalse(r.allowed)

    def test_trace_line_format(self) -> None:
        line = format_trace_line(
            PullTraceFrame(
                frame=42,
                raw_target=(310.0, 228.0),
                motion_target=(305.0, 225.0),
                center=(200.0, 200.0),
                error=(105.0, 25.0),
                pull_dxdy=(8, 2),
                pull_mag=8.25,
                pull_vel=(9.1, 2.2),
                pull_desired=(12.0, 3.0),
                gate_allowed=True,
                gate_reason="",
                mouse_move_called=(8, 2),
                detection_fresh=True,
                target_lost_frames=0,
                stale_detection=False,
            )
        )
        self.assertIn("frame=42", line)
        self.assertIn("pull_output=(8,2)", line)
        self.assertIn("gate_allowed=True", line)


if __name__ == "__main__":
    unittest.main()


class PullEdgeFollowTests(unittest.TestCase):
    def test_edge_distance_keeps_strong_pull(self) -> None:
        from pull import pull_fov_distance_scale

        from motion import fov_distance_scale
        naive_edge = fov_distance_scale(170.0, 200.0, 0.65)
        edge = pull_fov_distance_scale(170.0, 200.0, 0.65)
        self.assertGreater(edge, naive_edge)
        self.assertGreaterEqual(edge, 0.82)

    def test_edge_target_moves_mouse(self) -> None:
        ctrl = PullController(_tuning(fov_radius=200.0, fov_edge_min_scale=0.88, max_speed=24.0))
        cx, cy = 200.0, 200.0
        tgt = Target(380.0, 200.0, 400.0, 185.0, 0.9, bbox_x=360, bbox_y=120, bbox_w=40, bbox_h=100)
        pr = ctrl.compute_delta(tgt, cx, cy, time_sec=0.0)
        self.assertGreater(pr.magnitude, 2.0, "edge target should produce pull")
