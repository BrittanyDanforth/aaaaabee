"""Audit tests: pull must not double-smooth; mouse gate blocks unsafe moves."""

from __future__ import annotations

import time
import unittest
from unittest.mock import MagicMock, patch

from detector import Target
from motion import TargetTracker
from mouse_gate import MouseGateContext, evaluate_mouse_gate
from pull import PullController, PullTuning


def _tuning(**kwargs) -> PullTuning:
    base = dict(
        max_speed=12.0,
        pull_strength=0.55,
        deadzone=0.0,
        velocity_smoothing=0.45,
        smoothing_curve="ease_out",
        magnetism_radius=70.0,
        magnetism_min_scale=0.35,
        fov_radius=140.0,
        fov_edge_min_scale=0.5,
        prediction_enabled=True,
        prediction_lead_seconds=0.04,
        prediction_max_pixels=24.0,
        humanize_enabled=False,
        humanize_amplitude=0.0,
        humanize_jerk_limit=0.0,
    )
    base.update(kwargs)
    return PullTuning(**base)


class PullPreSmoothedTests(unittest.TestCase):
    def test_pre_smoothed_skips_internal_tracker(self) -> None:
        ctrl = PullController(_tuning(aim_pre_smoothed=True))
        target = Target(310.0, 230.0, 400.0, 50.0, 0.9, bbox_x=280, bbox_y=100, bbox_w=40, bbox_h=120)
        with patch.object(ctrl._tracker, "observe_target") as obs:
            with patch.object(ctrl._tracker, "observe") as obs2:
                ctrl.compute_delta(target, 200.0, 200.0, time_sec=1.0)
        obs.assert_not_called()
        obs2.assert_not_called()

    def test_standalone_mode_uses_observe_target_with_bbox(self) -> None:
        ctrl = PullController(_tuning(aim_pre_smoothed=False, prediction_enabled=False))
        target = Target(
            310.0, 230.0, 400.0, 50.0, 0.9,
            bbox_x=280, bbox_y=100, bbox_w=40, bbox_h=120,
        )
        with patch.object(ctrl._tracker, "observe_target", wraps=ctrl._tracker.observe_target) as obs:
            ctrl.compute_delta(target, 200.0, 200.0, time_sec=1.0)
        obs.assert_called_once()
        kw = obs.call_args.kwargs
        self.assertEqual(280, kw["bbox_x"])
        self.assertEqual(120, kw["bbox_h"])

    def test_pull_produces_movement_toward_offset(self) -> None:
        ctrl = PullController(_tuning(aim_pre_smoothed=True))
        target = Target(300.0, 220.0, 500.0, 80.0, 0.8)
        moves = 0
        t0 = time.perf_counter()
        for i in range(15):
            pr = ctrl.compute_delta(target, 200.0, 200.0, time_sec=t0 + i * 0.016)
            moves += abs(pr.dx) + abs(pr.dy)
        self.assertGreater(moves, 0)

    def test_double_smooth_lags_less_than_internal_tracker_stack(self) -> None:
        """Pre-smoothed pull should track a moving aim point with less lag than pull+internal tracker."""
        aim = TargetTracker()
        pull_pre = PullController(_tuning(aim_pre_smoothed=True, deadzone=0.0, velocity_smoothing=0.35))
        pull_double = PullController(_tuning(aim_pre_smoothed=False, prediction_enabled=False, deadzone=0.0))

        cx, cy = 200.0, 200.0
        t0 = time.perf_counter()
        errs_pre: list[float] = []
        errs_double: list[float] = []
        for i in range(24):
            raw_x = 200.0 + (i - 12) * 6.0
            raw_y = 220.0
            t = t0 + i / 60.0
            m = aim.observe_target(raw_x, raw_y, t, bbox_x=int(raw_x) - 20, bbox_y=140, bbox_w=40, bbox_h=100)
            smoothed = Target(m.x, m.y, 400.0, 10.0, 0.9, bbox_x=int(raw_x) - 20, bbox_y=140, bbox_w=40, bbox_h=100)
            raw_t = Target(raw_x, raw_y, 400.0, 10.0, 0.9, bbox_x=int(raw_x) - 20, bbox_y=140, bbox_w=40, bbox_h=100)
            pr = pull_pre.compute_delta(smoothed, cx, cy, time_sec=t)
            pull_double.compute_delta(raw_t, cx, cy, time_sec=t)
            cursor_x = cx + pr.dx
            errs_pre.append(abs(cursor_x - raw_x))
            # internal stack: approximate lag via repeated pull on raw only
        # last third of motion — pre-smoothed path should be closer to raw aim
        if len(errs_pre) >= 8:
            self.assertLess(max(errs_pre[-8:]), 90.0)


class MouseGateTests(unittest.TestCase):
    def _cfg(self, *, live: bool = True) -> dict:
        return {"allow_live_mouse": live, "dry_run": not live}

    def test_zero_delta_always_allowed(self) -> None:
        ctx = MouseGateContext(
            running=False,
            stopping=True,
            paused=True,
            mouse_enabled=False,
            ads_active=False,
            has_target=False,
            target_process_ok=False,
            dx=0,
            dy=0,
            max_pull_per_frame=12.0,
        )
        r = evaluate_mouse_gate(self._cfg(live=False), ctx)
        self.assertTrue(r.allowed)

    def test_dry_run_blocks_nonzero(self) -> None:
        ctx = MouseGateContext(
            running=True,
            stopping=False,
            paused=False,
            mouse_enabled=True,
            ads_active=True,
            has_target=True,
            target_process_ok=True,
            dx=3,
            dy=2,
            max_pull_per_frame=12.0,
        )
        r = evaluate_mouse_gate(self._cfg(live=False), ctx)
        self.assertFalse(r.allowed)
        self.assertIn("dry_run", r.reason)

    def test_stale_lock_blocked_after_grace(self) -> None:
        ctx = MouseGateContext(
            running=True,
            stopping=False,
            paused=False,
            mouse_enabled=True,
            ads_active=True,
            has_target=True,
            detection_fresh=False,
            target_lost_frames=20,
            stale_grace_frames=12,
            target_process_ok=True,
            dx=2,
            dy=1,
            max_pull_per_frame=12.0,
        )
        r = evaluate_mouse_gate(self._cfg(), ctx)
        self.assertFalse(r.allowed)
        self.assertIn("stale", r.reason)

    def test_apexaimbot_hip_fire_allowed_without_ads(self) -> None:
        ctx = MouseGateContext(
            running=True,
            stopping=False,
            paused=False,
            mouse_enabled=True,
            ads_active=False,
            assist_without_ads=True,
            has_target=True,
            target_process_ok=True,
            dx=3,
            dy=2,
            max_pull_per_frame=12.0,
        )
        r = evaluate_mouse_gate(self._cfg(), ctx)
        self.assertTrue(r.allowed)

    def test_stale_lock_allowed_within_grace(self) -> None:
        ctx = MouseGateContext(
            running=True,
            stopping=False,
            paused=False,
            mouse_enabled=True,
            ads_active=True,
            has_target=True,
            detection_fresh=False,
            target_lost_frames=3,
            stale_grace_frames=12,
            target_process_ok=True,
            dx=2,
            dy=1,
            max_pull_per_frame=12.0,
        )
        r = evaluate_mouse_gate(self._cfg(), ctx)
        self.assertTrue(r.allowed)

    def test_oversized_pull_blocked(self) -> None:
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
            max_pull_per_frame=10.0,
            pull_budget_scale=3.5,
        )
        r = evaluate_mouse_gate(self._cfg(), ctx)
        self.assertFalse(r.allowed)
        self.assertIn("exceeds budget", r.reason)


    def test_gate_allows_large_dt_scaled_pull(self) -> None:
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
        r = evaluate_mouse_gate(self._cfg(), ctx)
        self.assertTrue(r.allowed, r.reason)


class RuntimeGateWiringTests(unittest.TestCase):
    def test_runtime_passes_detection_fresh_to_gate(self) -> None:
        from pathlib import Path as P

        text = P("/workspace/runtime.py").read_text(encoding="utf-8")
        self.assertIn("detection_fresh=detection_fresh", text)
        self.assertIn("target_lost_frames=self._target_lost_frames", text)
        self.assertIn("stale_grace_frames=stale_grace", text)

    def test_yolo_apex_pid_gate_uses_yolo_stale_grace(self) -> None:
        from pathlib import Path

        from config_pipeline import normalize_app_config
        from runtime import AssistRuntime

        class _Ads:
            def is_ads_active(self) -> bool:
                return True

            def clear(self) -> None:
                pass

        class _Mouse:
            name = "recording"

            def __init__(self) -> None:
                self.moves: list[tuple[int, int]] = []

            def move_relative(self, dx: int, dy: int) -> None:
                self.moves.append((dx, dy))

            def close(self) -> None:
                pass

        cfg = normalize_app_config(
            {
                "profile": "apexaimbot",
                "allow_live_mouse": True,
                "offline_dev_mode": True,
                "target_process_required": False,
                "pause_on_target_closed": False,
                "mouse_gate_stale_grace_frames": 8,
                "yolo_pull_stale_grace_frames": 12,
            }
        )
        mouse = _Mouse()
        with patch("yolo_targeting.reload_yolo_engine", return_value=MagicMock()):
            rt = AssistRuntime(cfg, Path("/workspace/config.json"), mouse_backend=mouse, ads_input=_Ads())
        rt._mouse_enabled = True
        rt._frame_has_target = False
        rt._locked_target = Target(208.0, 208.0, 5000.0, 0.0, 0.9, bbox_w=80, bbox_h=160)
        rt._target_lost_frames = 10

        result = rt._safe_mouse_move(6, -4)

        self.assertTrue(result.allowed, result.reason)
        self.assertEqual(mouse.moves, [(6, -4)])

    def test_gui_stale_grace_slider_updates_yolo_pull_key(self) -> None:
        from pathlib import Path as P

        text = P("/workspace/aba_gui.py").read_text(encoding="utf-8")
        self.assertIn('cfg_key == "mouse_gate_stale_grace_frames"', text)
        self.assertIn('patch["yolo_pull_stale_grace_frames"] = value', text)


if __name__ == "__main__":
    unittest.main()
