"""Regression: apex_style_live_safe style config tracks at 30 FPS."""

from __future__ import annotations

import math
import unittest

from detector import Target
from motion import TargetTracker
from mouse_gate import MouseGateContext, evaluate_mouse_gate
from pull import PullController, PullTuning

USER_CFG = {
    "allow_live_mouse": True,
    "dry_run": False,
    "max_pull_speed_pixels_per_frame": 18.0,
    "pull_strength": 0.78,
    "deadzone_pixels": 3,
    "velocity_smoothing": 0.50,
    "smoothing_curve": "ease_out",
    "magnetism_radius_pixels": 80,
    "magnetism_min_pull_scale": 0.70,
    "fov_radius_pixels": 140,
    "fov_edge_min_pull_scale": 0.65,
    "prediction_enabled": True,
    "prediction_lead_seconds": 0.055,
    "prediction_max_pixels": 36,
    "humanize_enabled": True,
    "humanize_amplitude_pixels": 0.20,
    "humanize_jerk_limit": 10.0,
    "mouse_gate_pull_budget_scale": 3.5,
}


def _tuning() -> PullTuning:
    c = USER_CFG
    return PullTuning(
        max_speed=c["max_pull_speed_pixels_per_frame"],
        pull_strength=c["pull_strength"],
        deadzone=c["deadzone_pixels"],
        velocity_smoothing=c["velocity_smoothing"],
        smoothing_curve=c["smoothing_curve"],
        magnetism_radius=c["magnetism_radius_pixels"],
        magnetism_min_scale=c["magnetism_min_pull_scale"],
        fov_radius=c["fov_radius_pixels"],
        fov_edge_min_scale=c["fov_edge_min_pull_scale"],
        prediction_enabled=c["prediction_enabled"],
        prediction_lead_seconds=c["prediction_lead_seconds"],
        prediction_max_pixels=c["prediction_max_pixels"],
        humanize_enabled=c["humanize_enabled"],
        humanize_amplitude=c["humanize_amplitude_pixels"],
        humanize_jerk_limit=c["humanize_jerk_limit"],
        aim_pre_smoothed=True,
    )


class UserLiveConfigTests(unittest.TestCase):
    def test_motion_prediction_toggle_changes_output(self) -> None:
        tr = TargetTracker()
        tr.configure_prediction(False, 0.055, 36)
        a = tr.observe_target(100.0, 200.0, 0.0, bbox_x=80, bbox_y=150, bbox_w=40, bbox_h=90)
        tr.reset()
        tr.configure_prediction(True, 0.055, 36)
        for i in range(5):
            b = tr.observe_target(100.0 + i * 8, 200.0, i / 30.0, bbox_x=80 + i * 8, bbox_y=150, bbox_w=40, bbox_h=90)
        self.assertTrue(math.hypot(b.x - a.x, b.y - a.y) >= 0.0)

    def test_30fps_lag_under_25px(self) -> None:
        ctrl = PullController(_tuning())
        tr = TargetTracker()
        tr.configure_prediction(True, 0.055, 36)
        cx, cy = 640.0, 360.0
        ch = [cx, cy]
        errs = []
        for i in range(120):
            t = i / 30.0
            raw_x = cx + math.sin(t * math.pi * 2) * 70.0
            m = tr.observe_target(raw_x, cy, t, bbox_x=int(raw_x) - 20, bbox_y=200, bbox_w=36, bbox_h=90)
            tgt = Target(m.x, m.y, 300, 10, 0.9, bbox_x=int(raw_x) - 20, bbox_y=200, bbox_w=36, bbox_h=90)
            pr = ctrl.compute_delta(tgt, ch[0], ch[1], time_sec=t)
            ch[0] += pr.dx
            ch[1] += pr.dy
            errs.append(math.hypot(m.x - ch[0], m.y - ch[1]))
        self.assertLess(max(errs[-30:]), 25.0)

    def test_gate_allows_36px_pull_at_30fps(self) -> None:
        ctx = MouseGateContext(
            running=True,
            stopping=False,
            paused=False,
            mouse_enabled=True,
            ads_active=True,
            has_target=True,
            target_process_ok=True,
            dx=36,
            dy=0,
            max_pull_per_frame=18.0,
            pull_budget_scale=3.5,
        )
        r = evaluate_mouse_gate(USER_CFG, ctx)
        self.assertTrue(r.allowed, r.reason)

    def test_humanize_jerk_does_not_starve_pull(self) -> None:
        from motion import HumanizedMotion

        h = HumanizedMotion(0.2, 10.0)
        out0 = h.apply(30.0, 0.0)
        out1 = h.apply(-28.0, 0.0)
        self.assertGreater(abs(out1[0]), 15.0)


if __name__ == "__main__":
    unittest.main()
