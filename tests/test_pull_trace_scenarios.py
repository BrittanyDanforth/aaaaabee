"""Trace default-on + scenario: moving target logs pull/gate to file."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from detector import Target
from motion import TargetTracker
from profiles import PROFILE_APEX_STYLE_LIVE_SAFE, apply_profile
from pull import PullController, PullTuning
from pull_trace import (
    PullTraceFrame,
    analyze_trace_file,
    format_trace_line,
    log_trace_frame,
    setup_trace_logging,
)


class PullTraceDefaultTests(unittest.TestCase):
    def test_live_safe_has_trace_on_by_default(self) -> None:
        cfg = apply_profile({"profile": PROFILE_APEX_STYLE_LIVE_SAFE})
        self.assertTrue(cfg["trace_pull"])
        self.assertFalse(cfg["trace_pull_console"])

    def test_dry_run_trace_off(self) -> None:
        from profiles import PROFILE_APEX_STYLE_DRY_RUN

        cfg = apply_profile({"profile": PROFILE_APEX_STYLE_DRY_RUN})
        self.assertFalse(cfg["trace_pull"])


class PullTraceScenarioTests(unittest.TestCase):
    def test_moving_target_writes_log_with_pull_and_gate(self) -> None:
        cfg = apply_profile({"profile": PROFILE_APEX_STYLE_LIVE_SAFE})
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "trace.log"
            cfg = {**cfg, "trace_pull_log_file": str(log), "trace_pull_interval_frames": 1}
            self.assertTrue(setup_trace_logging(cfg, app_root=Path(tmp)))

            tuning = PullTuning(
                max_speed=cfg["max_pull_speed_pixels_per_frame"],
                pull_strength=cfg["pull_strength"],
                deadzone=cfg["deadzone_pixels"],
                velocity_smoothing=cfg["velocity_smoothing"],
                smoothing_curve=cfg["smoothing_curve"],
                magnetism_radius=cfg["magnetism_radius_pixels"],
                magnetism_min_scale=cfg["magnetism_min_pull_scale"],
                fov_radius=float(cfg["fov_radius_ads_pixels"]),
                fov_edge_min_scale=cfg["fov_edge_min_pull_scale"],
                prediction_enabled=False,
                prediction_lead_seconds=0.0,
                prediction_max_pixels=0.0,
                humanize_enabled=False,
                humanize_amplitude=0.0,
                humanize_jerk_limit=0.0,
                aim_pre_smoothed=True,
            )
            ctrl = PullController(tuning)
            tr = TargetTracker()
            cx, cy = 200.0, 200.0
            ch = [cx, cy]
            for i in range(12):
                t = i / 30.0
                raw_x = cx + math.sin(t * 4) * 50
                m = tr.observe_target(
                    raw_x, cy, t,
                    bbox_x=int(raw_x) - 18, bbox_y=140, bbox_w=36, bbox_h=88,
                )
                tgt = Target(
                    m.x, m.y, 300, 10, 0.9,
                    bbox_x=int(raw_x) - 18, bbox_y=140, bbox_w=36, bbox_h=88,
                )
                pr = ctrl.compute_delta(tgt, ch[0], ch[1], time_sec=t)
                ch[0] += pr.dx
                ch[1] += pr.dy
                log_trace_frame(
                    PullTraceFrame(
                        frame=i + 1,
                        raw_target=(raw_x, cy),
                        motion_target=(m.x, m.y),
                        center=(cx, cy),
                        error=(m.x - ch[0], m.y - ch[1]),
                        pull_dxdy=(pr.dx, pr.dy),
                        pull_mag=pr.magnitude,
                        pull_vel=(pr.vel_x, pr.vel_y),
                        pull_desired=(pr.desired_x, pr.desired_y),
                        gate_allowed=True,
                        gate_reason="",
                        mouse_move_called=(pr.dx, pr.dy),
                        detection_fresh=True,
                        target_lost_frames=0,
                        stale_detection=False,
                    ),
                    cfg,
                )

            self.assertTrue(log.is_file())
            body = log.read_text(encoding="utf-8")
            self.assertIn("pull_dxdy=", body)
            self.assertIn("motion_target=", body)
            self.assertIn("gate_allowed=True", body)
            summary = analyze_trace_file(log)
            self.assertTrue(summary["ok"])
            self.assertGreater(summary["blocks"], 4)

    def test_format_includes_has_target(self) -> None:
        line = format_trace_line(
            PullTraceFrame(
                1, (1, 2), (3, 4), (0, 0), (3, 4), (1, 0), 1.0, has_target=True
            )
        )
        self.assertIn("has_target=True", line)


if __name__ == "__main__":
    unittest.main()
