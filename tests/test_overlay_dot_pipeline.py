"""Overlay dot vs pull: smooth dot, responsive pull for micro-corrections."""

from __future__ import annotations

import math
import unittest

from motion import TargetMotion, TargetTracker, clamp_aim_to_display_fov
from pull import PullController, PullTuning


class OverlayDotPipelineTests(unittest.TestCase):
    def test_deadband_pull_faster_than_overlay(self) -> None:
        tr = TargetTracker()
        tr.configure_smoothing_tau(0.05, 0.02)
        t = 0.0
        m0 = tr.observe_target(
            500.0, 400.0, t, bbox_x=470, bbox_y=330, bbox_w=60, bbox_h=140,
            aim_is_body_anchor=True,
        )
        dt = 1.0 / 60.0
        pull_drift = 0.0
        overlay_drift = 0.0
        for i in range(1, 25):
            jitter_x = 500.0 + (1.0 if i % 2 == 0 else -1.0)
            jitter_y = 400.0 + (1.0 if i % 3 == 0 else -1.0)
            m = tr.observe_target(
                jitter_x,
                jitter_y,
                i * dt,
                bbox_x=470,
                bbox_y=330,
                bbox_w=60,
                bbox_h=140,
                aim_is_body_anchor=True,
            )
            ox, oy = m.overlay_xy()
            pull_drift = max(pull_drift, math.hypot(m.x - m0.x, m.y - m0.y))
            overlay_drift = max(
                overlay_drift,
                math.hypot(ox - m0.overlay_xy()[0], oy - m0.overlay_xy()[1]),
            )
        self.assertGreater(pull_drift, 0.05, "pull anchor creeps for micro-track")
        self.assertLess(overlay_drift, 0.85, "overlay stays smoother in deadband")

    def test_micro_pull_inside_deadzone(self) -> None:
        ctrl = PullController(
            PullTuning(
                max_speed=18.0,
                pull_strength=0.88,
                deadzone=3.0,
                velocity_smoothing=0.42,
                smoothing_curve="ease_out",
                magnetism_radius=65.0,
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
        )
        from detector import Target

        tgt = Target(
            centroid_x=642.0,
            centroid_y=401.0,
            area=4000.0,
            distance_to_center=2.5,
            confidence=0.85,
            bbox_x=612,
            bbox_y=331,
            bbox_w=60,
            bbox_h=140,
            body_shape_score=0.82,
            part_count=4,
            red_coverage=0.12,
            has_classified_torso=True,
        )
        moves = 0
        for i in range(12):
            pr = ctrl.compute_delta(tgt, 640.0, 400.0, time_sec=i / 60.0)
            if pr.dx != 0 or pr.dy != 0:
                moves += 1
        self.assertGreaterEqual(
            moves,
            4,
            "pull must assist for small in-ring error (~2px), not only at FOV edge",
        )

    def test_overlay_xy_defaults_to_pull_when_unset(self) -> None:
        m = TargetMotion(10.0, 20.0, 1.0, 2.0)
        self.assertEqual(m.overlay_xy(), (10.0, 20.0))

    def test_shared_fov_clamp_matches_runtime_ring(self) -> None:
        ax, ay = clamp_aim_to_display_fov(
            900.0, 400.0, 640.0, 400.0, detect_fov=200.0, display_fov=180.0,
        )
        dist = math.hypot(ax - 640.0, ay - 400.0)
        self.assertAlmostEqual(dist, min(200.0, 180.0) * 0.96, delta=0.5)


if __name__ == "__main__":
    unittest.main()
