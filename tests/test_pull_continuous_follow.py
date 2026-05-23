"""Pull must track smoothly during steady aim — not only snap on large dot jumps."""

from __future__ import annotations

import unittest

from detector import Target
from motion import TargetTracker
from pull import PullController, PullTuning
from target_lock import may_assist_pull_target


def _tuning(**kw) -> PullTuning:
    base = dict(
        max_speed=18.0,
        pull_strength=0.88,
        deadzone=2.0,
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
    base.update(kw)
    return PullTuning(**base)


def _target(cx: float, cy: float) -> Target:
    return Target(
        centroid_x=cx,
        centroid_y=cy,
        area=4000.0,
        distance_to_center=30.0,
        confidence=0.85,
        bbox_x=int(cx - 30),
        bbox_y=int(cy - 70),
        bbox_w=60,
        bbox_h=140,
        body_shape_score=0.82,
        part_count=4,
        red_coverage=0.12,
        has_classified_torso=True,
    )


class PullContinuousFollowTests(unittest.TestCase):
    def test_may_assist_pull_during_stale_grace(self) -> None:
        t = _target(640.0, 420.0)
        self.assertTrue(
            may_assist_pull_target(
                t,
                detection_fresh=False,
                center_y=540.0,
                target_lost_frames=3,
                stale_grace_frames=12,
            )
        )

    def test_deadband_still_moves_pull_anchor(self) -> None:
        tr = TargetTracker()
        tr.configure_smoothing_tau(0.05, 0.02)
        t = 0.0
        m0 = tr.observe_target(
            640.0, 420.0, t, bbox_x=610, bbox_y=350, bbox_w=60, bbox_h=140,
            aim_is_body_anchor=True,
        )
        t += 1.0 / 60.0
        m1 = tr.observe_target(
            642.0, 421.0, t, bbox_x=612, bbox_y=351, bbox_w=60, bbox_h=140,
            aim_is_body_anchor=True,
        )
        pull_drift = ((m1.x - m0.x) ** 2 + (m1.y - m0.y) ** 2) ** 0.5
        ov0 = m0.overlay_xy()
        ov1 = m1.overlay_xy()
        overlay_drift = ((ov1[0] - ov0[0]) ** 2 + (ov1[1] - ov0[1]) ** 2) ** 0.5
        self.assertGreater(pull_drift, 0.05, "pull anchor must move during micro-track in deadband")
        self.assertGreater(
            overlay_drift, 0.05,
            "overlay follow must track the same micro-motion as pull",
        )

    def test_steady_track_produces_pull_over_frames(self) -> None:
        ctrl = PullController(_tuning())
        cx, cy = 640.0, 400.0
        moves = 0
        for i in range(30):
            off = i * 0.35
            tgt = _target(640.0 + off, 420.0 + off * 0.2)
            pr = ctrl.compute_delta(tgt, cx, cy, time_sec=i / 60.0)
            if pr.dx != 0 or pr.dy != 0:
                moves += 1
        self.assertGreaterEqual(
            moves,
            8,
            "steady strafe should produce continuous pull, not only rare snaps",
        )
