"""End-to-end: lateral strafe must not freeze lock, motion, or pull after ~1 s."""

from __future__ import annotations

import copy
import math
import unittest
from dataclasses import replace

import profiles
from detector import DetectionResult, Target
from motion import TargetTracker
from pull import PullController, PullTuning
from target_lock import TargetLockState, apply_target_lock


def _cfg() -> dict:
    return copy.deepcopy(
        profiles.PROFILE_DEFAULTS[profiles.PROFILE_APEX_STYLE_LIVE_TRACE]
    )


def _body(cx: float, cy: float, **kw) -> Target:
    d = dict(
        centroid_x=cx,
        centroid_y=cy,
        area=5000.0,
        distance_to_center=30.0,
        confidence=0.88,
        bbox_x=int(cx - 30),
        bbox_y=int(cy - 70),
        bbox_w=60,
        bbox_h=140,
        body_shape_score=0.82,
        part_count=4,
        red_coverage=0.12,
        has_classified_torso=True,
        torso_score=0.55,
    )
    d.update(kw)
    return Target(**d)


class LateralStrafeTrackingTests(unittest.TestCase):
    def test_geometry_track_at_iou_020_updates_lock(self) -> None:
        """Sticky-pool overlap without soft-refine drift cap still updates geometry."""
        state = TargetLockState()
        cfg = _cfg()
        cfg["_runtime_detect_fov"] = 200
        locked = _body(640.0, 420.0)
        state.locked_target = locked
        state.target_lost_frames = 0
        # Same body, shifted 35px — fails soft_refine drift<28 but passes geometry IoU.
        shifted = _body(
            675.0,
            420.0,
            bbox_x=645,
            bbox_y=350,
        )
        effective, stale = apply_target_lock(
            state,
            DetectionResult(shifted, 1, shifted.confidence),
            center_y=540.0,
            cfg=cfg,
            fov_cx=640.0,
            fov_cy=540.0,
            frame_size=(1280, 1080),
        )
        self.assertFalse(stale)
        assert effective.target is not None
        self.assertAlmostEqual(effective.target.centroid_x, 675.0, delta=0.5)

    def test_lock_centroid_updates_through_50px_strafe(self) -> None:
        """Cumulative drift > 28 px must not freeze the locked target."""
        state = TargetLockState()
        cfg = _cfg()
        cfg["_runtime_detect_fov"] = 200
        cx0, cy0 = 640.0, 420.0
        locked = _body(cx0, cy0)
        state.locked_target = locked
        state.target_lost_frames = 0

        for i in range(1, 61):
            dx = i * 0.85
            new_t = _body(cx0 + dx, cy0, bbox_x=locked.bbox_x + int(dx))
            effective, stale = apply_target_lock(
                state,
                DetectionResult(new_t, 1, new_t.confidence),
                center_y=540.0,
                cfg=cfg,
                fov_cx=640.0,
                fov_cy=540.0,
                frame_size=(1280, 1080),
            )
            self.assertFalse(stale, f"frame {i} should stay fresh during strafe")
            self.assertIsNotNone(effective.target)
            assert effective.target is not None
            self.assertAlmostEqual(
                effective.target.centroid_x,
                cx0 + dx,
                delta=0.5,
                msg=f"frame {i}: lock must track detector, not freeze at cx0",
            )

        self.assertGreater(state.locked_target.centroid_x - cx0, 40.0)

    def test_motion_and_pull_follow_strafing_anchor(self) -> None:
        tr = TargetTracker()
        tr.configure_smoothing_tau(0.05, 0.02)
        tuning = PullTuning(
            max_speed=18.0,
            pull_strength=0.88,
            deadzone=1.5,
            velocity_smoothing=0.42,
            smoothing_curve="ease_out",
            magnetism_radius=65.0,
            magnetism_min_scale=0.35,
            fov_radius=200.0,
            fov_edge_min_scale=0.5,
            prediction_enabled=False,
            prediction_lead_seconds=0.04,
            prediction_max_pixels=24.0,
            humanize_enabled=False,
            humanize_amplitude=0.0,
            humanize_jerk_limit=0.0,
            aim_pre_smoothed=True,
        )
        pull = PullController(tuning)
        screen_cx, screen_cy = 640.0, 400.0
        t = 0.0
        dt = 1.0 / 60.0
        pull_moves = 0
        last_ox = 640.0
        for i in range(72):
            tx = 640.0 + i * 0.9
            ty = 420.0
            motion = tr.observe_target(
                tx,
                ty,
                t,
                bbox_x=int(tx - 30),
                bbox_y=350,
                bbox_w=60,
                bbox_h=140,
                aim_is_body_anchor=True,
            )
            ox, oy = motion.overlay_xy()
            self.assertGreater(ox, last_ox - 0.1, "overlay must advance with strafe")
            last_ox = ox
            tgt = replace(_body(tx, ty), centroid_x=ox, centroid_y=oy)
            pr = pull.compute_delta(tgt, screen_cx, screen_cy, time_sec=t)
            if pr.dx != 0 or pr.dy != 0:
                pull_moves += 1
            t += dt

        self.assertGreater(last_ox - 640.0, 35.0, "smoothed overlay should trail strafe")
        self.assertGreaterEqual(
            pull_moves,
            12,
            "pull must keep correcting while target strafes in front of crosshair",
        )

    def test_pull_error_to_overlay_stays_bounded(self) -> None:
        """Runtime pull target uses overlay_xy — error should stay modest vs dot."""
        tr = TargetTracker()
        tr.configure_smoothing_tau(0.04, 0.02)
        screen_cx, screen_cy = 640.0, 400.0
        t = 0.0
        dt = 1.0 / 60.0
        max_gap = 0.0
        for i in range(40):
            tx = 650.0 + math.sin(i * 0.15) * 8.0
            ty = 418.0
            motion = tr.observe_target(
                tx, ty, t, bbox_x=int(tx - 30), bbox_y=350, bbox_w=60, bbox_h=140,
            )
            ox, oy = motion.overlay_xy()
            gap = math.hypot(ox - screen_cx, oy - screen_cy) - math.hypot(
                tx - screen_cx, ty - screen_cy
            )
            max_gap = max(max_gap, abs(gap))
            t += dt
        self.assertLess(max_gap, 25.0, "overlay should not lag absurdly behind raw aim")


if __name__ == "__main__":
    unittest.main()
