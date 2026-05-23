"""~3s ADS hold regressions (30 FPS ≈ frame 90): sky creep, motion memory, anchor."""

from __future__ import annotations

import copy
import unittest
from pathlib import Path

import profiles
from detector import DetectionContext, DetectionResult, Target
from motion import TargetTracker
from target_lock import (
    MAX_LOCK_UPWARD_DRIFT_PX,
    TargetLockState,
    apply_target_lock,
    locked_target_may_refresh_motion_memory,
)


def _cfg() -> dict:
    return copy.deepcopy(
        profiles.PROFILE_DEFAULTS[profiles.PROFILE_APEX_STYLE_LIVE_TRACE]
    )


def _body(**kw) -> Target:
    d = dict(
        centroid_x=640.0,
        centroid_y=420.0,
        area=5000.0,
        distance_to_center=30.0,
        confidence=0.88,
        bbox_x=610,
        bbox_y=300,
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


class AdsThreeSecondDriftTests(unittest.TestCase):
    def test_cumulative_upward_refine_blocked_by_lock_anchor(self) -> None:
        """Without an anchor, identity-track refinements can climb ~8 px/frame into sky."""
        state = TargetLockState()
        cfg = _cfg()
        cfg["_runtime_detect_fov"] = 200
        locked = _body(centroid_y=420.0, bbox_y=300)
        state.locked_target = locked
        state._lock_anchor_cy = 420.0
        state.target_lost_frames = 0
        cy = 420.0
        for _ in range(14):
            cy -= 8.0
            up = _body(
                centroid_x=642.0,
                centroid_y=cy,
                bbox_y=int(300 + (420.0 - cy)),
                red_coverage=0.11,
            )
            effective, _ = apply_target_lock(
                state,
                DetectionResult(up, 1, up.confidence),
                center_y=540.0,
                cfg=cfg,
                fov_cx=640.0,
                fov_cy=540.0,
                frame_size=(1280, 1080),
            )
            assert effective.target is not None
            locked = effective.target
        self.assertGreaterEqual(
            locked.centroid_y,
            420.0 - MAX_LOCK_UPWARD_DRIFT_PX - 2.0,
            "lock must not drift more than anchor upward budget",
        )
        self.assertLessEqual(
            420.0 - locked.centroid_y,
            MAX_LOCK_UPWARD_DRIFT_PX + 4.0,
        )

    def test_motion_memory_stays_alive_across_95_detection_frames(self) -> None:
        """At 30 FPS, frame 90 used to decay memory; per-frame refresh must keep credit."""
        ctx = DetectionContext(motion_memory_frames=24)
        state = TargetLockState()
        locked = _body()
        state._lock_anchor_cy = locked.centroid_y
        center_y = 540.0
        gray = __import__("numpy").zeros((1080, 1280), dtype="uint8")
        for _ in range(95):
            if locked_target_may_refresh_motion_memory(
                locked, state, center_y=center_y
            ):
                ctx.note_motion_validated(
                    locked.bbox_x, locked.bbox_y, locked.bbox_w, locked.bbox_h
                )
            ctx.update_prev(gray)
        mem = ctx.motion_coverage_with_memory(
            locked.bbox_x, locked.bbox_y, locked.bbox_w, locked.bbox_h
        )
        self.assertGreater(
            mem,
            0.08,
            "stationary 3s ADS must not expire motion memory mid-fight",
        )

    def test_motion_memory_not_refreshed_for_sky_locked_bbox(self) -> None:
        state = TargetLockState()
        sky = _body(centroid_y=120.0, bbox_y=40, bbox_h=80, red_coverage=0.05)
        state._lock_anchor_cy = 120.0
        self.assertFalse(
            locked_target_may_refresh_motion_memory(
                sky, state, center_y=540.0
            )
        )

    def test_pull_anchor_caps_upward_jump(self) -> None:
        tracker = TargetTracker()
        m0 = tracker.observe_target(
            640.0,
            420.0,
            0.0,
            bbox_x=610,
            bbox_y=300,
            bbox_w=60,
            bbox_h=140,
            aim_is_body_anchor=True,
        )
        dt = 1.0 / 60.0
        m1 = tracker.observe_target(
            640.0,
            360.0,
            dt,
            bbox_x=610,
            bbox_y=240,
            bbox_w=60,
            bbox_h=140,
            aim_is_body_anchor=True,
        )
        self.assertGreater(
            m1.y,
            m0.y - 22.0,
            "pull anchor must not jump upward more than one capped step per frame",
        )

    def test_runtime_no_ads_hold_credit_decay_pulses(self) -> None:
        text = Path(__file__).resolve().parents[1].joinpath("runtime.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("locked_target_may_refresh_motion_memory", text)
        self.assertNotIn("_validated_credit) - 4", text)
        self.assertNotIn("_ads_hold_frames in (90, 180, 270, 360)", text)


if __name__ == "__main__":
    unittest.main()
