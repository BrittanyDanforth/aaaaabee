"""Long ADS hold must not drift lock upward / adopt sky clutter."""

from __future__ import annotations

import copy
import unittest

import profiles
from detector import Target
from target_lock import TargetLockState, apply_target_lock
from detector import DetectionResult


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


class LongAdsDriftTests(unittest.TestCase):
    def test_upward_sky_steal_blocked_from_instant_adopt(self) -> None:
        state = TargetLockState()
        cfg = _cfg()
        locked = _body(centroid_y=420.0, red_coverage=0.12)
        state.locked_target = locked
        state.target_lost_frames = 0
        sky_fp = _body(
            centroid_x=645.0,
            centroid_y=380.0,
            bbox_y=250,
            bbox_h=100,
            red_coverage=0.02,
            body_shape_score=0.85,
        )
        effective, _ = apply_target_lock(
            state,
            DetectionResult(sky_fp, 1, sky_fp.confidence),
            center_y=540.0,
            cfg=cfg,
            fov_cx=640.0,
            fov_cy=540.0,
            frame_size=(1280, 1080),
        )
        self.assertIs(effective.target, locked)
        self.assertEqual(state.locked_target, locked)

    def test_safe_track_allows_strafe_past_soft_drift_cap(self) -> None:
        state = TargetLockState()
        cfg = _cfg()
        cfg["_runtime_detect_fov"] = 200
        locked = _body(centroid_x=640.0, centroid_y=420.0)
        state.locked_target = locked
        state.target_lost_frames = 0
        shifted = _body(centroid_x=662.0, centroid_y=420.0, bbox_x=628, bbox_y=350)
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
        self.assertAlmostEqual(effective.target.centroid_x, 662.0, delta=0.5)

    def test_stale_sky_lock_purged_during_grace(self) -> None:
        state = TargetLockState()
        cfg = _cfg()
        locked = _body(
            centroid_y=120.0,
            bbox_y=40,
            bbox_h=80,
            red_coverage=0.02,
        )
        state.locked_target = locked
        state.target_lost_frames = 3
        effective, _ = apply_target_lock(
            state,
            DetectionResult(None, 0, 0.0),
            center_y=540.0,
            cfg=cfg,
            frame_size=(1280, 1080),
        )
        self.assertIsNone(effective.target)
        self.assertIsNone(state.locked_target)

    def test_overlay_confirm_resets_on_upward_sky_jump(self) -> None:
        from target_lock import overlay_may_show_target

        state = TargetLockState()
        t1 = _body(centroid_y=400.0, bbox_y=300, bbox_h=140)
        t2 = _body(centroid_y=180.0, bbox_y=40, bbox_h=80)
        overlay_may_show_target(
            t1, detection_fresh=True, center_y=540.0, lock_state=state
        )
        overlay_may_show_target(
            t1, detection_fresh=True, center_y=540.0, lock_state=state
        )
        self.assertGreaterEqual(state.overlay_confirm_frames, 2)
        overlay_may_show_target(
            t2, detection_fresh=True, center_y=540.0, lock_state=state
        )
        self.assertLess(state.overlay_confirm_frames, 2)
