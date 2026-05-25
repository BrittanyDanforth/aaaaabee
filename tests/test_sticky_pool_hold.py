"""sticky_pool_hold must not force stale overlay when lock geometry is unchanged."""

from __future__ import annotations

import copy
import unittest

import profiles
from detector import DetectionResult, Target
from profiles import PROFILE_APEX_STYLE_LIVE_TRACE
from target_lock import TargetLockState, apply_target_lock

PROFILE = PROFILE_APEX_STYLE_LIVE_TRACE


def _target(*, dist: float, y: int, h: int) -> Target:
    return Target(
        centroid_x=400.0,
        centroid_y=float(y + h // 2),
        area=2000.0,
        distance_to_center=dist,
        confidence=0.75,
        bbox_x=370,
        bbox_y=y,
        bbox_w=60,
        bbox_h=h,
        body_shape_score=0.72,
        part_count=4,
        red_coverage=0.12,
        fill_ratio=0.5,
        max_circularity=0.5,
        has_classified_torso=True,
        head_score=0.5,
        torso_score=0.5,
        limb_stack_score=0.4,
    )


class StickyPoolHoldTests(unittest.TestCase):
    def test_pool_hold_clears_lost_frames_not_stale(self) -> None:
        cfg = copy.deepcopy(profiles.PROFILE_DEFAULTS[PROFILE])
        cfg["new_lock_confirm_frames"] = 1
        cfg["_runtime_overlay_fov"] = 140
        cfg["_runtime_detect_fov"] = 180
        state = TargetLockState()
        locked = _target(dist=55.0, y=200, h=90)
        state.locked_target = locked
        state.target_lost_frames = 4
        raw = DetectionResult(
            locked,
            3,
            locked.confidence,
            debug_lines=["sticky_pool_hold dist=55 h=90"],
            active=False,
        )
        eff, stale = apply_target_lock(
            state,
            raw,
            center_y=225.0,
            cfg=cfg,
            fov_cx=400.0,
            fov_cy=225.0,
            frame_size=(800, 450),
        )
        self.assertFalse(stale, "pool hold should be a live lock continuation")
        self.assertEqual(0, state.target_lost_frames)
        self.assertIs(eff.target, locked)


if __name__ == "__main__":
    unittest.main()
