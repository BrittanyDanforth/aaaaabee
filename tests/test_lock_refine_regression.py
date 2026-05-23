"""Regression: weak lock refine must not adopt upward fragments on movers."""

from __future__ import annotations

import copy
import unittest

import profiles
from detector import DetectionResult, Target
from target_lock import TargetLockState, apply_target_lock


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


class LockRefineRegressionTests(unittest.TestCase):
    def test_upward_fragment_not_adopted_on_fresh_lock(self) -> None:
        state = TargetLockState()
        locked = _body(centroid_y=420.0, bbox_y=300, bbox_h=140)
        state.locked_target = locked
        state.target_lost_frames = 0
        # Overlapping head fragment above torso — high IoU, weak refine used to steal.
        fragment = _body(
            centroid_x=642.0,
            centroid_y=360.0,
            bbox_y=locked.bbox_y - 50,
            bbox_h=80,
            body_shape_score=0.78,
            red_coverage=0.10,
        )
        effective, _ = apply_target_lock(
            state,
            DetectionResult(fragment, 1, fragment.confidence),
            center_y=540.0,
            cfg=_cfg(),
            fov_cx=640.0,
            fov_cy=540.0,
            frame_size=(1280, 1080),
        )
        self.assertIs(effective.target, locked)
        self.assertEqual(state.locked_target.centroid_y, locked.centroid_y)


if __name__ == "__main__":
    unittest.main()
