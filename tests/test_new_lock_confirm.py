"""New-lock must confirm 2 frames before dot appears — stops ADS FP flash."""

from __future__ import annotations

import copy
import unittest

import profiles
from detector import DetectionResult, Target
from target_lock import (
    NEW_LOCK_CONFIRM_FRAMES,
    NEW_LOCK_MIN_BODY,
    TargetLockState,
    apply_target_lock,
)


def _make_target(**kw) -> Target:
    defaults = dict(
        centroid_x=400.0,
        centroid_y=300.0,
        area=4000.0,
        distance_to_center=20.0,
        confidence=0.85,
        bbox_x=370,
        bbox_y=200,
        bbox_w=60,
        bbox_h=120,
        body_shape_score=0.72,
        part_count=4,
        red_coverage=0.12,
        has_classified_torso=True,
    )
    defaults.update(kw)
    return Target(**defaults)


def _cfg() -> dict:
    cfg = copy.deepcopy(
        profiles.PROFILE_DEFAULTS[profiles.PROFILE_APEX_STYLE_LIVE_TRACE]
    )
    cfg["new_lock_confirm_frames"] = NEW_LOCK_CONFIRM_FRAMES
    return cfg


class NewLockConfirmTests(unittest.TestCase):
    def test_single_frame_fp_does_not_acquire(self) -> None:
        state = TargetLockState()
        cfg = _cfg()
        fp = _make_target(
            body_shape_score=NEW_LOCK_MIN_BODY + 0.02,
            has_classified_torso=True,
            torso_score=0.5,
            head_score=0.4,
            fill_ratio=0.5,
            max_circularity=0.45,
        )
        r1, stale1 = apply_target_lock(
            state,
            DetectionResult(fp, 1, fp.confidence),
            center_y=360.0,
            cfg=cfg,
        )
        self.assertIsNone(r1.target)
        self.assertFalse(stale1)
        self.assertIsNone(state.locked_target)
        self.assertEqual(state.new_lock_frames, 1)

    def test_two_matching_frames_acquire(self) -> None:
        state = TargetLockState()
        cfg = _cfg()
        fp = _make_target(
            has_classified_torso=True,
            torso_score=0.5,
            head_score=0.4,
            fill_ratio=0.5,
            max_circularity=0.45,
        )
        apply_target_lock(
            state,
            DetectionResult(fp, 1, fp.confidence),
            center_y=360.0,
            cfg=cfg,
        )
        fp2 = _make_target(centroid_x=402.0, centroid_y=301.0)
        r2, _ = apply_target_lock(
            state,
            DetectionResult(fp2, 1, fp2.confidence),
            center_y=360.0,
            cfg=cfg,
        )
        self.assertIsNotNone(state.locked_target)
        self.assertIs(r2.target, state.locked_target)

    def test_weak_body_never_confirms(self) -> None:
        state = TargetLockState()
        cfg = _cfg()
        weak = _make_target(body_shape_score=0.52, part_count=4)
        for _ in range(3):
            apply_target_lock(
                state,
                DetectionResult(weak, 0, 0.5),
                center_y=360.0,
                cfg=cfg,
            )
        self.assertIsNone(state.locked_target)


if __name__ == "__main__":
    unittest.main()
