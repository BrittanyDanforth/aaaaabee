"""sticky_pool_hold contract: preserves lock geometry but counts toward grace.

Earlier design reset ``target_lost_frames=0`` on every pool_hold, which made
any transient initial lock on a structural FP (gun rim, banner edge, tower
icon, cable, score-panel highlight) become an indefinite phantom — the same
shape kept getting recycled every frame and the grace counter never advanced
(F62-F77 banner phantom, F102-F134 right-rim phantom in gif_166_proof).

New contract:
  - pool_hold preserves the lock geometry (eff.target is the prior locked
    target) so a brief detection gap doesn't drop a real enemy
  - pool_hold INCREMENTS target_lost_frames so the natural 18-frame grace
    expires when the same sticky shape keeps regenerating without any
    candidate IoU-matching it across many frames
  - the overlay dot stays LIVE for up to POOL_HOLD_DOT_ACTIVE_MAX (8)
    consecutive pool_hold frames, then goes STALE to prevent a phantom
    dot from sitting on a structural FP for the full grace window
"""

from __future__ import annotations

import copy
import unittest

import profiles
from detector import DetectionResult, Target
from profiles import PROFILE_APEX_STYLE_LIVE_TRACE
from target_lock import (
    POOL_HOLD_DOT_ACTIVE_MAX,
    TargetLockState,
    apply_target_lock,
)

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


def _pool_hold_result(locked: Target, *, candidates: int) -> DetectionResult:
    return DetectionResult(
        locked,
        candidates,
        locked.confidence,
        debug_lines=["sticky_pool_hold dist=55 h=90"],
        active=False,
    )


def _make_cfg() -> dict:
    cfg = copy.deepcopy(profiles.PROFILE_DEFAULTS[PROFILE])
    cfg["new_lock_confirm_frames"] = 1
    cfg["_runtime_overlay_fov"] = 140
    cfg["_runtime_detect_fov"] = 180
    return cfg


class StickyPoolHoldTests(unittest.TestCase):
    def test_pool_hold_preserves_lock_and_increments_grace(self) -> None:
        cfg = _make_cfg()
        state = TargetLockState()
        locked = _target(dist=55.0, y=200, h=90)
        state.locked_target = locked
        state.target_lost_frames = 4
        eff, stale = apply_target_lock(
            state, _pool_hold_result(locked, candidates=3),
            center_y=225.0, cfg=cfg,
            fov_cx=400.0, fov_cy=225.0, frame_size=(800, 450),
        )
        self.assertFalse(stale, "pool_hold is not stale grace — it's a hold continuation")
        self.assertIs(eff.target, locked, "lock geometry must be preserved")
        self.assertEqual(
            5, state.target_lost_frames,
            "pool_hold must increment lost_frames so phantom locks expire",
        )

    def test_pool_hold_dot_active_within_cap(self) -> None:
        """First N pool_hold frames keep the overlay dot LIVE (active=True)."""
        cfg = _make_cfg()
        state = TargetLockState()
        locked = _target(dist=55.0, y=200, h=90)
        state.locked_target = locked
        state.target_lost_frames = 0
        eff, _ = apply_target_lock(
            state, _pool_hold_result(locked, candidates=3),
            center_y=225.0, cfg=cfg,
            fov_cx=400.0, fov_cy=225.0, frame_size=(800, 450),
        )
        self.assertTrue(eff.active, "early pool_hold frame must keep dot LIVE")
        self.assertEqual(1, state.target_lost_frames)

    def test_pool_hold_dot_goes_stale_beyond_cap(self) -> None:
        """Past POOL_HOLD_DOT_ACTIVE_MAX, dot is hidden but lock is preserved."""
        cfg = _make_cfg()
        state = TargetLockState()
        locked = _target(dist=55.0, y=200, h=90)
        state.locked_target = locked
        state.target_lost_frames = POOL_HOLD_DOT_ACTIVE_MAX
        eff, _ = apply_target_lock(
            state, _pool_hold_result(locked, candidates=3),
            center_y=225.0, cfg=cfg,
            fov_cx=400.0, fov_cy=225.0, frame_size=(800, 450),
        )
        self.assertFalse(
            eff.active,
            f"after {POOL_HOLD_DOT_ACTIVE_MAX} pool_hold frames the dot must hide",
        )
        self.assertIs(
            eff.target, locked,
            "lock geometry stays preserved even when dot is hidden",
        )

    def test_pool_hold_expires_lock_at_lost_max(self) -> None:
        """When grace fully expires, the lock is reset (state.locked_target = None)."""
        cfg = _make_cfg()
        lost_max = int(cfg["target_lost_frames_before_unlock"])
        state = TargetLockState()
        locked = _target(dist=55.0, y=200, h=90)
        state.locked_target = locked
        state.target_lost_frames = lost_max - 1
        eff, _ = apply_target_lock(
            state, _pool_hold_result(locked, candidates=3),
            center_y=225.0, cfg=cfg,
            fov_cx=400.0, fov_cy=225.0, frame_size=(800, 450),
        )
        self.assertIsNone(eff.target, "expired lock must drop the target")
        self.assertIsNone(state.locked_target, "state must reset at grace expiry")
        self.assertEqual(0, state.target_lost_frames)


if __name__ == "__main__":
    unittest.main()
