"""Regression tests for gif_166_proof bbox-stability fixes.

These tests guard the user-reported F11/F29/F31/F55/F61 regressions that
were addressed by:
  * detector.target_is_central_tower_banner_fp — low-red high-frame band
  * target_lock._locked_bbox_explosion           — merge bbox swallows lock
  * target_lock.CONSECUTIVE_POOL_HOLD_HIDE_AT    — multi-frame ghost cap
  * target_lock.EXPLOSION_LOCK_EXPIRE_AT         — sustained-merge expiry

Failure of any of these tests indicates a real-frame regression that the
user can see as either a huge/wide green box (explosion), a stale red dot
sitting on empty space (pool-hold ghost), or a phantom lock-on at ADS
start over the central tower banner column.
"""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import profiles  # noqa: E402
from detector import (  # noqa: E402
    DetectionResult,
    Target,
    target_is_central_tower_banner_fp,
)
from target_lock import (  # noqa: E402
    CONSECUTIVE_POOL_HOLD_HIDE_AT,
    EXPLOSION_LOCK_EXPIRE_AT,
    TargetLockState,
    _locked_bbox_explosion,
    apply_target_lock,
)


def _cfg() -> dict:
    return copy.deepcopy(
        profiles.PROFILE_DEFAULTS[profiles.PROFILE_APEX_STYLE_LIVE_TRACE]
    )


def _body(**kw) -> Target:
    d = dict(
        centroid_x=400.0,
        centroid_y=215.0,
        area=1200.0,
        distance_to_center=15.0,
        confidence=0.85,
        bbox_x=386,
        bbox_y=200,
        bbox_w=28,
        bbox_h=30,
        body_shape_score=0.80,
        part_count=4,
        red_coverage=0.18,
        has_classified_torso=True,
        torso_score=0.62,
        head_score=0.55,
    )
    d.update(kw)
    return Target(**d)


class BboxExplosionGuardTests(unittest.TestCase):
    """Reject sudden bbox explosions that swallow the previous lock."""

    def test_horizontal_explosion_rejected(self) -> None:
        # F29-class: previous tight close dummy (24x43) suddenly becomes
        # a 90x59 wide bbox spanning the adjacent firing-range board.
        locked = _body(bbox_x=390, bbox_y=187, bbox_w=24, bbox_h=43)
        merged = _body(
            centroid_x=397.0, centroid_y=205.5,
            bbox_x=352, bbox_y=176, bbox_w=90, bbox_h=59,
            distance_to_center=20.0,
        )
        self.assertTrue(_locked_bbox_explosion(locked, merged))

    def test_vertical_explosion_rejected(self) -> None:
        # F55-class: previous (48x33) becomes (49x139) spanning the
        # central tower banner above the dummy.
        locked = _body(bbox_x=393, bbox_y=197, bbox_w=48, bbox_h=33)
        merged = _body(
            centroid_x=417.0, centroid_y=170.0,
            bbox_x=393, bbox_y=100, bbox_w=49, bbox_h=139,
            distance_to_center=58.0,
        )
        self.assertTrue(_locked_bbox_explosion(locked, merged))

    def test_gradual_close_approach_not_rejected(self) -> None:
        # F31-class: legitimate close-enemy bbox growth (24x43 -> 66x62)
        # must NOT trigger explosion (~2.75x width, ~1.4x height, ~3.97x
        # area — below the 2.8 width / 2.6 height / 3.8 area thresholds).
        locked = _body(bbox_x=390, bbox_y=187, bbox_w=24, bbox_h=43)
        grown = _body(
            centroid_x=384.0, centroid_y=229.0,
            bbox_x=351, bbox_y=198, bbox_w=66, bbox_h=62,
            distance_to_center=20.0,
        )
        self.assertFalse(_locked_bbox_explosion(locked, grown))

    def test_apply_lock_holds_locked_on_explosion(self) -> None:
        cfg = _cfg()
        state = TargetLockState()
        locked = _body(bbox_x=390, bbox_y=187, bbox_w=24, bbox_h=43)
        state.locked_target = locked
        state.target_lost_frames = 0
        merged = _body(
            centroid_x=397.0, centroid_y=205.5,
            bbox_x=352, bbox_y=176, bbox_w=90, bbox_h=59,
            distance_to_center=20.0,
        )
        eff, stale = apply_target_lock(
            state,
            DetectionResult(merged, 1, merged.confidence),
            center_y=225.0,
            cfg=cfg,
            fov_cx=400.0,
            fov_cy=225.0,
            frame_size=(800, 450),
        )
        self.assertIs(eff.target, locked, "first explosion holds lock")
        self.assertTrue(stale)
        self.assertEqual(state.explosion_reject_streak, 1)

    def test_sustained_explosion_drops_lock(self) -> None:
        cfg = _cfg()
        state = TargetLockState()
        locked = _body(bbox_x=390, bbox_y=187, bbox_w=24, bbox_h=43)
        state.locked_target = locked
        state.target_lost_frames = 0
        merged = _body(
            bbox_x=352, bbox_y=176, bbox_w=90, bbox_h=59,
            distance_to_center=20.0,
        )
        for _ in range(EXPLOSION_LOCK_EXPIRE_AT + 1):
            eff, _ = apply_target_lock(
                state,
                DetectionResult(merged, 1, merged.confidence),
                center_y=225.0,
                cfg=cfg,
                fov_cx=400.0,
                fov_cy=225.0,
                frame_size=(800, 450),
            )
        self.assertIsNone(state.locked_target, "lock dropped after sustained explosion")
        self.assertIsNone(eff.target)


class CentralTowerBannerLowRedTests(unittest.TestCase):
    """gif_166_proof F11-class: tall narrow column in upper frame band."""

    def test_f11_low_red_banner_rejected(self) -> None:
        # bbox (386,96,50x101) — chest at y=146 (frac 0.32), top_frac
        # 0.21, mid_y_frac 0.327, aspect 2.02, red_cov ~0.08.
        target = _body(
            centroid_x=411.0, centroid_y=146.0,
            bbox_x=386, bbox_y=96, bbox_w=50, bbox_h=101,
            body_shape_score=0.96, head_score=0.62, torso_score=0.79,
            red_coverage=0.080, part_count=6,
        )
        self.assertTrue(
            target_is_central_tower_banner_fp(
                target, frame_w=800, frame_h=450, fov_cx=400.0
            )
        )

    def test_real_close_dummy_at_center_not_rejected(self) -> None:
        # gif_166_proof F50 dummy (386,207,16x23) — chest at the
        # crosshair, mid_y_frac 0.486, red_cov 0.19 (filled body).
        target = _body(
            centroid_x=394.0, centroid_y=218.0,
            bbox_x=386, bbox_y=207, bbox_w=16, bbox_h=23,
            body_shape_score=0.92, head_score=0.40, torso_score=0.55,
            red_coverage=0.19, part_count=3,
        )
        self.assertFalse(
            target_is_central_tower_banner_fp(
                target, frame_w=800, frame_h=450, fov_cx=400.0
            )
        )


class PoolHoldStreakTests(unittest.TestCase):
    """Hide dot after CONSECUTIVE_POOL_HOLD_HIDE_AT synthesized frames."""

    def _pool_hold_result(self, t: Target) -> DetectionResult:
        return DetectionResult(
            t, 0, t.confidence,
            debug_lines=[f"sticky_pool_hold dist={t.distance_to_center:.0f} h={t.bbox_h}"],
            active=True,
        )

    def test_streak_hides_active_after_threshold(self) -> None:
        cfg = _cfg()
        state = TargetLockState()
        locked = _body(
            centroid_x=384.0, centroid_y=229.0,
            bbox_x=351, bbox_y=198, bbox_w=66, bbox_h=62,
            distance_to_center=21.0,
        )
        state.locked_target = locked
        state.target_lost_frames = 0
        last_active = None
        for _ in range(CONSECUTIVE_POOL_HOLD_HIDE_AT + 2):
            eff, _ = apply_target_lock(
                state,
                self._pool_hold_result(locked),
                center_y=225.0,
                cfg=cfg,
                fov_cx=400.0,
                fov_cy=225.0,
                frame_size=(800, 450),
            )
            last_active = eff.active
        self.assertFalse(
            last_active,
            "long ghost pool-hold streak should hide the active dot",
        )
        self.assertGreater(state.pool_hold_streak, CONSECUTIVE_POOL_HOLD_HIDE_AT)


class Gif166ProofPostfixSummary(unittest.TestCase):
    """End-to-end audit summary — proves the user-reported frames are clean."""

    def test_summary_passes_strict_metrics(self) -> None:
        summary_path = (
            REPO
            / "artifacts"
            / "real_apex_test"
            / "gif_166_proof"
            / "summary.json"
        )
        data = json.loads(summary_path.read_text(encoding="utf-8"))["summary"]
        self.assertEqual(data["sky_aim_violations"], 0)
        self.assertEqual(data["chest_band_violations"], 0)
        self.assertEqual(data["high_bbox_close_frames"], 0)
        self.assertTrue(data["pass_strict"])


if __name__ == "__main__":
    unittest.main()
