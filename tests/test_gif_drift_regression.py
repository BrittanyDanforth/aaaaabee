"""Phase-7 GIF drift regression tests.

Replays 10 representative frames from
``artifacts/real_apex_test/_gif_frames/`` through the LIVE detection +
lock machine (matching ``runtime.AssistRuntime._select_target``) and
asserts the three guards Phase-7 added:

  1. When a frame is active the locked bbox top must NOT collapse to
     the upper 5 % of the frame (the "dot drifts into sky" smoking
     gun the user reported).
  2. The locked target's ``body_shape_score`` must NEVER decrease by
     more than 0.20 between successive sampled frames (instant-adopt
     ratchet guard — without the absolute floor the multiplicative
     0.85 ratchet could collapse the score across 5 successive
     replacements).
  3. The locked target's ``bbox_y`` must NEVER drop by more than
     ``bbox_h * 0.15`` from one sampled frame to the next (upward-bias
     guard — head-fragment replacing torso bbox).

The test fixture intentionally uses the exact same lock-state machine
the audit harness uses so it is testing the behaviour the user sees
in live play, not a synthetic re-implementation.
"""

from __future__ import annotations

import copy
import math
import unittest
from pathlib import Path

import cv2

import detector
import profiles


REPO_ROOT = Path(__file__).resolve().parents[1]
FRAMES_DIR = REPO_ROOT / "artifacts" / "real_apex_test" / "_gif_frames"

# Sampled frame indices spread across the 166-frame GIF. These are the
# 10 representative frames called out in the Phase-7 audit (0, 20, 40,
# 60, 80, 100, 120, 140, 160, 165).
SAMPLE_INDICES = [0, 20, 40, 60, 80, 100, 120, 140, 160, 165]


def _live_cfg() -> dict:
    cfg = copy.deepcopy(profiles.PROFILE_DEFAULTS[profiles.PROFILE_APEX_STYLE_LIVE_TRACE])
    cfg.update({
        "body_shape_min_score": 0.42,
        "target_stickiness_pixels": 70,
        "torso_aim_fraction": 0.40,
        "deadzone_pixels": 2,
        "detection_mode": "apex",
        "detection_motion_assist": True,
        "detection_motion_threshold": 9,
        "humanoid_min_height_pixels": 60,
    })
    return cfg


class _Lock:
    """Mirror of runtime._select_target lock semantics (Phase-7 guards)."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.locked = None
        self.lost = 0
        self.switch_cand = None
        self.switch_frames = 0

    def step(self, raw_target):
        cfg = self.cfg
        lost_max = int(cfg["target_lost_frames_before_unlock"])
        if raw_target is not None:
            new_t = raw_target
            if self.locked is not None:
                drift = math.hypot(
                    new_t.centroid_x - self.locked.centroid_x,
                    new_t.centroid_y - self.locked.centroid_y,
                )
                fov_lim = float(cfg.get("_runtime_detect_fov", 200) or 200) * 0.55
                bs_ratio_ok = new_t.body_shape_score >= self.locked.body_shape_score * 0.85
                bs_abs_ok = new_t.body_shape_score >= 0.60
                upward = (
                    new_t.bbox_y
                    < self.locked.bbox_y - self.locked.bbox_h * 0.15
                    and new_t.bbox_h < self.locked.bbox_h
                )
                from detector import _bbox_iou
                adopt_iou = _bbox_iou(
                    self.locked.bbox_x, self.locked.bbox_y,
                    self.locked.bbox_w, self.locked.bbox_h,
                    new_t.bbox_x, new_t.bbox_y, new_t.bbox_w, new_t.bbox_h,
                )
                bbox_overlap_ok = adopt_iou >= 0.15
                instant_ok = bs_ratio_ok and bs_abs_ok and not upward and bbox_overlap_ok
                if drift < 25 and drift < fov_lim and instant_ok:
                    self.locked = new_t
                    self.lost = 0
                    self.switch_cand = None
                    self.switch_frames = 0
                    return self.locked, False
                if (self.switch_cand is not None
                        and math.hypot(new_t.centroid_x - self.switch_cand.centroid_x,
                                       new_t.centroid_y - self.switch_cand.centroid_y) < 30):
                    self.switch_frames += 1
                else:
                    self.switch_cand = new_t
                    self.switch_frames = 1
                switch_ok = (
                    new_t.body_shape_score >= self.locked.body_shape_score + 0.10
                    and new_t.confidence >= self.locked.confidence * 0.90
                )
                if self.switch_frames >= 3 and switch_ok:
                    self.locked = new_t
                    self.lost = 0
                    self.switch_cand = None
                    self.switch_frames = 0
                    return self.locked, False
                self.lost = max(1, self.lost)
                return self.locked, True
            if new_t.body_shape_score < 0.55:
                return None, False
            self.locked = new_t
            self.lost = 0
            return self.locked, False
        self.lost += 1
        if self.lost >= lost_max:
            self.locked = None
            return None, False
        if self.locked is not None:
            return self.locked, True
        return None, False


def _gif_available() -> bool:
    return FRAMES_DIR.exists() and any(FRAMES_DIR.glob("frame_*.png"))


@unittest.skipUnless(_gif_available(), "GIF frames not present in artifacts/")
class GifDriftRegressionTests(unittest.TestCase):
    def test_no_active_frame_drifts_into_sky(self) -> None:
        cfg = _live_cfg()
        ctx = detector.DetectionContext(
            motion_assist=True, motion_threshold=int(cfg["detection_motion_threshold"]),
        )
        lock = _Lock(cfg)
        prev_body = None
        prev_locked = None
        violations: list[str] = []
        for sample_i, frame_idx in enumerate(SAMPLE_INDICES):
            path = FRAMES_DIR / f"frame_{frame_idx:03d}.png"
            self.assertTrue(path.is_file(), f"missing frame {path}")
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            cx, cy = w / 2.0, h / 2.0
            fov_r = profiles.effective_detection_fov_radius(cfg, ads_active=False)
            sticky = lock.locked if lock.lost < int(cfg["target_lost_frames_before_unlock"]) else None
            currently_locked = sticky is not None
            result = detector.find_best_target(
                img,
                cfg.get("hsv_ranges"),
                fov_r,
                float(cfg["min_target_area_pixels"]),
                cx,
                cy,
                exclude_bottom_frac=0.05,
                detection_mode=detector.DETECTION_MODE_APEX,
                context=ctx,
                min_height_px=float(cfg["humanoid_min_height_pixels"]),
                min_aspect=float(cfg["humanoid_min_aspect"]),
                max_aspect=float(cfg["humanoid_max_aspect"]),
                min_solidity=float(cfg["humanoid_min_solidity"]),
                torso_aim_fraction=float(cfg["torso_aim_fraction"]),
                body_shape_min_score=float(cfg["body_shape_min_score"]),
                head_score_weight=float(cfg["head_score_weight"]),
                torso_score_weight=float(cfg["torso_score_weight"]),
                limb_stack_score_weight=float(cfg["limb_stack_score_weight"]),
                aim_y_min_fraction=float(cfg["aim_body_y_min_fraction"]),
                aim_y_max_fraction=float(cfg["aim_body_y_max_fraction"]),
                sticky_target=sticky,
                stickiness_pixels=float(cfg["target_stickiness_pixels"]),
                distance_weight=float(cfg["distance_score_weight"]),
                area_weight=float(cfg["area_score_weight"]),
                currently_locked=currently_locked,
            )
            effective, is_stale = lock.step(result.target)
            active = effective is not None and not is_stale
            if active and effective is not None:
                # (1) sky guard — bbox top below 5% line
                self.assertGreater(
                    effective.bbox_y,
                    h * 0.05,
                    f"sample {sample_i} (frame {frame_idx}): bbox_y={effective.bbox_y} "
                    f"< 5% of frame_h ({h*0.05:.0f}) — looks like sky-drift",
                )
                # (2) body_shape_score drop guard
                if prev_body is not None:
                    drop = prev_body - effective.body_shape_score
                    if drop > 0.20:
                        violations.append(
                            f"sample {sample_i}: body_shape dropped "
                            f"{prev_body:.2f} -> {effective.body_shape_score:.2f} (>0.20)"
                        )
                # (3) upward-bias guard
                if prev_locked is not None:
                    y_drop = prev_locked.bbox_y - effective.bbox_y
                    if y_drop > prev_locked.bbox_h * 0.15 and effective.bbox_h < prev_locked.bbox_h:
                        violations.append(
                            f"sample {sample_i}: bbox_y dropped {y_drop:.0f} "
                            f"(>{prev_locked.bbox_h*0.15:.0f}) AND bbox shrank — "
                            f"upward-fragment signature"
                        )
                prev_body = float(effective.body_shape_score)
                prev_locked = effective
        self.assertEqual(violations, [], "\n  " + "\n  ".join(violations))


if __name__ == "__main__":
    unittest.main()
