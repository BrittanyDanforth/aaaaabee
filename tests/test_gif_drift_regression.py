"""Phase-7 GIF drift regression tests.

Replays 10 representative frames from
``artifacts/real_apex_test/_gif_frames/`` through the LIVE detection +
lock machine (``target_lock.apply_target_lock`` — same code as runtime)
and asserts the guards that prevent sky-drift and upward fragments.
"""

from __future__ import annotations

import copy
import unittest
from pathlib import Path

import cv2

import detector
import motion as motion_mod
import profiles
from motion import TargetTracker
from target_lock import (
    TargetLockMachine,
    detection_sticky_context,
    viewmodel_exclude_bottom,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FRAMES_DIR = REPO_ROOT / "artifacts" / "real_apex_test" / "_gif_frames"

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


def _gif_available() -> bool:
    return FRAMES_DIR.exists() and any(FRAMES_DIR.glob("frame_*.png"))


@unittest.skipUnless(_gif_available(), "GIF frames not present in artifacts/")
class GifDriftRegressionTests(unittest.TestCase):
    def test_no_active_frame_drifts_into_sky(self) -> None:
        cfg = _live_cfg()
        ctx = detector.DetectionContext(
            motion_assist=True, motion_threshold=int(cfg["detection_motion_threshold"]),
        )
        lock = TargetLockMachine(cfg, center_y=360.0)
        prev_body = None
        prev_locked = None
        violations: list[str] = []
        for sample_i, frame_idx in enumerate(SAMPLE_INDICES):
            path = FRAMES_DIR / f"frame_{frame_idx:03d}.png"
            self.assertTrue(path.is_file(), f"missing frame {path}")
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            cx, cy = w / 2.0, h / 2.0
            lock.center_y = cy
            fov_r = profiles.effective_detection_fov_radius(cfg, ads_active=False)
            cfg["_runtime_detect_fov"] = float(fov_r)
            sticky, currently_locked, _ = detection_sticky_context(lock.state, cfg)
            result = detector.find_best_target(
                img,
                cfg.get("hsv_ranges"),
                fov_r,
                float(cfg["min_target_area_pixels"]),
                cx,
                cy,
                exclude_bottom_frac=viewmodel_exclude_bottom(cfg),
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
            merged, is_stale = lock.step_detection(result)
            effective = merged.target
            active = effective is not None and not is_stale
            if active and effective is not None:
                self.assertGreater(
                    effective.bbox_y,
                    h * 0.05,
                    f"sample {sample_i} (frame {frame_idx}): bbox_y={effective.bbox_y} "
                    f"< 5% of frame_h ({h*0.05:.0f}) — looks like sky-drift",
                )
                if prev_body is not None:
                    drop = prev_body - effective.body_shape_score
                    if drop > 0.20:
                        violations.append(
                            f"sample {sample_i}: body_shape dropped "
                            f"{prev_body:.2f} -> {effective.body_shape_score:.2f} (>0.20)"
                        )
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

    def test_gif_motion_overlay_stays_inside_body_bbox(self) -> None:
        cfg = _live_cfg()
        ctx = detector.DetectionContext(
            motion_assist=True,
            motion_threshold=int(cfg["detection_motion_threshold"]),
        )
        lock = TargetLockMachine(cfg, center_y=360.0)
        tracker = TargetTracker()
        outside: list[str] = []
        for frame_idx in SAMPLE_INDICES:
            path = FRAMES_DIR / f"frame_{frame_idx:03d}.png"
            img = __import__("cv2").imread(str(path), __import__("cv2").IMREAD_COLOR)
            h, w = img.shape[:2]
            cx, cy = w / 2.0, h / 2.0
            lock.center_y = cy
            fov_r = profiles.effective_detection_fov_radius(cfg, ads_active=False)
            sticky, currently_locked, _ = detection_sticky_context(lock.state, cfg)
            result = detector.find_best_target(
                img,
                cfg.get("hsv_ranges"),
                fov_r,
                float(cfg["min_target_area_pixels"]),
                cx,
                cy,
                exclude_bottom_frac=viewmodel_exclude_bottom(cfg),
                detection_mode=detector.DETECTION_MODE_APEX,
                context=ctx,
                min_height_px=float(cfg["humanoid_min_height_pixels"]),
                sticky_target=sticky,
                stickiness_pixels=float(cfg["target_stickiness_pixels"]),
                currently_locked=currently_locked,
            )
            merged, is_stale = lock.step_detection(result)
            t = merged.target
            if t is None or is_stale:
                continue
            m = tracker.observe_target(
                t.centroid_x,
                t.centroid_y,
                frame_idx / 30.0,
                bbox_x=t.bbox_x,
                bbox_y=t.bbox_y,
                bbox_w=t.bbox_w,
                bbox_h=t.bbox_h,
                aim_is_body_anchor=True,
            )
            ox, oy = m.overlay_xy()
            y_hi = t.bbox_y + t.bbox_h * motion_mod._body_y_hi_frac
            if oy > y_hi + 3.0:
                outside.append(
                    f"frame {frame_idx}: overlay_y={oy:.0f} > chest_hi={y_hi:.0f}"
                )
        self.assertEqual(outside, [], "\n  " + "\n  ".join(outside))


if __name__ == "__main__":
    unittest.main()
