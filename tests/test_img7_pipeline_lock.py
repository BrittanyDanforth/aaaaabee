"""img7 + gun-column FPs must not survive detector → lock → overlay."""

from __future__ import annotations

import copy
from pathlib import Path

import cv2
import pytest

import detector
import profiles
from detector import DetectionResult
from target_lock import (
    OVERLAY_CONFIRM_FRAMES,
    TargetLockState,
    apply_target_lock,
    overlay_may_show_target,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
IMG7 = REPO_ROOT / "artifacts" / "real_apex_test" / "_inputs" / "img7_sky_dot_bug.png"


def _live_cfg() -> dict:
    cfg = copy.deepcopy(
        profiles.PROFILE_DEFAULTS[profiles.PROFILE_APEX_STYLE_LIVE_TRACE]
    )
    cfg.update(
        {
            "body_shape_min_score": 0.42,
            "target_stickiness_pixels": 70,
            "detection_motion_assist": True,
            "detection_motion_threshold": 9,
            "humanoid_min_height_pixels": 60,
            "new_lock_confirm_frames": 2,
        }
    )
    return cfg


def _shift(frame, dx: int):
    import numpy as np

    h, w = frame.shape[:2]
    out = np.zeros_like(frame)
    if dx >= 0:
        out[:, dx:] = frame[:, : w - dx]
    else:
        out[:, : w + dx] = frame[:, -dx:]
    return out


def _find(img, cfg, ctx, *, sticky=None, currently_locked=False):
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    fov_r = profiles.effective_detection_fov_radius(cfg, ads_active=False)
    return detector.find_best_target(
        img,
        cfg.get("hsv_ranges"),
        fov_r,
        float(cfg["min_target_area_pixels"]),
        cx,
        cy,
        debug=True,
        exclude_bottom_frac=0.30,
        detection_mode=detector.DETECTION_MODE_APEX,
        context=ctx,
        sticky_target=sticky,
        currently_locked=currently_locked,
        stickiness_pixels=float(cfg["target_stickiness_pixels"]),
        body_shape_min_score=float(cfg["body_shape_min_score"]),
    )


def test_img7_full_pipeline_no_lock_or_dot() -> None:
    """Real img7: detector + frame lock + overlay confirm — all must stay off."""
    if not IMG7.exists():
        pytest.skip("img7 fixture missing")
    img = cv2.imread(str(IMG7), cv2.IMREAD_COLOR)
    assert img is not None
    h, w = img.shape[:2]
    cy = h / 2.0
    cfg = _live_cfg()
    ctx = detector.DetectionContext(
        motion_assist=True,
        motion_threshold=int(cfg["detection_motion_threshold"]),
    )
    panned = cv2.cvtColor(_shift(img, 6), cv2.COLOR_BGR2GRAY)
    ctx.prev_gray = panned
    ctx.prev_size = (w, h)

    lock_state = TargetLockState()
    for pass_idx in range(4):
        sticky, currently_locked, _ = __import__(
            "target_lock", fromlist=["detection_sticky_context"]
        ).detection_sticky_context(lock_state, cfg)
        det = _find(
            img,
            cfg,
            ctx,
            sticky=sticky,
            currently_locked=currently_locked,
        )
        det, _stale = apply_target_lock(
            lock_state,
            det,
            center_y=cy,
            cfg=cfg,
        )
        fresh = det.target is not None and lock_state.target_lost_frames == 0
        show = overlay_may_show_target(
            det.target,
            detection_fresh=fresh,
            center_y=cy,
            lock_state=lock_state,
        )
        assert det.target is None, f"pass {pass_idx}: lock target must stay None"
        assert lock_state.locked_target is None, f"pass {pass_idx}: no frame lock"
        assert not show, f"pass {pass_idx}: overlay must not show dot"
        assert lock_state.overlay_confirm_frames < OVERLAY_CONFIRM_FRAMES


def test_gun_column_fp_never_confirms_lock() -> None:
    """Synthetic img7-class column: high body score, near-zero red, sky band."""
    cfg = _live_cfg()
    center_y = 360.0
    gun_col = detector.Target(
        centroid_x=640.0,
        centroid_y=80.0,
        area=8000.0,
        distance_to_center=30.0,
        confidence=0.9,
        bbox_x=600,
        bbox_y=20,
        bbox_w=80,
        bbox_h=200,
        body_shape_score=0.88,
        head_score=0.95,
        torso_score=0.0,
        limb_stack_score=0.95,
        part_count=9,
        red_coverage=0.01,
        fill_ratio=0.55,
        max_circularity=0.5,
        has_classified_torso=False,
    )
    state = TargetLockState()
    for _ in range(4):
        det, _ = apply_target_lock(
            state,
            DetectionResult(gun_col, 1, gun_col.confidence),
            center_y=center_y,
            cfg=cfg,
        )
        assert det.target is None
        assert state.locked_target is None
        show = overlay_may_show_target(
            None,
            detection_fresh=False,
            center_y=center_y,
            lock_state=state,
        )
        assert not show


def test_artifact_img7_gun_column_is_viewmodel_fp() -> None:
    """Regression: audit img7 cand[2] (torso 0.64, red 0.011) is viewmodel FP."""
    h, w = 736, 1193
    cx, cy = w / 2.0, h / 2.0
    gun = detector.Target(
        centroid_x=535.525,
        centroid_y=369.077,
        area=3216.0,
        distance_to_center=61.0,
        confidence=0.85,
        bbox_x=485,
        bbox_y=269,
        bbox_w=106,
        bbox_h=246,
        body_shape_score=1.0,
        head_score=0.87,
        torso_score=0.64,
        limb_stack_score=1.0,
        part_count=14,
        red_coverage=0.011,
        fill_ratio=0.24,
        max_circularity=0.61,
    )
    assert detector.target_is_viewmodel_column_fp(
        gun, frame_w=w, frame_h=h, fov_cx=cx, fov_cy=cy
    )


def test_stale_low_red_lock_purged_on_grace() -> None:
    """Grace hold must drop when locked blob has no enemy-red (sky/HUD drift)."""
    cfg = _live_cfg()
    center_y = 360.0
    state = TargetLockState()
    bad = detector.Target(
        centroid_x=400.0,
        centroid_y=90.0,
        area=5000.0,
        distance_to_center=10.0,
        confidence=0.8,
        bbox_x=370,
        bbox_y=30,
        bbox_w=60,
        bbox_h=120,
        body_shape_score=0.75,
        part_count=3,
        red_coverage=0.02,
        fill_ratio=0.6,
        max_circularity=0.5,
    )
    state.locked_target = bad
    state.target_lost_frames = 2
    det, stale = apply_target_lock(
        state,
        DetectionResult(None, 0, 0.0),
        center_y=center_y,
        cfg=cfg,
        fov_cx=400.0,
        fov_cy=center_y,
        frame_size=(1280, 720),
    )
    assert state.locked_target is None
    assert det.target is None
    assert not stale
