"""YOLO pure-path helpers (no full runtime loop)."""

from __future__ import annotations

from detector import DetectionResult, Target
from profiles import PROFILE_APEXAIMBOT, PROFILE_DEFAULTS, effective_yolo_grab_half
from target_lock import TargetLockState, apply_yolo_target_lock, may_assist_pull_target_yolo


def test_profile_apexaimbot_has_yolo_keys() -> None:
    p = PROFILE_DEFAULTS[PROFILE_APEXAIMBOT]
    assert p["detection_mode"] == "yolo"
    assert p["pull_mode"] == "apexaimbot_pid"
    assert p["yolo_skip_motion_smooth"] is True
    assert p["apex_pid_subtick_hz"] == 120
    assert p["yolo_pull_stale_grace_frames"] == 12
    assert p["yolo_switch_confirm_frames"] == 2


def test_yolo_grab_half_is_208_for_416() -> None:
    cfg = PROFILE_DEFAULTS[PROFILE_APEXAIMBOT]
    assert effective_yolo_grab_half(cfg) == 208


def test_may_assist_pull_yolo_no_cv_plausible() -> None:
    t = Target(100.0, 100.0, 10.0, 5.0, 0.1)  # terrible CV scores
    assert may_assist_pull_target_yolo(t, detection_fresh=True)


def test_apply_yolo_lock_resets_on_expiry() -> None:
    state = TargetLockState()
    t = Target(100.0, 100.0, 500.0, 10.0, 0.9)
    cfg = {"target_lost_frames_before_unlock": 2}
    apply_yolo_target_lock(state, DetectionResult(t, 1, 0.9), cfg=cfg)
    apply_yolo_target_lock(
        state, DetectionResult(None, 0, 0.0), cfg=cfg
    )
    for _ in range(3):
        apply_yolo_target_lock(
            state, DetectionResult(None, 0, 0.0), cfg=cfg
        )
    assert state.locked_target is None


def _yt(x: float, *, conf: float = 0.9) -> Target:
    return Target(
        x,
        208.0,
        5000.0,
        abs(x - 208.0),
        conf,
        bbox_x=int(x - 30),
        bbox_y=148,
        bbox_w=60,
        bbox_h=120,
    )


def test_yolo_lock_holds_far_one_frame_transient_during_stale() -> None:
    state = TargetLockState()
    cfg = {
        "target_lost_frames_before_unlock": 18,
        "yolo_switch_reset_pixels": 80.0,
        "yolo_switch_confirm_frames": 2,
    }
    apply_yolo_target_lock(state, DetectionResult(_yt(208.0), 1, 0.9), cfg=cfg)
    apply_yolo_target_lock(state, DetectionResult(None, 0, 0.0), cfg=cfg)
    eff, stale = apply_yolo_target_lock(
        state, DetectionResult(_yt(330.0, conf=0.55), 1, 0.55), cfg=cfg
    )
    assert stale is True
    assert eff.target is not None
    assert eff.target.centroid_x == 208.0
    assert state.switch_frames == 1
    assert any("yolo_switch_pending" in ln for ln in (eff.debug_lines or []))


def test_yolo_lock_confirms_persistent_far_reacquire() -> None:
    state = TargetLockState()
    cfg = {
        "target_lost_frames_before_unlock": 18,
        "yolo_switch_reset_pixels": 80.0,
        "yolo_switch_confirm_frames": 2,
    }
    apply_yolo_target_lock(state, DetectionResult(_yt(208.0), 1, 0.9), cfg=cfg)
    apply_yolo_target_lock(state, DetectionResult(None, 0, 0.0), cfg=cfg)
    apply_yolo_target_lock(
        state, DetectionResult(_yt(330.0, conf=0.55), 1, 0.55), cfg=cfg
    )
    eff, stale = apply_yolo_target_lock(
        state, DetectionResult(_yt(332.0, conf=0.56), 1, 0.56), cfg=cfg
    )
    assert stale is False
    assert eff.target is not None
    assert eff.target.centroid_x == 332.0
    assert state.target_lost_frames == 0


def test_yolo_pull_stale_grace_covers_mid_dropout() -> None:
    t = _yt(208.0)
    assert may_assist_pull_target_yolo(
        t, detection_fresh=False, target_lost_frames=12, stale_grace_frames=12
    )
    assert not may_assist_pull_target_yolo(
        t, detection_fresh=False, target_lost_frames=13, stale_grace_frames=12
    )
