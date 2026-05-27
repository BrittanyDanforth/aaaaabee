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
