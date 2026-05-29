"""Config normalize must fix CV detection + leftover ApexAimBot stack keys."""

from __future__ import annotations

from config_pipeline import (
    ENTER_YOLO_STACK_PATCH,
    LEAVE_YOLO_STACK_PATCH,
    merge_enter_yolo_stack,
    merge_leave_yolo_stack,
    normalize_app_config,
    reconcile_detection_stack,
)


def test_reconcile_cv_detection_with_apexaimbot_pull() -> None:
    raw = {
        "profile": "apexaimbot",
        "detection_mode": "apex",
        "pull_mode": "apexaimbot_pid",
        "mouse_backend": "apexaimbot",
    }
    out = reconcile_detection_stack(raw)
    assert out["pull_mode"] == "aba"
    assert out["mouse_backend"] == "auto"
    assert out["profile"] == "apex_style_live_trace"


def test_normalize_yolo_shipped_defaults_unchanged() -> None:
    cfg = normalize_app_config(
        {
            "profile": "apexaimbot",
            "detection_mode": "yolo",
            "pull_mode": "apexaimbot_pid",
            "mouse_backend": "apexaimbot",
        }
    )
    assert cfg["detection_mode"] == "yolo"
    assert cfg["pull_mode"] == "apexaimbot_pid"
    assert cfg["profile"] == "apexaimbot"


def test_merge_leave_yolo_stack_on_preset_patch() -> None:
    patch = merge_leave_yolo_stack({"detection_mode": "apex", "pull_strength": 0.5})
    for key, val in LEAVE_YOLO_STACK_PATCH.items():
        assert patch[key] == val


def test_merge_leave_yolo_stack_skips_yolo() -> None:
    patch = merge_leave_yolo_stack(
        {"detection_mode": "yolo", "pull_mode": "apexaimbot_pid"}
    )
    assert patch["pull_mode"] == "apexaimbot_pid"


def test_merge_leave_yolo_stack_tuning_only_does_not_unwind() -> None:
    patch = merge_leave_yolo_stack({"pull_strength": 0.5, "deadzone_pixels": 3})
    assert "detection_mode" not in patch
    assert patch.get("pull_mode") != "aba"


def test_reconcile_yolo_incomplete_stack() -> None:
    raw = {
        "profile": "apex_style_live_trace",
        "detection_mode": "yolo",
        "pull_mode": "aba",
        "humanoid_min_height_pixels": 60,
    }
    out = reconcile_detection_stack(raw)
    assert out["pull_mode"] == "apexaimbot_pid"
    assert out["profile"] == "apexaimbot"
    assert out["humanoid_min_height_pixels"] == 0


def test_merge_enter_yolo_stack() -> None:
    patch = merge_enter_yolo_stack({"detection_mode": "yolo"})
    assert patch["pull_mode"] == ENTER_YOLO_STACK_PATCH["pull_mode"]


def test_normalize_yolo_only_detection_mode() -> None:
    cfg = normalize_app_config(
        {"profile": "apexaimbot", "detection_mode": "yolo", "allow_live_mouse": True}
    )
    assert cfg["detection_mode"] == "yolo"
    assert cfg["pull_mode"] == "apexaimbot_pid"
    assert cfg["humanoid_min_height_pixels"] == 0
