"""Config normalize must fix CV detection + leftover ApexAimBot stack keys."""

from __future__ import annotations

from config_pipeline import (
    LEAVE_YOLO_STACK_PATCH,
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
