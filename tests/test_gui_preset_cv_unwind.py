"""Presets that return to CV detection must unwind ApexAimBot pull/mouse/profile."""

from __future__ import annotations

from unittest.mock import MagicMock

from config_pipeline import merge_leave_yolo_stack, normalize_app_config
from runtime_controller import RuntimeController


def test_stable_preset_unwinds_yolo_pull_stack() -> None:
    base = normalize_app_config(
        {
            "profile": "apexaimbot",
            "detection_mode": "yolo",
            "pull_mode": "apexaimbot_pid",
            "mouse_backend": "apexaimbot",
        }
    )
    ctrl = RuntimeController(base, MagicMock())
    preset = merge_leave_yolo_stack({"detection_mode": "apex", "pull_strength": 0.55})
    merged = ctrl.apply_config_patch(preset, persist=False)
    assert merged["detection_mode"] == "apex"
    assert merged["pull_mode"] == "aba"
    assert merged["mouse_backend"] == "auto"
    assert merged["profile"] == "apex_style_live_trace"
