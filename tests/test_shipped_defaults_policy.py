"""Shipped config.json must pass ban policy and normalize to YOLO stack."""

from __future__ import annotations

from pathlib import Path

from ban_safety import validate_runtime_policy
from config_pipeline import load_app_config, normalize_app_config
from profiles import is_yolo_detection, uses_apex_pid_pull

REPO = Path(__file__).resolve().parents[1]


def test_shipped_config_json_passes_live_policy() -> None:
    cfg = load_app_config(REPO / "config.json")
    ok, msg = validate_runtime_policy(cfg)
    assert ok, msg
    assert cfg.get("profile") == "apexaimbot"
    assert is_yolo_detection(cfg)
    assert uses_apex_pid_pull(cfg)


def test_normalize_fixes_cv_with_leftover_apex_stack() -> None:
    cfg = normalize_app_config(
        {
            "detection_mode": "apex",
            "profile": "apexaimbot",
            "pull_mode": "apexaimbot_pid",
            "mouse_backend": "apexaimbot",
            "allow_live_mouse": False,
        }
    )
    assert cfg["pull_mode"] == "aba"
    assert cfg["profile"] == "apex_style_live_trace"
