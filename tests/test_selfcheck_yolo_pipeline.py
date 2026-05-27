"""Self-check must validate production YOLO pipeline without torch."""

from __future__ import annotations

from config_pipeline import normalize_app_config
from profiles import PROFILE_APEXAIMBOT
from self_check import run_yolo_pipeline_check


def test_yolo_pipeline_check_passes_without_torch() -> None:
    cfg = normalize_app_config(
        {
            "profile": PROFILE_APEXAIMBOT,
            "detection_mode": "yolo",
            "pull_mode": "apexaimbot_pid",
        }
    )
    ok, lines, err = run_yolo_pipeline_check(cfg)
    assert ok, err
    assert any("yolo_detect_and_lock" in ln for ln in lines)


def test_yolo_pipeline_check_skips_cv_mode() -> None:
    cfg = normalize_app_config(
        {"profile": "apex_style_dry_run", "detection_mode": "apex", "pull_mode": "aba"}
    )
    ok, lines, err = run_yolo_pipeline_check(cfg)
    assert ok and err is None
    assert any("SKIP" in ln for ln in lines)
