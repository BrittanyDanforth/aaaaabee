"""Repeated YOLO ↔ CV flips must not leave stale engines, locks, or pull stack."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from config_pipeline import normalize_app_config
from runtime import AssistRuntime


def _yolo_cfg() -> dict:
    return normalize_app_config(
        {
            "profile": "apexaimbot",
            "detection_mode": "yolo",
            "pull_mode": "apexaimbot_pid",
            "mouse_backend": "apexaimbot",
            "allow_live_mouse": False,
        }
    )


def _cv_cfg() -> dict:
    return normalize_app_config(
        {
            "profile": "apex_style_live_trace",
            "detection_mode": "apex",
            "pull_mode": "aba",
            "mouse_backend": "auto",
            "allow_live_mouse": False,
        }
    )


@pytest.fixture
def runtime() -> AssistRuntime:
    with patch("yolo_targeting.reload_yolo_engine") as reload:
        engines: list[MagicMock] = []

        def _load(_cfg: dict) -> MagicMock:
            eng = MagicMock(name=f"engine_{len(engines)}")
            engines.append(eng)
            return eng

        reload.side_effect = _load
        rt = AssistRuntime(
            _cv_cfg(),
            MagicMock(),
            process_debounce=MagicMock(),
            on_external_stop=lambda: None,
        )
        rt._dry = True
        yield rt


def test_churn_yolo_cv_yolo_clears_lock_and_engine(runtime: AssistRuntime) -> None:
    runtime.sync_config_subsystems(_yolo_cfg())
    assert runtime._yolo_engine is not None
    runtime._target_lock.locked_target = MagicMock()
    runtime._last_apex_box = (30.0, 60.0)

    runtime.sync_config_subsystems(_cv_cfg())
    assert runtime._yolo_engine is None
    assert runtime._last_apex_box is None
    assert runtime._target_lock.locked_target is None
    assert runtime._pull is not None

    runtime.sync_config_subsystems(_yolo_cfg())
    assert runtime._yolo_engine is not None
    assert runtime._pull is None
    assert runtime._detect_ctx is None


def test_detection_mode_patch_unwinds_stack_via_normalize(tmp_path) -> None:
    from runtime_controller import RuntimeController

    cfg_path = tmp_path / "config.json"
    cfg_path.write_text("{}", encoding="utf-8")
    base = normalize_app_config(_yolo_cfg())
    ctrl = RuntimeController(base, cfg_path)
    merged = ctrl.apply_config_patch({"detection_mode": "apex"}, persist=False)
    assert merged["pull_mode"] == "aba"
    assert merged["mouse_backend"] == "auto"
