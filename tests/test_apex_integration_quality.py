"""Code-quality tests: one cache path, config merge order, no double PID."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from apex_aim_loop import ApexAimSettings, compute_apex_pid_pull
from detector import Target
from apexaimbot_bridge import _cache_key, prepare_apex_cfg, reset_apexaimbot_cache
from assist import load_config
from config_pipeline import merge_apex_ini, normalize_app_config
from profiles import PROFILE_APEX_STYLE_DRY_RUN, PROFILE_APEXAIMBOT, apply_profile
from yolo_detector import get_yolo_engine, reload_yolo_engine

REPO = Path(__file__).resolve().parents[1]


def test_normalize_app_config_merges_ini_before_validate() -> None:
    raw = apply_profile({"profile": PROFILE_APEXAIMBOT})
    cfg = normalize_app_config(raw)
    assert cfg["yolo_iou_thres"] == 0.8
    assert cfg["yolo_inference_size"] == 416
    assert cfg.get("apexaimbot_sens") == 5.0


def test_cv_profile_unchanged_by_apex_merge() -> None:
    raw = apply_profile({"profile": PROFILE_APEX_STYLE_DRY_RUN})
    cfg = normalize_app_config(raw)
    assert cfg["detection_mode"] == "apex"
    assert cfg.get("pull_mode", "aba") == "aba"


def test_reload_yolo_engine_clears_bridge_cache() -> None:
    import apexaimbot_bridge as bridge

    cfg = normalize_app_config(apply_profile({"profile": PROFILE_APEXAIMBOT}))
    with patch.object(
        bridge.ApexAimBotRuntime,
        "from_app_config",
        return_value=MagicMock(config=MagicMock(weights_path=Path("w.pt"))),
    ):
        get_yolo_engine(cfg)
        assert bridge._engine_cache is not None
        reload_yolo_engine(cfg)
        assert bridge._engine_cache is not None
        reset_apexaimbot_cache()
        assert bridge._engine_cache is None


def test_compute_apex_pid_pull_skips_integrator_when_subticks() -> None:
    engine = MagicMock()
    engine.config.min_step = 10
    engine.config.max_step = 6
    engine.config.lock_range_y = 0.5
    engine.config.aim_offset_fraction = 0.2
    tgt = Target(
        110.0,
        84.0,
        4000.0,
        10.0,
        0.9,
        bbox_w=50,
        bbox_h=80,
        apex_raw_offset_x=10.0,
        apex_raw_offset_y=20.0,
    )
    with patch("apex_aim_loop.in_lock_box", return_value=True):
        with patch("apex_aim_loop.pid_mouse_delta") as pm:
            pr = compute_apex_pid_pull(
                engine,
                target=tgt,
                frame_cx=100.0,
                frame_cy=100.0,
                box_wh=(50.0, 80.0),
                hip_fire=True,
                use_subticks=True,
            )
    pm.assert_not_called()
    assert pr.dx == 0 and pr.dy == 0


def test_compute_apex_pid_pull_calls_pid_when_no_subticks() -> None:
    engine = MagicMock()
    engine.config.min_step = 10
    engine.config.max_step = 6
    engine.config.lock_range_y = 0.5
    engine.config.aim_offset_fraction = 0.2
    tgt = Target(
        110.0,
        84.0,
        4000.0,
        10.0,
        0.9,
        bbox_w=50,
        bbox_h=80,
        apex_raw_offset_x=10.0,
        apex_raw_offset_y=20.0,
    )
    with patch("apex_aim_loop.in_lock_box", return_value=True):
        with patch("apex_aim_loop.pid_mouse_delta", return_value=(3, -2)) as pm:
            pr = compute_apex_pid_pull(
                engine,
                target=tgt,
                frame_cx=100.0,
                frame_cy=100.0,
                box_wh=(50.0, 80.0),
                hip_fire=False,
                use_subticks=False,
            )
    pm.assert_called_once()
    assert pr.dx == 3 and pr.dy == -2


def test_apex_aim_settings_subtick_default_from_cfg() -> None:
    cfg = {"detection_mode": "yolo", "pull_mode": "apexaimbot_pid", "apex_pid_subtick_hz": 120}
    s = ApexAimSettings.from_cfg(cfg, firing=True, ads_live=False)
    assert s.active
    assert s.subtick_hz == 120
    assert s.hip_fire


def test_cache_key_changes_when_pid_changes() -> None:
    a = _cache_key(prepare_apex_cfg({"detection_mode": "yolo", "apexaimbot_pid_x_p": 0.36}))
    b = _cache_key(prepare_apex_cfg({"detection_mode": "yolo", "apexaimbot_pid_x_p": 0.5}))
    assert a != b


def test_gui_patch_validation_raises() -> None:
    from config_validation import ConfigError
    from runtime_controller import RuntimeController

    cfg = load_config(REPO / "config.json")
    ctrl = RuntimeController(cfg, REPO / "config.json")
    with pytest.raises(ConfigError):
        ctrl.apply_config_patch({"pull_mode": "not_a_mode"}, persist=False)
