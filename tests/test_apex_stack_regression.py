"""Regression tests for PR #28 fixes and consolidated YOLO stack."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from apexaimbot_bridge import get_apexaimbot_runtime, reset_apexaimbot_cache
from config_pipeline import normalize_app_config, merge_apex_ini
from detector import Target
from profiles import PROFILE_APEXAIMBOT, apply_profile, load_config
from target_lock import TargetLockState
from yolo_targeting import (
    detect_yolo_target,
    tick_yolo_lock_idle,
    yolo_detect_and_lock,
)

REPO = Path(__file__).resolve().parents[1]


def test_profiles_load_config_uses_ini_merge() -> None:
    merged_ini = merge_apex_ini(
        apply_profile({"profile": PROFILE_APEXAIMBOT, "detection_mode": "yolo"})
    )
    assert "yolo_weights_path" in merged_ini
    cfg = normalize_app_config(
        {"profile": PROFILE_APEXAIMBOT, "detection_mode": "yolo", "allow_live_mouse": True}
    )
    assert cfg["pull_mode"] == "apexaimbot_pid"
    assert float(cfg.get("yolo_confidence_min", 0)) >= 0.5


def test_normalize_matches_pipeline_order() -> None:
    raw = {"profile": PROFILE_APEXAIMBOT, "detection_mode": "yolo", "allow_live_mouse": True}
    cfg = normalize_app_config(raw)
    assert cfg["pull_mode"] == "apexaimbot_pid"


def test_engine_cache_not_poisoned_on_load_failure() -> None:
    import apexaimbot_bridge as bridge

    reset_apexaimbot_cache()
    cfg = {
        "detection_mode": "yolo",
        "yolo_yolov5_root": "third_party/apexaimbot",
        "yolo_weights_path": "third_party/apexaimbot/weights/DOES_NOT_EXIST.pt",
    }
    with patch("apexaimbot_bridge._load_runtime", side_effect=RuntimeError("fail")):
        assert get_apexaimbot_runtime(cfg) is None
    assert bridge._engine_cache is None
    with patch("apexaimbot_bridge._load_runtime") as load:
        load.return_value = MagicMock()
        rt = get_apexaimbot_runtime(cfg)
        assert rt is not None
        assert bridge._engine_cache is not None


def test_yolo_detect_and_lock_advances_idle_grace() -> None:
    state = TargetLockState()
    t = Target(100.0, 80.0, 500.0, 10.0, 0.9, bbox_w=40, bbox_h=80)
    state.locked_target = t
    state.target_lost_frames = 0
    cfg = {
        "detection_mode": "yolo",
        "target_lost_frames_before_unlock": 5,
        "yolo_apex_nearest_lock": True,
    }
    tick_yolo_lock_idle(cfg, state)
    assert state.target_lost_frames == 1
    assert state.locked_target is t


def test_yolo_detect_and_lock_expires_after_idle_ticks() -> None:
    state = TargetLockState()
    t = Target(100.0, 80.0, 500.0, 10.0, 0.9, bbox_w=40, bbox_h=80)
    state.locked_target = t
    cfg = {
        "detection_mode": "yolo",
        "target_lost_frames_before_unlock": 2,
        "yolo_apex_nearest_lock": True,
    }
    for _ in range(3):
        tick_yolo_lock_idle(cfg, state)
    assert state.locked_target is None


def test_detect_yolo_delegates_to_bridge() -> None:
    frame = np.zeros((416, 416, 3), dtype=np.uint8)
    fake_t = Target(200.0, 180.0, 5000.0, 10.0, 0.88, bbox_w=50, bbox_h=100)
    mock_rt = MagicMock()
    with patch(
        "yolo_targeting.find_best_yolo_target",
        return_value=__import__("detector").DetectionResult(
            fake_t, 1, 0.88, active=True
        ),
    ) as fb:
        r = detect_yolo_target(
            {"yolo_confidence_min": 0.5, "min_target_area_pixels": 1},
            frame,
            mock_rt,
            fov_radius=208,
            center_x=208.0,
            center_y=208.0,
        )
    fb.assert_called_once()
    assert r.target is fake_t


def test_runtime_controller_save_uses_full_replace() -> None:
    from runtime_controller import RuntimeController

    ctrl = RuntimeController(
        {"detection_mode": "apex", "allow_live_mouse": False, "profile": "apex_style_dry_run"},
        REPO / "config.json",
    )
    with patch.object(ctrl, "apply_config_patch") as patch_apply:
        ctrl.save_config({"detection_mode": "yolo", "profile": PROFILE_APEXAIMBOT})
        patch_apply.assert_called_once()
        _args, kwargs = patch_apply.call_args
        assert kwargs.get("full_replace") is True


def test_save_config_does_not_recurse() -> None:
    from runtime_controller import RuntimeController

    ctrl = RuntimeController(
        {"detection_mode": "apex", "allow_live_mouse": False, "profile": "apex_style_dry_run"},
        REPO / "config.json",
    )
    writes: list[str] = []

    def fake_write(cfg: dict) -> None:
        writes.append("disk")

    with patch.object(ctrl, "_write_config_disk", side_effect=fake_write):
        with patch.object(ctrl, "apply_config_patch", wraps=ctrl.apply_config_patch) as patch_apply:
            ctrl.save_config({"detection_mode": "apex", "capture_fps": 45})
            assert patch_apply.call_count == 1
            assert writes == ["disk"]


def test_sync_config_subsystems_mode_flip_resets_lock() -> None:
    import sys

    sys.modules.setdefault("mss", MagicMock())
    from runtime import AssistRuntime

    cfg_apex = normalize_app_config(
        {"profile": "apex_style_dry_run", "detection_mode": "apex", "pull_mode": "aba"}
    )
    rt = AssistRuntime(cfg_apex, REPO / "config.json")
    t = Target(10.0, 10.0, 100.0, 5.0, 0.9, bbox_w=20, bbox_h=40)
    rt._target_lock.locked_target = t
    cfg_yolo = normalize_app_config(
        {
            "profile": PROFILE_APEXAIMBOT,
            "detection_mode": "yolo",
            "pull_mode": "apexaimbot_pid",
            "allow_live_mouse": False,
        }
    )
    with patch("yolo_targeting.reload_yolo_engine", return_value=MagicMock()):
        rt.sync_config_subsystems(cfg_yolo)
    assert rt._target_lock.locked_target is None
    cfg_back = normalize_app_config(
        {"profile": "apex_style_dry_run", "detection_mode": "apex", "pull_mode": "aba"}
    )
    rt.sync_config_subsystems(cfg_back)
    assert rt._yolo_engine is None


def test_capture_fps_hot_reload_on_patch() -> None:
    from runtime_controller import RuntimeController

    live = MagicMock()
    live.running = True
    live._configured_fps = 30
    live._stats = MagicMock()
    live._stats.configured_fps = 30
    ctrl = RuntimeController(
        {"detection_mode": "apex", "allow_live_mouse": False, "profile": "apex_style_dry_run"},
        REPO / "config.json",
    )
    ctrl._runtime = live
    patch_cfg = normalize_app_config(
        {
            "profile": "apex_style_live_trace",
            "detection_mode": "apex",
            "allow_live_mouse": True,
            "capture_fps": 90,
        }
    )
    ctrl.apply_config_patch(patch_cfg, persist=False, full_replace=True)
    assert live._configured_fps == 90
    assert live._stats.configured_fps == 90


def test_apply_config_patch_same_path_as_save_for_live() -> None:
    from runtime_controller import RuntimeController

    live = MagicMock()
    live.running = True
    live._detect_ctx = MagicMock()
    live._pull = MagicMock()
    ctrl = RuntimeController(
        normalize_app_config(
            {"profile": "apex_style_dry_run", "detection_mode": "apex", "pull_mode": "aba"}
        ),
        REPO / "config.json",
    )
    ctrl._runtime = live
    merged = ctrl.apply_config_patch({"pull_strength": 0.77}, persist=False)
    assert live.config == merged
    live._pull.update_tuning.assert_called_once()


def test_yolo_targeting_single_lock_path() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    fake_t = Target(50.0, 50.0, 1000.0, 5.0, 0.9, bbox_w=20, bbox_h=40)
    state = TargetLockState()
    engine = MagicMock()
    with patch(
        "yolo_targeting.find_best_yolo_target",
        return_value=__import__("detector").DetectionResult(
            fake_t, 1, 0.9, active=True
        ),
    ):
        result, box = yolo_detect_and_lock(
            {
                "detection_mode": "yolo",
                "yolo_confidence_min": 0.5,
                "min_target_area_pixels": 1,
                "target_stickiness_pixels": 90,
                "humanoid_min_height_pixels": 0,
                "yolo_apex_nearest_lock": True,
                "target_lost_frames_before_unlock": 10,
            },
            frame,
            engine,
            state,
            fov_radius=50,
            center_x=50.0,
            center_y=50.0,
            frame_size=(100, 100),
        )
    assert result.target is fake_t
    assert box == (20.0, 40.0)
