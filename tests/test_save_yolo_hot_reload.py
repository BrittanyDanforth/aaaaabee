"""Save Settings (full_replace) must match slider hot-reload for YOLO/Apex keys."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from config_pipeline import normalize_app_config
from profiles import PROFILE_APEXAIMBOT, effective_capture_fps
from runtime_controller import (
    RuntimeController,
    _YOLO_ENGINE_TUNE_KEYS,
    should_reload_yolo_engine,
)

REPO = Path(__file__).resolve().parents[1]


def _tune_fragment(base: dict, tune_key: str) -> dict:
    if tune_key == "yolo_aim_fraction":
        return {tune_key: 0.39}
    if tune_key == "yolo_confidence_min":
        return {tune_key: 0.52}
    if tune_key == "yolo_use_fp16":
        return {tune_key: not bool(base.get(tune_key, False))}
    if tune_key in ("yolo_inference_size", "yolo_max_det", "yolo_grab_width", "yolo_grab_height"):
        return {tune_key: int(base.get(tune_key, 416)) + 2}
    if tune_key == "yolo_device":
        return {tune_key: "cpu"}
    if tune_key == "yolo_weights_path":
        return {tune_key: "third_party/apexaimbot/weights/APEX416SFP32.engine"}
    if tune_key == "yolo_yolov5_root":
        return {tune_key: "third_party/apexaimbot"}
    return {tune_key: float(base.get(tune_key, 0.1)) + 0.02}


def _yolo_base() -> dict:
    return normalize_app_config(
        {
            "profile": PROFILE_APEXAIMBOT,
            "detection_mode": "yolo",
            "pull_mode": "apexaimbot_pid",
            "allow_live_mouse": False,
        }
    )


def _live_ctrl() -> tuple[RuntimeController, MagicMock]:
    ctrl = RuntimeController(_yolo_base(), REPO / "config.json")
    live = MagicMock()
    live.running = True
    live._dry = True
    live.config = _yolo_base()
    live._configured_fps = 60
    live._stats = MagicMock()
    live._stats.configured_fps = 60
    live._aim_tracker = MagicMock()
    live._detect_ctx = None
    live._pull = None
    live.sync_config_subsystems = MagicMock()
    live._reset_apex_aim_state = MagicMock()
    live._ensure_apex_recoil = MagicMock()
    live._yolo_engine = MagicMock(name="engine")
    ctrl._runtime = live
    return ctrl, live


@pytest.mark.parametrize("tune_key", sorted(_YOLO_ENGINE_TUNE_KEYS))
def test_should_reload_on_full_save_for_each_engine_tune_key(tune_key: str) -> None:
    base = _yolo_base()
    patch = _tune_fragment(base, tune_key)
    merged = normalize_app_config({**base, **patch})
    assert should_reload_yolo_engine(patch, merged, full_replace=False)
    assert should_reload_yolo_engine(patch, merged, full_replace=True)


@pytest.mark.parametrize("tune_key", sorted(_YOLO_ENGINE_TUNE_KEYS))
def test_save_and_slider_both_reload_engine_for_tune_key(tune_key: str) -> None:
    ctrl, live = _live_ctrl()
    base = dict(live.config)
    fragment = _tune_fragment(base, tune_key)

    reload_counts: list[str] = []

    def track_reload(cfg: dict) -> MagicMock:
        reload_counts.append(tune_key)
        return MagicMock(name=f"eng_{tune_key}")

    with patch("yolo_targeting.reload_yolo_engine", side_effect=track_reload):
        ctrl.apply_config_patch(fragment, persist=False)
        assert reload_counts == [tune_key]

        reload_counts.clear()
        live.config = dict(base)
        merged = normalize_app_config({**base, **fragment})
        with patch.object(ctrl, "_write_config_disk"):
            ctrl.save_config(merged)
        assert reload_counts == [tune_key]


def test_full_save_in_yolo_always_should_reload() -> None:
    base = _yolo_base()
    assert should_reload_yolo_engine(
        dict(base), normalize_app_config(base), full_replace=True
    )


def test_mode_only_patch_does_not_require_engine_reload() -> None:
    """Mode flip reload is sync_config_subsystems; patch-only mode keys skip engine reload."""
    base = _yolo_base()
    patch = {"detection_mode": "yolo", "pull_mode": "apexaimbot_pid"}
    merged = normalize_app_config({**base, **patch})
    assert not should_reload_yolo_engine(patch, merged, full_replace=False)


def test_capture_fps_save_matches_slider() -> None:
    ctrl, live = _live_ctrl()
    base = normalize_app_config(
        {
            "profile": "apex_style_live_trace",
            "detection_mode": "yolo",
            "pull_mode": "apexaimbot_pid",
            "allow_live_mouse": True,
        }
    )
    live.config = dict(base)
    frag = {"capture_fps": 90, "profile": "apex_style_live_trace", "allow_live_mouse": True}
    ctrl.apply_config_patch(frag, persist=False)
    assert live._configured_fps == 90
    live._configured_fps = 30
    with patch.object(ctrl, "_write_config_disk"):
        ctrl.save_config(normalize_app_config({**base, **frag}))
    assert live._configured_fps == 90


def test_recoil_key_save_calls_ensure_recoil_without_engine_reload() -> None:
    ctrl, live = _live_ctrl()
    reloaded: list[str] = []

    with patch(
        "yolo_targeting.reload_yolo_engine",
        side_effect=lambda _c: reloaded.append("reload") or MagicMock(),
    ):
        ctrl.apply_config_patch({"apexaimbot_sens": 2.5}, persist=False)
    assert reloaded == []
    live._ensure_apex_recoil.assert_called_once()
