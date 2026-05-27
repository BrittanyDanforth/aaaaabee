"""Mocked integration: YOLO ↔ apex mode flips and config hot-reload parity (no torch/weights)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from apexaimbot_bridge import reset_apexaimbot_cache
from config_pipeline import normalize_app_config
from detector import Target
from profiles import PROFILE_APEXAIMBOT, effective_capture_fps, uses_apex_pid_pull
from pull import PullController
from runtime_controller import RuntimeController

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _ensure_mss_stub() -> None:
    sys.modules.setdefault("mss", MagicMock())


def _apex_cv_cfg(**overrides: object) -> dict:
    base = normalize_app_config(
        {
            "profile": "apex_style_dry_run",
            "detection_mode": "apex",
            "pull_mode": "aba",
            "allow_live_mouse": False,
        }
    )
    base.update(overrides)
    return base


def _yolo_pid_cfg(**overrides: object) -> dict:
    base = normalize_app_config(
        {
            "profile": PROFILE_APEXAIMBOT,
            "detection_mode": "yolo",
            "pull_mode": "apexaimbot_pid",
            "allow_live_mouse": False,
        }
    )
    base.update(overrides)
    return base


def _make_assist_runtime(cfg: dict):
    from runtime import AssistRuntime

    with patch("yolo_targeting.reload_yolo_engine", return_value=None):
        return AssistRuntime(cfg, REPO / "config.json")


def test_repeated_yolo_apex_yolo_subsystem_state() -> None:
    """Three mode flips must not leak lock, engine, pull, or detect context."""
    engines: list[MagicMock] = []

    def fake_reload(_cfg: dict) -> MagicMock:
        eng = MagicMock(name=f"yolo_engine_{len(engines)}")
        engines.append(eng)
        return eng

    reset_apexaimbot_cache()
    with patch("yolo_targeting.reload_yolo_engine", side_effect=fake_reload):
        rt = _make_assist_runtime(_apex_cv_cfg())
        rt._target_lock.locked_target = Target(
            1.0, 2.0, 10.0, 1.0, 0.9, bbox_w=10, bbox_h=20
        )
        rt._target_lock.target_lost_frames = 3
        rt._last_apex_box = (40.0, 80.0)
        rt._last_motion = MagicMock(name="motion")
        rt._frame_has_target = True
        rt._apex_recoil = MagicMock()
        rt._overlay = MagicMock(name="overlay")

        rt.sync_config_subsystems(_apex_cv_cfg())
        assert rt._detect_ctx is not None
        assert isinstance(rt._pull, PullController)

        rt.sync_config_subsystems(_yolo_pid_cfg())
        assert rt._target_lock.locked_target is None
        assert rt._target_lock.target_lost_frames == 0
        assert rt._last_apex_box is None
        assert rt._last_motion is None
        assert rt._frame_has_target is False
        rt._overlay.set_state.assert_called_with(False, None)
        assert rt._detect_ctx is None
        assert rt._pull is None
        assert uses_apex_pid_pull(rt.config) or rt._pull is None
        assert rt._yolo_engine is engines[0]

        rt.sync_config_subsystems(_apex_cv_cfg())
        assert rt._yolo_engine is None
        assert rt._detect_ctx is not None
        assert isinstance(rt._pull, PullController)

        rt.sync_config_subsystems(_yolo_pid_cfg())
        assert rt._yolo_engine is engines[1]
        assert engines[0] is not engines[1]
        assert rt._pull is None
        assert rt._detect_ctx is None


def test_repeated_flips_via_runtime_controller_call_sync() -> None:
    ctrl = RuntimeController(_apex_cv_cfg(), REPO / "config.json")
    live = _make_assist_runtime(_apex_cv_cfg())
    live.running = True
    ctrl._runtime = live

    engines: list[MagicMock] = []

    with patch("yolo_targeting.reload_yolo_engine", side_effect=lambda _c: engines.append(MagicMock()) or engines[-1]), patch.object(
        live, "sync_config_subsystems", wraps=live.sync_config_subsystems
    ) as sync:
        for _ in range(2):
            ctrl.apply_config_patch(
                normalize_app_config(
                    {
                        "detection_mode": "yolo",
                        "pull_mode": "apexaimbot_pid",
                        "profile": PROFILE_APEXAIMBOT,
                        "allow_live_mouse": False,
                    }
                ),
                persist=False,
                full_replace=True,
            )
            ctrl.apply_config_patch(_apex_cv_cfg(), persist=False, full_replace=True)
        assert sync.call_count >= 4
        assert len(engines) >= 2


def test_full_save_in_yolo_mode_reloads_engine_for_aim_fraction() -> None:
    """Save Settings (full_replace) must apply yolo_aim_fraction — not skip reload."""
    from runtime_controller import RuntimeController

    base = _yolo_pid_cfg()
    ctrl = RuntimeController(base, REPO / "config.json")
    live = MagicMock()
    live.running = True
    live._dry = True
    live.config = dict(base)
    live._yolo_engine = MagicMock(name="old_engine")
    live._aim_tracker = MagicMock()
    live._detect_ctx = None
    live._pull = None
    live.sync_config_subsystems = MagicMock()
    live._reset_apex_aim_state = MagicMock()
    ctrl._runtime = live

    reloaded: list[dict] = []

    with patch(
        "yolo_targeting.reload_yolo_engine",
        side_effect=lambda c: reloaded.append(dict(c)) or MagicMock(name="new_engine"),
    ):
        saved = normalize_app_config({**base, "yolo_aim_fraction": 0.42})
        with patch.object(ctrl, "_write_config_disk"):
            ctrl.save_config(saved)

    assert reloaded, "full_save in YOLO mode must reload engine when aim fraction changes"
    assert abs(float(reloaded[-1]["yolo_aim_fraction"]) - 0.42) < 1e-6
    assert live._yolo_engine is not None


def test_reset_apex_pid_on_mode_change() -> None:
    rt = _make_assist_runtime(_yolo_pid_cfg())
    eng = MagicMock()
    rt._yolo_engine = eng
    with patch("apexaimbot_bridge.reset_apexaimbot_pid") as reset_pid:
        rt.sync_config_subsystems(_apex_cv_cfg())
        reset_pid.assert_called_once_with(eng)


@pytest.mark.parametrize(
    "patch_key,patch_fragment,assert_fn",
    [
        (
            "capture_fps",
            {"capture_fps": 90, "profile": "apex_style_live_trace", "allow_live_mouse": True},
            lambda live: effective_capture_fps(live.config) == 90
            and live._configured_fps == 90,
        ),
        (
            "pull_mode",
            {"pull_mode": "aba", "detection_mode": "apex", "profile": "apex_style_dry_run"},
            lambda live: not uses_apex_pid_pull(live.config),
        ),
        (
            "mouse_backend",
            {"mouse_backend": "recording"},
            lambda live: str(live.config.get("mouse_backend")) == "recording",
        ),
        (
            "detection_mode",
            {"detection_mode": "yolo", "pull_mode": "apexaimbot_pid", "profile": PROFILE_APEXAIMBOT},
            lambda live: str(live.config.get("detection_mode")) == "yolo",
        ),
        (
            "yolo_aim_fraction",
            {"yolo_aim_fraction": 0.35, "detection_mode": "yolo", "profile": PROFILE_APEXAIMBOT},
            lambda live: abs(float(live.config.get("yolo_aim_fraction", 0)) - 0.35) < 1e-6,
        ),
    ],
)
def test_slider_patch_and_save_use_same_hot_reload_path(
    patch_key: str, patch_fragment: dict, assert_fn
) -> None:
    base = _apex_cv_cfg()
    if patch_key in ("yolo_aim_fraction", "detection_mode"):
        base = _yolo_pid_cfg()
    ctrl = RuntimeController(base, REPO / "config.json")
    live = MagicMock()
    live.running = True
    live._dry = False
    live.config = dict(base)
    live._configured_fps = effective_capture_fps(base)
    live._stats = MagicMock()
    live._stats.configured_fps = live._configured_fps
    live._detect_ctx = MagicMock()
    live._pull = MagicMock()
    live._aim_tracker = MagicMock()
    live.sync_config_subsystems = MagicMock()
    live._reset_apex_aim_state = MagicMock()
    live._ensure_apex_recoil = MagicMock()
    ctrl._runtime = live

    merged_slider = normalize_app_config({**base, **patch_fragment})

    with patch("yolo_targeting.reload_yolo_engine", return_value=MagicMock()), patch(
        "mouse_io.create_mouse_backend", return_value=MagicMock(name="mouse")
    ):
        ctrl.apply_config_patch(patch_fragment, persist=False)
        assert assert_fn(live)

        live.config = dict(base)
        live._configured_fps = effective_capture_fps(base)
        live.sync_config_subsystems.reset_mock()
        live._reset_apex_aim_state.reset_mock()

        with patch.object(ctrl, "_write_config_disk"):
            ctrl.save_config(merged_slider)

        assert assert_fn(live)
        if patch_key in ("detection_mode", "pull_mode"):
            assert live.sync_config_subsystems.call_count >= 1
