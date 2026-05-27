"""ApexAimBot integration audit — merge_app_config, recoil sens, lock/PID hooks."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from apexaimbot_bridge import prepare_apex_cfg, reset_apexaimbot_pid
from assist import load_config
from profiles import PROFILE_APEXAIMBOT, PROFILE_DEFAULTS
from target_lock import TargetLockState, apply_yolo_target_lock
from third_party.apexaimbot.recoil_controller import compute_recoil_modifier
from detector import DetectionResult, Target

REPO = Path(__file__).resolve().parents[1]


def test_prepare_apex_cfg_merges_ini_defaults() -> None:
    cfg = prepare_apex_cfg(
        {
            "detection_mode": "yolo",
            "yolo_yolov5_root": "third_party/apexaimbot",
        }
    )
    assert cfg.get("yolo_iou_thres") == 0.25
    assert cfg.get("yolo_max_det") == 3
    assert "yolo_weights_path" in cfg


def test_compute_recoil_modifier_auto_sens() -> None:
    mod = compute_recoil_modifier(
        {"apexaimbot_auto_sens_modifier": True, "apexaimbot_sens": 5.0, "apexaimbot_ads_sens": 1.0}
    )
    assert abs(mod - 0.8) < 1e-6


def test_compute_recoil_modifier_manual() -> None:
    mod = compute_recoil_modifier(
        {
            "apexaimbot_auto_sens_modifier": False,
            "apexaimbot_recoil_modifier": 1.2,
        }
    )
    assert mod == 1.2


def test_yolo_lock_switch_calls_on_new_target() -> None:
    state = TargetLockState()
    t1 = Target(100.0, 100.0, 500.0, 10.0, 0.9)
    t2 = Target(300.0, 100.0, 500.0, 10.0, 0.9)
    cfg = {"target_lost_frames_before_unlock": 5, "yolo_switch_reset_pixels": 50.0}
    calls: list[str] = []

    def _on_new() -> None:
        calls.append("new")

    apply_yolo_target_lock(
        state, DetectionResult(t1, 1, 0.9), cfg=cfg, on_new_target=_on_new
    )
    apply_yolo_target_lock(
        state, DetectionResult(t2, 1, 0.9), cfg=cfg, on_new_target=_on_new
    )
    assert calls == ["new"]


def test_shipped_config_defaults_to_apexaimbot_profile() -> None:
    cfg_path = REPO / "config.json"
    cfg = load_config(cfg_path)
    assert cfg["profile"] == PROFILE_APEXAIMBOT
    assert cfg["detection_mode"] == "yolo"
    assert cfg["pull_mode"] == "apexaimbot_pid"
    assert cfg["apex_pid_subtick_hz"] == 120
    assert "yolo_weights_path" in cfg


def test_profile_apexaimbot_includes_weights_path() -> None:
    p = PROFILE_DEFAULTS[PROFILE_APEXAIMBOT]
    assert p["detection_mode"] == "yolo"
    assert "APEX416" in p["yolo_weights_path"] or "APEX22W" in p["yolo_weights_path"]


def test_mouse_gate_recoil_only_skips_target_lock() -> None:
    from mouse_gate import MouseGateContext, evaluate_mouse_gate

    cfg = {"allow_live_mouse": True, "offline_dev_mode": True}
    ctx = MouseGateContext(
        running=True,
        stopping=False,
        paused=False,
        mouse_enabled=True,
        ads_active=False,
        assist_without_ads=True,
        has_target=False,
        detection_fresh=False,
        target_lost_frames=99,
        stale_grace_frames=0,
        target_process_ok=True,
        dx=0,
        dy=2,
        max_pull_per_frame=22.0,
        recoil_only=True,
    )
    r = evaluate_mouse_gate(cfg, ctx)
    assert r.allowed, r.reason


def test_reset_apexaimbot_pid_clears_integral() -> None:
    rt = MagicMock()
    rt.pid_x.PIDOutput = 99.0
    rt.pid_x.Error = 1.0
    rt.pid_y.PIDOutput = 88.0
    reset_apexaimbot_pid(rt)
    assert rt.pid_x.PIDOutput == 0.0
    assert rt.pid_y.PIDOutput == 0.0
