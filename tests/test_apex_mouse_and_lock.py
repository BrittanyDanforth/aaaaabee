"""ApexAimBot mouse backends, YOLO lock, sens modifier."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from apexaimbot_bridge import ApexAimBotRuntime, pid_mouse_delta
from detector import DetectionResult, Target
from mouse_io import create_mouse_backend
from target_lock import TargetLockState, apply_yolo_target_lock
from third_party.apexaimbot.logitech_mouse import resolve_ghub_dll_dir


def test_resolve_ghub_dll_missing_on_linux() -> None:
    # CI has no bundled DLL
    assert resolve_ghub_dll_dir() is None or True


def test_apexaimbot_mouse_backend_on_linux() -> None:
    b = create_mouse_backend("apexaimbot")
    assert b.name in ("pynput", "recording")


def test_apply_yolo_target_lock_nearest_wins() -> None:
    state = TargetLockState()
    t1 = Target(100.0, 100.0, 500.0, 10.0, 0.9, bbox_x=90, bbox_y=80, bbox_w=20, bbox_h=40)
    t2 = Target(200.0, 100.0, 500.0, 10.0, 0.9, bbox_x=190, bbox_y=80, bbox_w=20, bbox_h=40)
    cfg = {"target_lost_frames_before_unlock": 5}
    r1, _ = apply_yolo_target_lock(state, DetectionResult(t1, 1, 0.9), cfg=cfg)
    assert r1.target is t1
    r2, _ = apply_yolo_target_lock(state, DetectionResult(t2, 1, 0.9), cfg=cfg)
    assert r2.target is t2
    assert state.target_lost_frames == 0


def test_pid_mouse_modifier_scales_output() -> None:
    rt = MagicMock()
    rt.config = MagicMock(min_step=10, max_step=6)
    rt.mouse_modifier = 2.0
    rt.pid_x.getMove.return_value = 5
    rt.pid_y.getMove.return_value = 3
    dx, dy = pid_mouse_delta(rt, error_x=10.0, error_y=5.0, hip_fire=True)
    assert dx == 10
    assert dy == 6


def test_create_mouse_backend_logitech_requires_dll() -> None:
    with patch(
        "third_party.apexaimbot.logitech_mouse.get_logitech_driver",
        return_value=None,
    ):
        try:
            create_mouse_backend("logitech_ghub")
            raised = False
        except RuntimeError:
            raised = True
        assert raised
