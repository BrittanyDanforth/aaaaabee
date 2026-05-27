"""ApexAimBot PID / lock-box parity with main.py."""

from __future__ import annotations

from unittest.mock import MagicMock

from apexaimbot_bridge import in_lock_box, pid_mouse_delta


def _rt() -> MagicMock:
    rt = MagicMock()
    rt.config.min_step = 10
    rt.config.max_step = 6
    rt.config.lock_range_y = 0.5
    rt.config.lock_range_x = 1.0
    rt.pid_x.getMove.return_value = 3.0
    rt.pid_y.getMove.return_value = -2.0
    return rt


def test_pid_uses_min_step_on_hip_fire() -> None:
    rt = _rt()
    pid_mouse_delta(rt, error_x=5.0, error_y=-3.0, hip_fire=True)
    rt.pid_x.getMove.assert_called_once_with(5.0, 10)


def test_pid_uses_max_step_on_ads() -> None:
    rt = _rt()
    pid_mouse_delta(rt, error_x=5.0, error_y=-3.0, hip_fire=False)
    rt.pid_x.getMove.assert_called_once_with(5.0, 6)


def test_lock_box_wider_on_hip_fire() -> None:
    rt = _rt()
    assert in_lock_box(
        rt, error_x=90.0, error_y=40.0, box_width=100.0, box_height=100.0, hip_fire=True
    )
    assert not in_lock_box(
        rt, error_x=90.0, error_y=40.0, box_width=100.0, box_height=100.0, hip_fire=False
    )
