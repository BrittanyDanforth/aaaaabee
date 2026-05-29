"""ApexAimBot PID / lock-box parity with main.py."""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

from apex_aim_loop import compute_apex_pid_pull, run_apex_subtick_window
from apexaimbot_bridge import in_lock_box, pid_mouse_delta
from detector import Target
from third_party.apexaimbot.PID import PID_PLUS_PLUS


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


def _pid_runtime() -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            aim_offset_fraction=0.2,
            lock_range_x=1.0,
            lock_range_y=0.5,
            min_step=10,
            max_step=6,
        ),
        pid_x=PID_PLUS_PLUS(0, 0.36, 0.032, 0.01),
        pid_y=PID_PLUS_PLUS(0, 0.2, 0.0, 0.0),
        mouse_modifier=1.0,
    )


def _pid_target() -> Target:
    return Target(
        centroid_x=260.0,
        centroid_y=216.0,
        apex_raw_offset_x=52.0,
        apex_raw_offset_y=8.0,
        area=9600.0,
        distance_to_center=54.4,
        confidence=0.9,
        bbox_x=220,
        bbox_y=156,
        bbox_w=80,
        bbox_h=120,
    )


class _Gate:
    allowed = True


def test_subtick_enabled_suppresses_main_step_but_ads_window_moves() -> None:
    rt = _pid_runtime()
    target = _pid_target()
    pr = compute_apex_pid_pull(
        rt,
        target=target,
        frame_cx=208.0,
        frame_cy=208.0,
        box_wh=(80.0, 120.0),
        hip_fire=False,
        use_subticks=True,
    )
    assert (pr.dx, pr.dy) == (0, 0)

    moves: list[tuple[int, int]] = []
    run_apex_subtick_window(
        rt,
        {},
        target=target,
        frame_cx=208.0,
        frame_cy=208.0,
        box_wh=(80.0, 120.0),
        hip_fire=False,
        deadline=time.perf_counter() + 0.020,
        subtick_hz=120,
        pid_enabled=True,
        recoil_enabled=False,
        is_firing=lambda: False,
        mouse_move=lambda dx, dy, *, recoil_only=False: moves.append((dx, dy)) or _Gate(),
        recoil_tick=lambda cfg, skip_x: False,
        on_pid_moved=lambda: None,
        sleep=lambda seconds: None,
    )
    assert moves, "ADS-only Apex PID subticks must emit while the main step is suppressed"


def test_subtick_deadline_overrun_gets_one_pid_fallback_tick() -> None:
    rt = _pid_runtime()
    target = _pid_target()
    moves: list[tuple[int, int]] = []
    run_apex_subtick_window(
        rt,
        {},
        target=target,
        frame_cx=208.0,
        frame_cy=208.0,
        box_wh=(80.0, 120.0),
        hip_fire=False,
        deadline=time.perf_counter() - 0.001,
        subtick_hz=120,
        pid_enabled=True,
        recoil_enabled=False,
        is_firing=lambda: True,
        mouse_move=lambda dx, dy, *, recoil_only=False: moves.append((dx, dy)) or _Gate(),
        recoil_tick=lambda cfg, skip_x: False,
        on_pid_moved=lambda: None,
        sleep=lambda seconds: None,
    )
    assert len(moves) == 1
