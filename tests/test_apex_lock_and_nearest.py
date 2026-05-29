"""Upstream parity: nearest origin, lock-box Y gate, nearest-box confidence."""

from __future__ import annotations

from unittest.mock import MagicMock

from apexaimbot_bridge import apex_pid_errors, in_lock_box
from detector import Target
from third_party.apexaimbot.nearest import send_nearest_pos_to_mouse_ctrl


def test_nearest_uses_custom_origin() -> None:
    boxes = [("enemy", 0.75, 0.5, 0.2, 0.4, 80)]
    r = send_nearest_pos_to_mouse_ctrl(
        boxes, grab_width=400.0, grab_height=400.0, origin_x=300.0, origin_y=200.0
    )
    assert r is not None
    pos, _w, _h, conf = r
    assert conf == 0.8
    assert abs(pos[0]) < 50.0


def test_lock_box_uses_raw_y_not_aim_offset() -> None:
    rt = MagicMock()
    rt.config.lock_range_x = 1.0
    rt.config.lock_range_y = 0.5
    bh = 100.0
    raw_y = 55.0
    assert not in_lock_box(
        rt,
        error_x=10.0,
        error_y=raw_y - bh * 0.2,
        box_width=100.0,
        box_height=bh,
        hip_fire=True,
        raw_offset_y=raw_y,
    )
    assert in_lock_box(
        rt,
        error_x=10.0,
        error_y=0.0,
        box_width=100.0,
        box_height=bh,
        hip_fire=True,
        raw_offset_y=40.0,
    )


def test_apex_pid_errors_match_upstream() -> None:
    t = Target(
        110.0,
        74.0,
        5000.0,
        10.0,
        0.9,
        bbox_w=50,
        bbox_h=100,
        apex_raw_offset_x=10.0,
        apex_raw_offset_y=50.0,
    )
    err_x, err_y, _, _ = apex_pid_errors(
        t, frame_cx=100.0, frame_cy=100.0, aim_offset_fraction=0.2
    )
    assert err_x == 10.0
    assert err_y == 50.0 - 100.0 * 0.2
