"""send_nearest_pos_to_mouse_ctrl — copied from ApexAimBot main.py (no torch)."""

from __future__ import annotations

from typing import Any


def send_nearest_pos_to_mouse_ctrl(
    box_list: list[tuple[Any, ...]],
    *,
    grab_width: float,
    grab_height: float,
    origin_x: float | None = None,
    origin_y: float | None = None,
) -> tuple[tuple[float, float], float, float, float] | None:
    if not box_list:
        return None

    min_distance_sq = float("inf")
    min_position = None
    box_width = None
    box_height = None
    box_conf = 0.85

    pick_x = grab_width / 2 if origin_x is None else float(origin_x)
    pick_y = grab_height / 2 if origin_y is None else float(origin_y)

    for box in box_list:
        box_center_x = box[1] * grab_width
        box_center_y = box[2] * grab_height
        bw = box[3] * grab_width
        bh = box[4] * grab_height

        distance_sq = (box_center_x - pick_x) ** 2 + (box_center_y - pick_y) ** 2
        if distance_sq < min_distance_sq:
            min_distance_sq = distance_sq
            min_position = (box_center_x - pick_x, box_center_y - pick_y)
            box_width = bw
            box_height = bh
            box_conf = float(box[5]) / 100.0

    if min_position is None or box_width is None or box_height is None:
        return None
    return min_position, box_width, box_height, box_conf
