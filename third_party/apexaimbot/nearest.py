"""send_nearest_pos_to_mouse_ctrl — copied from ApexAimBot main.py (no torch)."""

from __future__ import annotations

from typing import Any


def send_nearest_pos_to_mouse_ctrl(
    box_list: list[tuple[Any, ...]],
    *,
    grab_width: float,
    grab_height: float,
) -> tuple[tuple[float, float], float, float] | None:
    if not box_list:
        return None

    min_distance_sq = float("inf")
    min_position = None
    box_width = None
    box_height = None

    half_grab_width = grab_width / 2
    half_grab_height = grab_height / 2

    for box in box_list:
        box_center_x = box[1] * grab_width
        box_center_y = box[2] * grab_height
        box_width = box[3] * grab_width
        box_height = box[4] * grab_height

        distance_sq = (box_center_x - half_grab_width) ** 2 + (
            box_center_y - half_grab_height
        ) ** 2
        if distance_sq < min_distance_sq:
            min_distance_sq = distance_sq
            min_position = (
                box_center_x - half_grab_width,
                box_center_y - half_grab_height,
            )

    if min_position is None:
        return None
    return min_position, box_width, box_height
