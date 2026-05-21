"""Reference frames modeled on the 4 user screenshots (firing-range dummy + ADS)."""

from __future__ import annotations

import cv2
import numpy as np

FRAME_W = 1280
FRAME_H = 720
CX = FRAME_W // 2
CY = FRAME_H // 2
RED = (0, 0, 255)
WHITE = (235, 235, 235)
GREY = (160, 160, 165)
BG_SAND = (95, 88, 78)
BG_SKY = (120, 90, 60)


def _draw_training_dummy(
    frame: np.ndarray,
    cx: int,
    foot_y: int,
    *,
    side_shift: int = 0,
    crouch: bool = False,
    occlude_bottom: float = 0.0,
) -> None:
    """Segmented plates: red face/neck/chest/waist/thighs on white/grey limbs."""
    cx += side_shift
    s = 0.85 if crouch else 1.0
    head_h, neck_h, chest_h, waist_h = int(22 * s), int(10 * s), int(38 * s), int(18 * s)
    thigh_h, shin_h = int(28 * s), int(32 * s)
    gaps = (int(6 * s), int(5 * s), int(6 * s), int(8 * s))
    total = head_h + neck_h + chest_h + waist_h + thigh_h + shin_h + sum(gaps)
    top = foot_y - total
    y = top
    # head white shell + red faceplate
    cv2.rectangle(frame, (cx - 16, y), (cx + 16, y + head_h), WHITE, -1)
    cv2.rectangle(frame, (cx - 10, y + 4), (cx + 10, y + head_h - 3), RED, -1)
    y += head_h + gaps[0]
    cv2.rectangle(frame, (cx - 7, y), (cx + 7, y + neck_h), RED, -1)
    y += neck_h + gaps[1]
    cv2.rectangle(frame, (cx - 26, y), (cx + 26, y + chest_h), WHITE, -1)
    cv2.rectangle(frame, (cx - 18, y + 6), (cx + 18, y + chest_h - 4), RED, -1)
    y += chest_h + gaps[2]
    cv2.rectangle(frame, (cx - 14, y), (cx + 14, y + waist_h), RED, -1)
    y += waist_h + gaps[3]
    for dx, thigh_x in ((-14, cx - 14), (6, cx + 6)):
        cv2.rectangle(frame, (thigh_x, y), (thigh_x + 14, y + thigh_h), RED, -1)
        cv2.rectangle(frame, (thigh_x + 1, y + thigh_h), (thigh_x + 12, y + thigh_h + shin_h), GREY, -1)
    if occlude_bottom > 0:
        cut = int(frame.shape[0] * (1.0 - occlude_bottom))
        frame[cut:, :] = (20, 20, 22)


def gen_firing_range_standing() -> np.ndarray:
    """Image 2/4: vertical dummy on sand, red plates separated by white/grey."""
    frame = np.full((FRAME_H, FRAME_W, 3), BG_SAND, dtype=np.uint8)
    _draw_training_dummy(frame, CX, CY + 130)
    return frame


def gen_ads_dummy_with_sky_balloon() -> tuple[np.ndarray, tuple[int, int]]:
    """Image 1/3: dummy lower-center + round red sky blob (must lose to body)."""
    frame = np.full((FRAME_H, FRAME_W, 3), BG_SKY, dtype=np.uint8)
    frame[: int(FRAME_H * 0.55), :] = (200, 140, 90)
    _draw_training_dummy(frame, CX, CY + 160, occlude_bottom=0.22)
    balloon_xy = (CX + 180, int(FRAME_H * 0.14))
    cv2.circle(frame, balloon_xy, 36, RED, -1)
    cv2.circle(frame, balloon_xy, 36, RED, 2)
    return frame, balloon_xy


def gen_close_vertical_dummy() -> np.ndarray:
    """Image 2 close-up: tall humanoid column."""
    frame = np.full((FRAME_H, FRAME_W, 3), BG_SAND, dtype=np.uint8)
    _draw_training_dummy(frame, CX, CY + 200, side_shift=0)
    return frame


def gen_dynamic_pose_dummy() -> np.ndarray:
    """Image 1 dynamic pose: staggered legs, weapon block (grey occluder)."""
    frame = np.full((FRAME_H, FRAME_W, 3), BG_SAND, dtype=np.uint8)
    _draw_training_dummy(frame, CX - 30, CY + 140, crouch=True)
    cv2.rectangle(frame, (CX - 50, CY + 35), (CX + 35, CY + 75), (35, 35, 40), -1)
    return frame


def gen_body_plus_balloon_compete() -> np.ndarray:
    frame, _ = gen_ads_dummy_with_sky_balloon()
    return frame
