"""Screen capture region around crosshair — FOV crop with padding for edge targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CaptureRegion:
    """MSS grab box plus offset to map frame coords -> monitor coords."""

    left: int
    top: int
    width: int
    height: int
    offset_x: float
    offset_y: float

    def as_mss_dict(self) -> dict[str, int]:
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
        }


def build_capture_region(
    monitor: dict[str, Any],
    center_x: float,
    center_y: float,
    fov_radius: int,
    *,
    use_crop: bool = True,
    crop_padding: float = 1.3,
) -> CaptureRegion:
    """
    Build a square capture around (center_x, center_y) in monitor space.
    fov_radius should include detection margin; crop_padding expands past that.
    """
    mon_left = int(monitor["left"])
    mon_top = int(monitor["top"])
    mon_w = int(monitor["width"])
    mon_h = int(monitor["height"])

    if use_crop:
        half = max(40, int(fov_radius * max(1.0, crop_padding)))
    else:
        half = max(mon_w, mon_h) // 2

    cx = mon_left + int(round(center_x))
    cy = mon_top + int(round(center_y))

    left = max(mon_left, cx - half)
    top = max(mon_top, cy - half)
    right = min(mon_left + mon_w, cx + half)
    bottom = min(mon_top + mon_h, cy + half)
    width = max(1, right - left)
    height = max(1, bottom - top)

    return CaptureRegion(
        left=left,
        top=top,
        width=width,
        height=height,
        offset_x=float(left - mon_left),
        offset_y=float(top - mon_top),
    )


def grab_bgr(sct: Any, region: CaptureRegion) -> np.ndarray:
    """Grab BGR frame from mss; returns empty array on failure."""
    import cv2

    shot = sct.grab(region.as_mss_dict())
    frame = np.asarray(shot)
    if frame.size == 0:
        return np.zeros((0, 0, 3), dtype=np.uint8)
    return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)


def to_monitor_coords(frame_x: float, frame_y: float, region: CaptureRegion) -> tuple[float, float]:
    return region.offset_x + frame_x, region.offset_y + frame_y


def frame_from_monitor(mx: float, my: float, region: CaptureRegion) -> tuple[float, float]:
    return mx - region.offset_x, my - region.offset_y
