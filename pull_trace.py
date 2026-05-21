"""Frame-by-frame pull / gate trace logging for lag audits."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger("overlay_assist.pull_trace")


@dataclass
class PullTraceFrame:
    frame: int
    raw_target: tuple[float, float]
    motion_target: tuple[float, float]
    center: tuple[float, float]
    error: tuple[float, float]
    pull_dxdy: tuple[int, int]
    pull_mag: float
    pull_vel: tuple[float, float]
    pull_desired: tuple[float, float]
    gate_allowed: bool
    gate_reason: str
    mouse_move_called: tuple[int, int]
    detection_fresh: bool
    target_lost_frames: int
    stale_detection: bool


def format_trace_line(t: PullTraceFrame) -> str:
    ex, ey = t.error
    return (
        f"frame={t.frame}\n"
        f"raw_target=({t.raw_target[0]:.1f},{t.raw_target[1]:.1f})\n"
        f"motion_target=({t.motion_target[0]:.1f},{t.motion_target[1]:.1f})\n"
        f"center=({t.center[0]:.1f},{t.center[1]:.1f})\n"
        f"error=({ex:.1f},{ey:.1f})\n"
        f"pull_dxdy=({t.pull_dxdy[0]},{t.pull_dxdy[1]})\n"
        f"pull_mag={t.pull_mag:.2f}\n"
        f"pull_vel=({t.pull_vel[0]:.2f},{t.pull_vel[1]:.2f})\n"
        f"detection_fresh={t.detection_fresh}\n"
        f"target_lost_frames={t.target_lost_frames}\n"
        f"stale_detection={t.stale_detection}\n"
        f"gate_allowed={t.gate_allowed}\n"
        f"gate_reason={t.gate_reason or ''}\n"
        f"mouse_move_called=({t.mouse_move_called[0]},{t.mouse_move_called[1]})"
    )


def log_trace_frame(t: PullTraceFrame) -> None:
    logger.info("%s", format_trace_line(t))
