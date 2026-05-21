"""Frame-by-frame pull / gate trace logging — default ON for live profiles (file only)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("overlay_assist.pull_trace")

_configured = False
_log_path: Path | None = None


@dataclass
class PullTraceFrame:
    frame: int
    raw_target: tuple[float, float]
    motion_target: tuple[float, float]
    center: tuple[float, float]
    error: tuple[float, float]
    pull_dxdy: tuple[int, int]
    pull_mag: float
    pull_vel: tuple[float, float] = (0.0, 0.0)
    pull_desired: tuple[float, float] = (0.0, 0.0)
    gate_allowed: bool = True
    gate_reason: str = ""
    mouse_move_called: tuple[int, int] = (0, 0)
    detection_fresh: bool = False
    target_lost_frames: int = 0
    stale_detection: bool = False
    has_target: bool = True
    ads_active: bool = True


def format_trace_line(t: PullTraceFrame) -> str:
    ex, ey = t.error
    return (
        f"frame={t.frame}\n"
        f"has_target={t.has_target}\n"
        f"ads_active={t.ads_active}\n"
        f"raw_target=({t.raw_target[0]:.1f},{t.raw_target[1]:.1f})\n"
        f"motion_target=({t.motion_target[0]:.1f},{t.motion_target[1]:.1f})\n"
        f"center=({t.center[0]:.1f},{t.center[1]:.1f})\n"
        f"error=({ex:.1f},{ey:.1f})\n"
        f"pull_dxdy=({t.pull_dxdy[0]},{t.pull_dxdy[1]})\n"
        f"pull_mag={t.pull_mag:.2f}\n"
        f"pull_vel=({t.pull_vel[0]:.2f},{t.pull_vel[1]:.2f})\n"
        f"pull_desired=({t.pull_desired[0]:.2f},{t.pull_desired[1]:.2f})\n"
        f"detection_fresh={t.detection_fresh}\n"
        f"target_lost_frames={t.target_lost_frames}\n"
        f"stale_detection={t.stale_detection}\n"
        f"gate_allowed={t.gate_allowed}\n"
        f"gate_reason={t.gate_reason or ''}\n"
        f"mouse_move_called=({t.mouse_move_called[0]},{t.mouse_move_called[1]})"
    )


def trace_log_path() -> Path | None:
    return _log_path


def setup_trace_logging(config: dict[str, Any], *, app_root: Path | None = None) -> bool:
    """Configure file/console handlers. Live default: file only, no console spam."""
    global _configured, _log_path
    enabled = bool(config.get("trace_pull", False))
    if not enabled:
        return False

    root = app_root or Path.cwd()
    log_path = Path(str(config.get("trace_pull_log_file", "logs/pull_trace.log")))
    if not log_path.is_absolute():
        log_path = root / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _log_path = log_path

    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8", mode="a")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    if bool(config.get("trace_pull_console", False)):
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    _configured = True
    logger.info(
        "=== pull trace ON profile=%s interval=%s max_frames=%s file=%s ===",
        config.get("profile", ""),
        config.get("trace_pull_interval_frames", 1),
        config.get("trace_pull_max_frames", 0),
        log_path,
    )
    return True


def should_log_frame(frame_index: int, config: dict[str, Any]) -> bool:
    max_frames = int(config.get("trace_pull_max_frames", 0))
    if max_frames > 0 and frame_index > max_frames:
        return False
    interval = max(1, int(config.get("trace_pull_interval_frames", 1)))
    return (frame_index % interval) == 0


def log_trace_frame(t: PullTraceFrame, config: dict[str, Any] | None = None) -> None:
    if not _configured:
        return
    if config is not None and not should_log_frame(t.frame, config):
        return
    logger.info("%s", format_trace_line(t))


def analyze_trace_file(path: Path, *, tail_blocks: int = 5) -> dict[str, Any]:
    """Quick parse of recent trace blocks for audits."""
    if not path.is_file():
        return {"ok": False, "reason": "missing log"}
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = [b.strip() for b in text.split("frame=") if b.strip()]
    recent = blocks[-tail_blocks:] if blocks else []
    gate_blocks = sum(1 for b in recent if "gate_allowed=False" in b)
    zero_mouse = sum(1 for b in recent if "mouse_move_called=(0,0)" in b and "error=(" in b)
    return {
        "ok": True,
        "blocks": len(blocks),
        "recent_gate_blocks": gate_blocks,
        "recent_zero_mouse_with_error": zero_mouse,
    }
