"""OS / display helpers (Windows-first, safe on Linux for CI)."""

from __future__ import annotations

import platform
import sys
from typing import Any


def is_windows() -> bool:
    return sys.platform == "win32"


def enable_dpi_awareness() -> str:
    """Best-effort DPI awareness so capture coords match cursor on Windows."""
    if not is_windows():
        return "skipped (not Windows)"
    try:
        import ctypes

        # Per-monitor V2 (Windows 10+)
        awareness = ctypes.c_int(-4)  # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
        if ctypes.windll.user32.SetProcessDpiAwarenessContext(awareness):
            return "PerMonitorV2"
        # System DPI aware
        if ctypes.windll.user32.SetProcessDPIAware():
            return "SystemDPIAware"
    except Exception as exc:  # noqa: BLE001 — report to user log
        return f"failed: {exc}"
    return "unavailable"


def list_monitors_mss() -> list[dict[str, Any]]:
    import mss

    with mss.mss() as sct:
        out = []
        for i, mon in enumerate(sct.monitors):
            if i == 0:
                continue  # virtual desktop aggregate
            out.append(
                {
                    "index": i,
                    "left": mon["left"],
                    "top": mon["top"],
                    "width": mon["width"],
                    "height": mon["height"],
                }
            )
        return out


def platform_summary() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "os": platform.platform(),
        "machine": platform.machine(),
    }
