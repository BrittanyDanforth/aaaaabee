"""DPI awareness stub for Windows; no-op elsewhere."""

from __future__ import annotations


def enable_dpi_awareness() -> str:
    try:
        import ctypes

        if hasattr(ctypes, "windll"):
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
            return "per-monitor-v2"
    except Exception:
        pass
    return "default"
