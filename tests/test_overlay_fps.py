"""Overlay refresh rate: config helpers + overlay tick contract."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock

from profiles import (
    PROFILE_APEX_STYLE_LIVE_TRACE,
    apply_profile,
    effective_capture_fps,
    effective_overlay_fps,
)

OVERLAY_PATH = Path(__file__).resolve().parents[1] / "overlay_window.py"
RUNTIME_PATH = Path(__file__).resolve().parents[1] / "runtime.py"


class OverlayFpsConfigTests(unittest.TestCase):
    def test_live_trace_profile_overlay_at_least_capture(self) -> None:
        cfg = apply_profile({"profile": PROFILE_APEX_STYLE_LIVE_TRACE})
        cap = effective_capture_fps(cfg)
        ov = effective_overlay_fps(cfg)
        self.assertGreaterEqual(ov, cap)
        self.assertGreaterEqual(cfg["overlay_fps"], 60)
        self.assertGreaterEqual(cfg["capture_fps"], 60)

    def test_overlay_fps_never_below_capture(self) -> None:
        cfg = apply_profile(
            {
                "profile": PROFILE_APEX_STYLE_LIVE_TRACE,
                "capture_fps": 90,
                "overlay_fps": 30,
            }
        )
        self.assertGreaterEqual(effective_overlay_fps(cfg), 90)


class OverlayFpsSourceTests(unittest.TestCase):
    def test_overlay_window_uses_configurable_tick(self) -> None:
        text = OVERLAY_PATH.read_text(encoding="utf-8")
        self.assertIn("overlay_fps", text)
        self.assertIn("self._tick_ms", text)
        self.assertIn("_request_redraw", text)
        self.assertNotIn("after(33, tick)", text)

    def test_runtime_wires_overlay_fps_and_dot_alpha(self) -> None:
        text = RUNTIME_PATH.read_text(encoding="utf-8")
        self.assertIn("effective_overlay_fps", text)
        self.assertIn("set_overlay_fps", text)
        self.assertIn("overlay_dot_smooth_alpha", text)
        self.assertIn("set_dot_glide_alpha", text)


try:
    import tkinter as _tk  # noqa: F401

    _TK = True
except ImportError:
    _TK = False


@unittest.skipUnless(_TK, "tkinter not available")
class OverlayImmediateRedrawTests(unittest.TestCase):
    def test_set_state_schedules_redraw(self) -> None:
        from overlay_window import OverlayWindow

        win = OverlayWindow(800, 600, fov_radius=120, overlay_fps=90)
        root = MagicMock()
        win._root = root
        win._canvas = MagicMock()
        win._fov_id = 1
        win._target_id = 2
        win._cx = 400
        win._cy = 300
        win._drawn_fov_radius = 120
        win._closed = False

        win.set_state(True, (500.0, 400.0))
        root.after.assert_called_with(0, win._redraw)


if __name__ == "__main__":
    unittest.main()
