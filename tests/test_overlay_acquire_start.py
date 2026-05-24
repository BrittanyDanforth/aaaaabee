"""First overlay paint should glide from ring center, not pop onto target."""

from __future__ import annotations

import math
import unittest


class OverlayAcquireStartTests(unittest.TestCase):
    def test_initial_display_is_ring_center(self) -> None:
        try:
            import tkinter as _tk  # noqa: F401
        except ImportError:
            self.skipTest("tkinter not available")
        from overlay_window import OverlayWindow

        win = OverlayWindow.__new__(OverlayWindow)
        win._cx = 400.0
        win._cy = 300.0
        dest = (520.0, 380.0)
        start = win._initial_dot_display(dest)
        self.assertAlmostEqual(start[0], 400.0)
        self.assertAlmostEqual(start[1], 300.0)
        self.assertGreater(math.hypot(start[0] - dest[0], start[1] - dest[1]), 50.0)


if __name__ == "__main__":
    unittest.main()
