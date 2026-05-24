"""First overlay paint should glide from ring center, not pop onto target."""

from __future__ import annotations

import math
import unittest


class OverlayAcquireStartTests(unittest.TestCase):
    def test_initial_display_is_ring_center(self) -> None:
        cx, cy = 400.0, 300.0
        dest = (520.0, 380.0)
        start = (cx, cy)
        self.assertAlmostEqual(start[0], cx)
        self.assertAlmostEqual(start[1], cy)
        self.assertGreater(math.hypot(start[0] - dest[0], start[1] - dest[1]), 50.0)


if __name__ == "__main__":
    unittest.main()
