"""O1 regression: only ONE visible FOV ring in the debug window by default.

The user's screenshots showed two concentric green rings — a smaller
inner ring from the overlay (drawn at display_fov) and a larger outer
ring from the OpenCV debug window (drawn at detect_fov). After the O1
audit fix:

* draw_debug takes a display_fov_radius parameter and draws the green
  ring at THAT radius (matching the live overlay) by default;
* the debug_show_detect_ring config flag (default False) controls
  whether a SECOND faint ring is drawn at the detection FOV. Default
  off means only ONE ring is ever visible.

This test verifies both invariants from the source side (no Tk needed).
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import cv2
import numpy as np

import detector
import config_validation

DETECTOR_PATH = Path(__file__).resolve().parents[1] / "detector.py"
RUNTIME_PATH = Path(__file__).resolve().parents[1] / "runtime.py"


class DebugRingDefaultsTests(unittest.TestCase):
    def test_draw_debug_signature_accepts_display_fov(self) -> None:
        text = DETECTOR_PATH.read_text(encoding="utf-8")
        self.assertIn(
            "display_fov_radius: int | None = None",
            text,
            "draw_debug must take a display_fov_radius keyword (O1 audit fix)",
        )
        self.assertIn(
            "debug_show_detect_ring: bool = False",
            text,
            "draw_debug must gate the detect-FOV ring on a config flag "
            "(default False) so only one ring is visible (O1 audit fix)",
        )

    def test_runtime_passes_display_fov_to_draw_debug(self) -> None:
        text = RUNTIME_PATH.read_text(encoding="utf-8")
        self.assertIn(
            "display_fov_radius=overlay_fov",
            text,
            "runtime must forward display_fov to draw_debug (O1 audit fix)",
        )

    def test_config_validation_normalises_flag(self) -> None:
        # Load the canonical config.json (has all required keys) and verify
        # that validate_config defaults debug_show_detect_ring to False.
        from assist import load_config
        repo_root = Path(__file__).resolve().parents[1]
        cfg = load_config(repo_root / "config.json")
        out = config_validation.validate_config(cfg)
        self.assertIn("debug_show_detect_ring", out)
        self.assertFalse(
            out["debug_show_detect_ring"],
            "debug_show_detect_ring must default to False so only ONE "
            "FOV ring is visible (O1 audit fix)",
        )

    def test_draw_debug_default_draws_single_ring(self) -> None:
        frame = np.full((720, 1280, 3), 30, dtype=np.uint8)
        # display_fov < detect_fov is the user's screenshot scenario.
        out = detector.draw_debug(
            frame, None, fov_radius=220, fov_center_x=640.0, fov_center_y=360.0,
            display_fov_radius=180,
        )
        # Sweep along the horizontal centerline: count rows with a strong
        # green ring transition. There should be ONE band of green pixels
        # (the display ring), not two.
        green_intensities = []
        cy = 360
        for r in range(50, 260):
            px = out[cy, 640 + r]
            green_intensities.append(int(px[1]) - max(int(px[0]), int(px[2])))
        # Find peaks: where green dominance > 80.
        peaks = []
        i = 0
        while i < len(green_intensities):
            if green_intensities[i] > 80:
                j = i
                while j < len(green_intensities) and green_intensities[j] > 80:
                    j += 1
                peaks.append((i + 50, j + 50))
                i = j
            else:
                i += 1
        self.assertEqual(
            len(peaks), 1,
            f"expected ONE green ring crossing in default debug draw; "
            f"got {len(peaks)} ({peaks})",
        )
        # The ring must align with the DISPLAY radius (180), not the
        # detection radius (220).
        ring_r = (peaks[0][0] + peaks[0][1]) / 2.0
        self.assertAlmostEqual(ring_r, 180.0, delta=3.0)

    def test_draw_debug_opt_in_shows_second_ring(self) -> None:
        frame = np.full((720, 1280, 3), 30, dtype=np.uint8)
        out = detector.draw_debug(
            frame, None, fov_radius=220, fov_center_x=640.0, fov_center_y=360.0,
            display_fov_radius=180,
            debug_show_detect_ring=True,
        )
        cy = 360
        green_intensities = []
        for r in range(50, 260):
            px = out[cy, 640 + r]
            green_intensities.append(int(px[1]) - max(int(px[0]), int(px[2])))
        peaks = []
        i = 0
        while i < len(green_intensities):
            if green_intensities[i] > 40:
                j = i
                while j < len(green_intensities) and green_intensities[j] > 40:
                    j += 1
                peaks.append((i + 50, j + 50))
                i = j
            else:
                i += 1
        self.assertEqual(
            len(peaks), 2,
            f"opt-in second ring must produce TWO crossings; got {len(peaks)} ({peaks})",
        )


if __name__ == "__main__":
    unittest.main()
