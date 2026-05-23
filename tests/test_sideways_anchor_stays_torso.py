"""D5 regression: sideways-viewed character anchor must stay on torso.

The user reported that when a character is viewed from the side with a
gun arm extended, the aim dot drifts onto the weapon rather than the
chest column. The previous chest anchor used ``np.mean(xs)`` over the
red-mask roi in the upper chest band, which mass-pulls toward an
extended limb. D5 (audit) switches to the densest-column (smoothed
peak) and raises the multi-part torso role X blend.

The test draws a sideways character: torso left of centre, gun arm
extending right; computes the anchor; verifies it lands within +/-15 %
of the bbox width of the true torso centre.
"""

from __future__ import annotations

import unittest

import cv2
import numpy as np

import detector

H, W = 720, 1280
CX, CY = W // 2, H // 2
FOV = 220
MIN_AREA = 40.0


def _draw_sideways(frame: np.ndarray, cx: int, cy: int) -> int:
    """Return the true torso centre x coordinate."""
    red = (40, 40, 220)
    dark_red = (40, 40, 200)
    cv2.ellipse(frame, (cx - 20, cy - 50), (16, 20), 0, 0, 360, red, -1)
    # Torso spans (cx - 40, cy - 25) to (cx, cy + 45) — centre x = cx - 20.
    torso_center_x = cx - 20
    cv2.rectangle(frame, (cx - 40, cy - 25), (cx, cy + 45), red, -1)
    # Gun arm extends to the right of torso.
    cv2.rectangle(frame, (cx, cy - 8), (cx + 50, cy + 4), dark_red, -1)
    cv2.rectangle(frame, (cx + 40, cy - 4), (cx + 52, cy + 10), dark_red, -1)
    # Legs.
    cv2.rectangle(frame, (cx - 36, cy + 45), (cx - 20, cy + 105), red, -1)
    cv2.rectangle(frame, (cx - 16, cy + 45), (cx, cy + 105), red, -1)
    return torso_center_x


class SidewaysAnchorTests(unittest.TestCase):
    def test_anchor_x_stays_within_15pct_of_torso_center(self) -> None:
        frame = np.full((H, W, 3), 35, dtype=np.uint8)
        torso_cx = _draw_sideways(frame, CX, CY)
        r = detector.find_best_target(
            frame, [], FOV, MIN_AREA, float(CX), float(CY),
            debug=True, detection_mode="apex",
        )
        self.assertTrue(
            r.active,
            f"sideways character must detect; debug={r.debug_lines[-5:]}",
        )
        assert r.target is not None
        # D5 requirement: anchor X within +/-15 % of bbox_w from torso centre.
        tol = max(8.0, 0.15 * float(r.target.bbox_w))
        offset = abs(r.target.centroid_x - float(torso_cx))
        self.assertLess(
            offset,
            tol,
            f"anchor drifted onto gun arm: offset={offset:.1f}px "
            f"tol={tol:.1f}px (torso_x={torso_cx}, "
            f"anchor_x={r.target.centroid_x:.1f}, bbox_w={r.target.bbox_w})",
        )


if __name__ == "__main__":
    unittest.main()
