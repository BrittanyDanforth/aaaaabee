"""D1+D2+D3 regression: detect a Horizon-style character with a thin red glow.

The user reported in live Apex play that subtle red-glow enemies (Horizon
white character with a 2-px red shimmer at HSV~(0,145,130)) were being
ignored entirely. The synthetic scene below approximates that case:

* a white body silhouette (head + chest + arms + legs);
* a 2-px-wide red glow tracing each body part contour at HSV(0,145,130),
  i.e. S=145, V=130 — well below the previous gate (S>=80 V>=70 with
  MORPH_OPEN(3) erasing the thin ring).

Before the audit fix the detector ignored this character. After the fix
(D1 — MORPH_OPEN shrunk so 2-px rings survive; D2 — S>=55 V>=50; D3 —
lowered red-coverage softener thresholds + outline-coverage path) the
character must be ``active=True`` with red_coverage > 0 and confidence
comfortably above the floor.
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


def _hsv_to_bgr(h: int, s: int, v: int) -> tuple[int, int, int]:
    px = np.array([[[h, s, v]]], dtype=np.uint8)
    bgr = cv2.cvtColor(px, cv2.COLOR_HSV2BGR)
    return tuple(int(c) for c in bgr[0, 0])


def _draw_glow_horizon_character(frame: np.ndarray, cx: int, cy: int) -> None:
    halo = _hsv_to_bgr(0, 145, 130)
    white = (220, 220, 220)
    grey = (180, 180, 180)
    cv2.ellipse(frame, (cx, cy - 50), (20, 24), 0, 0, 360, halo, -1)
    cv2.ellipse(frame, (cx, cy - 50), (18, 22), 0, 0, 360, white, -1)
    cv2.rectangle(frame, (cx - 32, cy - 27), (cx + 32, cy + 47), halo, -1)
    cv2.rectangle(frame, (cx - 30, cy - 25), (cx + 30, cy + 45), white, -1)
    cv2.rectangle(frame, (cx - 48, cy - 17), (cx - 28, cy + 32), halo, -1)
    cv2.rectangle(frame, (cx - 46, cy - 15), (cx - 30, cy + 30), grey, -1)
    cv2.rectangle(frame, (cx + 28, cy - 17), (cx + 48, cy + 32), halo, -1)
    cv2.rectangle(frame, (cx + 30, cy - 15), (cx + 46, cy + 30), grey, -1)
    cv2.rectangle(frame, (cx - 22, cy + 43), (cx, cy + 107), halo, -1)
    cv2.rectangle(frame, (cx - 20, cy + 45), (cx - 2, cy + 105), white, -1)
    cv2.rectangle(frame, (cx, cy + 43), (cx + 22, cy + 107), halo, -1)
    cv2.rectangle(frame, (cx + 2, cy + 45), (cx + 20, cy + 105), white, -1)


class ApexGlowDetectionTests(unittest.TestCase):
    def test_horizon_glow_character_is_detected(self) -> None:
        frame = np.full((H, W, 3), 35, dtype=np.uint8)
        _draw_glow_horizon_character(frame, CX, CY)
        r = detector.find_best_target(
            frame, [], FOV, MIN_AREA, float(CX), float(CY),
            debug=True, detection_mode="apex",
        )
        self.assertTrue(
            r.active,
            "Horizon-style glow character must be detected; "
            f"debug={r.debug_lines[-5:]}",
        )
        assert r.target is not None
        self.assertGreater(r.target.red_coverage, 0.0)
        self.assertGreater(r.target.confidence, 0.30)
        # Bbox should cover the character (~100x190 with glow border).
        self.assertGreater(r.target.bbox_h, 140)
        self.assertGreater(r.target.bbox_w, 60)

    def test_hsv_red_gate_catches_dim_glow_pixel(self) -> None:
        """D2 sanity: a single pixel at HSV(0, 145, 130) must pass the
        red-enemy mask gate. Previously S>=80 V>=70 was OK but the
        MORPH_OPEN(3) erased lone pixels; D1 keeps them when raw mask is
        sparse. Tests the threshold + open-kernel invariants together."""
        frame = np.full((H, W, 3), 35, dtype=np.uint8)
        # Draw a small 3x3 patch of the glow colour.
        halo = _hsv_to_bgr(0, 145, 130)
        cv2.rectangle(frame, (CX - 1, CY - 1), (CX + 2, CY + 2), halo, -1)
        red = detector.build_red_enemy_mask(frame)
        self.assertGreater(
            int((red > 0).sum()),
            0,
            "build_red_enemy_mask must keep a small glow patch alive (D1+D2 audit fixes)",
        )


if __name__ == "__main__":
    unittest.main()
