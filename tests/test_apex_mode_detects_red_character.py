"""Bug-1 regression: DETECTION_MODE_APEX must detect a clearly red Apex
character standing dead-centre in the FOV.

The user reported during live Apex play that a red enemy (red helmet, red
armour) stood dead-centre and the aim-assist dot pinned to the FOV centre
instead of locking on. Root cause: the old ``build_red_outline_mask``
applied ``cv2.MORPH_GRADIENT`` to the filled red HSV mask, leaving only a
thin perimeter ribbon. Combined with the shape mask, this:

* collapsed the body's vertical-profile signal (no transitions = v_score
  drops to zero on uniformly red columns);
* dropped the candidate's confidence below ``_MIN_CONFIDENCE``;
* and the runtime never returned ``active=True``, so the dot pinned to
  centre with no live target.

The fix replaces the outline-only mask with a filled red-enemy mask used
to compute ``Target.red_coverage`` per candidate. ``score_target`` then
softens the single-part penalty when a tall red-humanoid bbox is observed
and adds a small additive bonus — enough to clear the confidence gate on
a real red enemy without re-activating the synthetic blurred-blob false
positives covered by ``test_synthetic_frames``.
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


def _draw_red_apex_character(frame: np.ndarray, cx: int, cy: int) -> None:
    """Approximate a real Apex enemy: red helmet, red chest armour, red arms
    and red leg armour. Uses BGR (40, 40, 220) — close to the actual in-game
    red enemy outline saturation. Whole body is one connected silhouette
    (analyze_figure sees ``part_count == 1``), which is the case that fails
    detection without the red-coverage softener."""
    red = (40, 40, 220)
    dark_red = (40, 40, 200)
    cv2.ellipse(frame, (cx, cy - 50), (18, 22), 0, 0, 360, red, -1)
    cv2.rectangle(frame, (cx - 30, cy - 25), (cx + 30, cy + 45), red, -1)
    cv2.rectangle(frame, (cx - 46, cy - 15), (cx - 30, cy + 30), dark_red, -1)
    cv2.rectangle(frame, (cx + 30, cy - 15), (cx + 46, cy + 30), dark_red, -1)
    cv2.rectangle(frame, (cx - 20, cy + 45), (cx - 2, cy + 105), red, -1)
    cv2.rectangle(frame, (cx + 2, cy + 45), (cx + 20, cy + 105), red, -1)


class ApexModeRedCharacterTests(unittest.TestCase):
    def test_red_character_at_fov_center_is_detected(self) -> None:
        """Live-game repro: red Apex enemy dead-centre. The detector MUST
        return ``active=True`` with a bbox covering the character."""
        frame = np.full((H, W, 3), 30, dtype=np.uint8)
        _draw_red_apex_character(frame, CX, CY)

        r = detector.find_best_target(
            frame,
            [],
            FOV,
            MIN_AREA,
            float(CX),
            float(CY),
            debug=True,
            detection_mode="apex",
        )
        self.assertTrue(
            r.active,
            "apex mode must detect the centre red enemy; "
            f"debug={r.debug_lines[-5:]}",
        )
        assert r.target is not None
        # The detected bbox must cover the character — character spans
        # roughly (CX-46, CY-72) to (CX+46, CY+105).
        self.assertGreater(r.target.bbox_w, 70)
        self.assertGreater(r.target.bbox_h, 140)
        # And the aim point must land near the centre of the FOV (within
        # the character's body, not on a peripheral artefact).
        from math import hypot

        d = hypot(r.target.centroid_x - CX, r.target.centroid_y - CY)
        self.assertLess(
            d,
            120.0,
            f"aim point too far from centre red enemy: d={d:.1f}",
        )

    def test_red_coverage_is_recorded_on_target(self) -> None:
        """``Target.red_coverage`` should be near 1 for a uniformly red
        character — this is the signal that gates the single-part softener."""
        frame = np.full((H, W, 3), 30, dtype=np.uint8)
        _draw_red_apex_character(frame, CX, CY)
        r = detector.find_best_target(
            frame, [], FOV, MIN_AREA, float(CX), float(CY), debug=True,
            detection_mode="apex",
        )
        self.assertTrue(r.active)
        assert r.target is not None
        self.assertGreater(
            r.target.red_coverage,
            0.30,
            f"expected high red coverage on red character, got "
            f"{r.target.red_coverage:.2f}",
        )

    def test_non_red_mode_does_not_record_red_coverage(self) -> None:
        """Shape mode must not pay the red-mask cost — ``red_coverage`` is
        expected to stay at zero."""
        frame = np.full((H, W, 3), 30, dtype=np.uint8)
        _draw_red_apex_character(frame, CX, CY)
        r = detector.find_best_target(
            frame, [], FOV, MIN_AREA, float(CX), float(CY), debug=True,
            detection_mode="shape",
        )
        if r.target is not None:
            self.assertEqual(r.target.red_coverage, 0.0)


class BlurredBlobMustStayRejectedTests(unittest.TestCase):
    """The red-coverage softener must NOT re-activate the synthetic
    ``single_merged_blob_from_blur`` generator covered by
    ``test_synthetic_frames``. Pin the rejection here as well so the next
    regression run catches a drift back to over-permissive scoring."""

    def test_blurred_red_blob_still_inactive(self) -> None:
        red = (0, 0, 255)
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        head_h, chest_h, knee_h = 26, 44, 22
        head_w, chest_w, knee_w = 28, 42, 24
        foot_y = CY + 90
        total_h = head_h + chest_h + knee_h
        hy = foot_y - total_h
        cv2.rectangle(frame, (CX - head_w // 2, hy), (CX + head_w // 2, hy + head_h), red, -1)
        cy0 = hy + head_h
        cv2.rectangle(frame, (CX - chest_w // 2, cy0), (CX + chest_w // 2, cy0 + chest_h), red, -1)
        ky0 = cy0 + chest_h
        cv2.rectangle(frame, (CX - knee_w // 2, ky0), (CX + knee_w // 2, ky0 + knee_h), red, -1)
        merged = cv2.GaussianBlur(frame, (21, 21), 0)
        r = detector.find_best_target(
            merged, [], FOV, MIN_AREA, float(CX), float(CY), debug=True,
            detection_mode="apex",
        )
        self.assertFalse(r.active, r.debug_lines)


if __name__ == "__main__":
    unittest.main()
