"""Scope/sight reticle rejection — ADS reticles at the FOV centre must NOT
become detection targets.

In Apex Legends a weapon scope overlays the viewmodel near the crosshair when
ADS is active (1x/2x HCOG, 3x classic, holo dots). The bottom-22% viewmodel
exclude does not help because the reticle is dead-centre. The detector adds
a FOV-centre-aware reject (``RejectReason.SCOPE_RETICLE``) that catches
extreme-aspect lines, pip-sized solid dots, and chevron/triangle shapes
inside the inner crosshair zone, while leaving body-sized blobs untouched.
"""

from __future__ import annotations

import unittest

import cv2
import numpy as np

import detector


CX = 640
CY = 360
H = 720
W = 1280
FOV = 200


def _grey_scene() -> np.ndarray:
    return np.full((H, W, 3), 60, dtype=np.uint8)


def _draw_reticle_horizontal_line(frame: np.ndarray) -> None:
    cv2.rectangle(frame, (CX - 22, CY - 1), (CX + 22, CY + 1), (0, 0, 220), -1)


def _draw_reticle_vertical_line(frame: np.ndarray) -> None:
    cv2.rectangle(frame, (CX - 1, CY - 16), (CX + 1, CY + 16), (0, 0, 220), -1)


def _draw_reticle_dot(frame: np.ndarray, *, radius: int = 6) -> None:
    cv2.circle(frame, (CX, CY), radius, (0, 0, 230), -1)


def _draw_reticle_chevron(frame: np.ndarray) -> None:
    pts = np.array(
        [
            [CX, CY - 8],
            [CX + 8, CY + 6],
            [CX - 8, CY + 6],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(frame, [pts], (0, 0, 220))


def _draw_humanoid(
    frame: np.ndarray,
    cx: int,
    foot_y: int,
) -> None:
    """Apex-style segmented red plates (head/chest/knees) — guaranteed multi-part."""
    RED = (0, 0, 255)
    head_h, chest_h, knee_h = 26, 44, 22
    head_w, chest_w, knee_w = 28, 42, 24
    gap1, gap2 = 10, 12
    hy = foot_y - (head_h + gap1 + chest_h + gap2 + knee_h)
    cv2.rectangle(frame, (cx - head_w // 2, hy), (cx + head_w // 2, hy + head_h), RED, -1)
    cy0 = hy + head_h + gap1
    cv2.rectangle(frame, (cx - chest_w // 2, cy0), (cx + chest_w // 2, cy0 + chest_h), RED, -1)
    ky0 = cy0 + chest_h + gap2
    cv2.rectangle(frame, (cx - knee_w // 2, ky0), (cx + knee_w // 2, ky0 + knee_h), RED, -1)


def _candidate_reject_reasons(frame: np.ndarray) -> list[str]:
    candidates, _mask, _parts = detector.enumerate_candidates(
        frame,
        None,
        FOV,
        min_area=20.0,
        fov_center_x=float(CX),
        fov_center_y=float(CY),
        detection_mode="shape",
    )
    return [c.reject_reason for c in candidates]


class ScopeReticleFilterTests(unittest.TestCase):
    def test_horizontal_reticle_line_does_not_become_target(self) -> None:
        frame = _grey_scene()
        _draw_reticle_horizontal_line(frame)
        r = detector.find_best_target(
            frame, None, FOV, 20.0, float(CX), float(CY), detection_mode="shape"
        )
        self.assertFalse(r.active, "horizontal reticle line must not produce a target")
        self.assertIsNone(r.target)

    def test_vertical_reticle_line_does_not_become_target(self) -> None:
        frame = _grey_scene()
        _draw_reticle_vertical_line(frame)
        r = detector.find_best_target(
            frame, None, FOV, 20.0, float(CX), float(CY), detection_mode="shape"
        )
        self.assertFalse(r.active, "vertical reticle line must not produce a target")
        self.assertIsNone(r.target)

    def test_red_dot_sight_does_not_become_target(self) -> None:
        frame = _grey_scene()
        _draw_reticle_dot(frame, radius=5)
        r = detector.find_best_target(
            frame, None, FOV, 20.0, float(CX), float(CY), detection_mode="shape"
        )
        self.assertFalse(r.active, "red dot sight must not produce a target")
        self.assertIsNone(r.target)

    def test_chevron_reticle_does_not_become_target(self) -> None:
        frame = _grey_scene()
        _draw_reticle_chevron(frame)
        r = detector.find_best_target(
            frame, None, FOV, 20.0, float(CX), float(CY), detection_mode="shape"
        )
        self.assertFalse(r.active, "chevron reticle must not produce a target")
        self.assertIsNone(r.target)

    def test_body_with_overlaid_reticle_still_detected(self) -> None:
        """Body slightly offset from centre + reticle directly on crosshair.

        The reticle must be rejected (no extra phantom target competing with the
        body for the centre-distance bonus); the body must still be detected.
        """
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        _draw_humanoid(frame, CX + 45, CY + 80)
        _draw_reticle_horizontal_line(frame)
        _draw_reticle_dot(frame, radius=5)
        r = detector.find_best_target(
            frame, None, FOV, 60.0, float(CX), float(CY), detection_mode="shape", debug=True
        )
        self.assertTrue(r.active, msg=r.debug_lines)
        assert r.target is not None
        self.assertGreater(r.target.bbox_h, r.target.bbox_w, "selected target should be a body, not a horizontal reticle")
        self.assertGreaterEqual(r.target.bbox_h, 60)

    def test_reticle_filter_only_fires_near_fov_centre(self) -> None:
        """A small low-extent blob FAR from the FOV centre must NOT be reticle-rejected.

        This protects real targets that happen to be tiny (e.g. distant peeking
        head) from a sweeping reticle filter."""
        frame = _grey_scene()
        scale = detector._scale(W, H)
        far_part = detector._RedPart(
            contour=np.zeros((1, 1, 2), dtype=np.int32),
            x=CX + 200, y=CY + 200, w=3, h=3, area=9.0,
            cx=float(CX + 201), cy=float(CY + 201),
            aspect_wh=1.0, aspect_hw=1.0,
            solidity=1.0, extent=1.0, circularity=1.0,
        )
        near_part = detector._RedPart(
            contour=np.zeros((1, 1, 2), dtype=np.int32),
            x=CX - 1, y=CY - 1, w=3, h=3, area=9.0,
            cx=float(CX), cy=float(CY),
            aspect_wh=1.0, aspect_hw=1.0,
            solidity=1.0, extent=1.0, circularity=1.0,
        )
        self.assertTrue(
            detector._is_scope_reticle(near_part, scale, float(CX), float(CY)),
            "tiny pip-shape at FOV centre must trigger scope_reticle reject",
        )
        self.assertFalse(
            detector._is_scope_reticle(far_part, scale, float(CX), float(CY)),
            "tiny pip-shape far from FOV centre must NOT trigger scope_reticle reject",
        )


class ScopeReticleEnumerationTests(unittest.TestCase):
    """Belt-and-braces: ensure no candidate is produced for a frame whose only
    foreground content is a reticle."""

    def test_reticle_only_frame_emits_no_accepted_candidate(self) -> None:
        for draw in (
            _draw_reticle_horizontal_line,
            _draw_reticle_vertical_line,
            lambda f: _draw_reticle_dot(f, radius=5),
            _draw_reticle_chevron,
        ):
            frame = _grey_scene()
            draw(frame)
            reasons = _candidate_reject_reasons(frame)
            with self.subTest(draw=getattr(draw, "__name__", "lambda")):
                self.assertNotIn("ok", reasons, reasons)


if __name__ == "__main__":
    unittest.main()
