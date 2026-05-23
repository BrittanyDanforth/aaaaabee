"""Firing-range board at crosshair must lose to humanoid dummy at FOV edge."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

import detector

RED = (0, 0, 255)
H, W = 720, 1280
CX, CY = W // 2, H // 2
FOV = 220
MIN_AREA = 40.0
HSV_RED = [
    {"lower": [0, 100, 100], "upper": [12, 255, 255]},
    {"lower": [170, 100, 100], "upper": [180, 255, 255]},
]


def _apex_dummy(frame: np.ndarray, cx: int, foot_y: int, *, scale: float = 1.0) -> None:
    s = scale
    head_h, chest_h, knee_h = int(26 * s), int(44 * s), int(22 * s)
    head_w, chest_w, knee_w = int(28 * s), int(42 * s), int(24 * s)
    gaps = (int(10 * s), int(12 * s))
    total_h = head_h + gaps[0] + chest_h + gaps[1] + knee_h
    hy = foot_y - total_h
    cv2.rectangle(
        frame, (cx - head_w // 2, hy), (cx - head_w // 2 + head_w, hy + head_h), RED, -1
    )
    cy0 = hy + head_h + gaps[0]
    cv2.rectangle(
        frame,
        (cx - chest_w // 2, cy0),
        (cx - chest_w // 2 + chest_w, cy0 + chest_h),
        RED,
        -1,
    )
    ky0 = cy0 + chest_h + gaps[1]
    cv2.rectangle(
        frame,
        (cx - knee_w // 2, ky0),
        (cx - knee_w // 2 + knee_w, ky0 + knee_h),
        RED,
        -1,
    )


def _range_board(frame: np.ndarray, cx: int, cy: int, *, h: int = 200, w: int = 70) -> None:
    """Tall saturated red column (practice board outline) near crosshair."""
    x0 = cx - w // 2
    y0 = cy - h // 2
    cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), RED, -1)
    cv2.rectangle(frame, (x0 + 8, y0 + 8), (x0 + w - 8, y0 + h - 8), (0, 0, 180), -1)


class RangeBoardPriorityTests(unittest.TestCase):
    def test_edge_dummy_beats_center_range_board(self) -> None:
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        _range_board(frame, CX, CY - 40, h=210, w=72)
        _apex_dummy(frame, CX + 165, CY + 110, scale=1.25)
        r = detector.find_best_target(
            frame,
            HSV_RED,
            FOV,
            MIN_AREA,
            float(CX),
            float(CY),
            stickiness_pixels=90.0,
            debug=True,
            detection_mode="apex",
        )
        self.assertTrue(r.active, "\n".join(r.debug_lines))
        assert r.target is not None
        self.assertGreater(
            r.target.centroid_x,
            CX + 80,
            "closer humanoid at FOV edge should beat central range board\n"
            + "\n".join(r.debug_lines[-8:]),
        )
        self.assertGreaterEqual(int(r.target.part_count), 2)

    def test_range_board_fp_classifier(self) -> None:
        from detector import Target

        board = Target(
            centroid_x=float(CX),
            centroid_y=float(CY),
            area=12000.0,
            distance_to_center=10.0,
            confidence=0.8,
            bbox_x=CX - 35,
            bbox_y=CY - 100,
            bbox_w=70,
            bbox_h=200,
            body_shape_score=0.62,
            part_count=1,
            red_coverage=0.35,
            has_classified_torso=False,
            torso_score=0.1,
            head_score=0.0,
            fill_ratio=0.72,
            max_circularity=0.4,
        )
        self.assertTrue(detector.target_is_range_board_fp(board, motion_overlap=0.0))


if __name__ == "__main__":
    unittest.main()
