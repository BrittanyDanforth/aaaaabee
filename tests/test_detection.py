"""Structure-aware detection: segmented Apex dummy, reject walls/solid red."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

import detection


HSV_RED = [{"lower": [0, 120, 120], "upper": [12, 255, 255]}]


def _apex_training_dummy(frame: np.ndarray, cx: int, foot_y: int) -> None:
    """Head + chest + knee plates with gaps (white/black), not one solid blob."""
    head_w, head_h = 28, 26
    chest_w, chest_h = 40, 44
    knee_w, knee_h = 22, 20
    red = (0, 0, 230)

    hx = cx - head_w // 2
    hy = foot_y - 118
    cv2.rectangle(frame, (hx, hy), (hx + head_w, hy + head_h), red, -1)

    cx0 = cx - chest_w // 2
    cy0 = hy + head_h + 10
    cv2.rectangle(frame, (cx0, cy0), (cx0 + chest_w, cy0 + chest_h), red, -1)

    kx = cx - knee_w // 2
    ky0 = cy0 + chest_h + 12
    cv2.rectangle(frame, (kx, ky0), (kx + knee_w, ky0 + knee_h), red, -1)


def _red_horizontal_stripe(frame: np.ndarray, y: int, h: int = 22) -> None:
    frame[y : y + h, :] = (0, 0, 230)


def _solid_red_block(frame: np.ndarray, cx: int, cy: int, w: int, h: int) -> None:
    x1, y1 = cx - w // 2, cy - h // 2
    cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 240), -1)


class DetectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.h, self.w = 480, 640
        self.fov = 160
        self.cx = self.w / 2
        self.cy = self.h / 2

    def _detect(self, frame: np.ndarray):
        return detection.find_best_target(
            frame,
            HSV_RED,
            self.fov,
            min_area=80.0,
            fov_center_x=self.cx,
            fov_center_y=self.cy,
        )

    def test_detects_segmented_apex_dummy(self) -> None:
        frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        _apex_training_dummy(frame, int(self.cx), int(self.cy + 80))
        result = self._detect(frame)
        self.assertIsNotNone(result.target)
        assert result.target is not None
        self.assertGreaterEqual(result.target.part_count, 2)
        self.assertGreater(result.target.humanoid_score, detection._MIN_FIGURE_SCORE)

    def test_detects_two_dummies_picks_closer_to_center(self) -> None:
        frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        _apex_training_dummy(frame, int(self.cx - 10), int(self.cy + 70))
        _apex_training_dummy(frame, int(self.cx + 140), int(self.cy + 70))
        result = self._detect(frame)
        self.assertIsNotNone(result.target)
        assert result.target is not None
        self.assertLess(abs(result.target.centroid_x - self.cx), 80.0)

    def test_rejects_horizontal_red_wall_stripe(self) -> None:
        frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        _red_horizontal_stripe(frame, int(self.cy))
        result = self._detect(frame)
        self.assertIsNone(result.target)

    def test_rejects_solid_red_wall_block(self) -> None:
        frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        _solid_red_block(frame, int(self.cx), int(self.cy), w=420, h=50)
        result = self._detect(frame)
        self.assertIsNone(result.target)

    def test_rejects_single_solid_red_rectangle(self) -> None:
        frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        _solid_red_block(frame, int(self.cx), int(self.cy + 40), w=50, h=110)
        result = self._detect(frame)
        self.assertIsNone(result.target)

    def test_rejects_health_bar_shape(self) -> None:
        frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        cv2.rectangle(frame, (int(self.cx - 50), int(self.cy - 60)), (int(self.cx + 50), int(self.cy - 52)), (0, 0, 255), -1)
        result = self._detect(frame)
        self.assertIsNone(result.target)

    def test_rejects_viewmodel_in_bottom_band(self) -> None:
        frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        cv2.circle(frame, (int(self.cx), self.h - 40), 10, (0, 0, 255), -1)
        result = self._detect(frame)
        self.assertIsNone(result.target)

    def test_segmentation_score_rejects_solid_column(self) -> None:
        mask = np.zeros((120, 60), dtype=np.uint8)
        mask[10:110, 20:40] = 255
        score = detection._vertical_red_segmentation_score(mask, 0, 0, 60, 120)
        self.assertLess(score, 0.5)

    def test_segmentation_score_accepts_striped_column(self) -> None:
        mask = np.zeros((120, 60), dtype=np.uint8)
        mask[10:30, 20:40] = 255
        mask[50:75, 20:40] = 255
        mask[90:105, 20:40] = 255
        score = detection._vertical_red_segmentation_score(mask, 0, 0, 60, 120)
        self.assertGreater(score, 0.5)


class MotionTests(unittest.TestCase):
    def test_tracker_smooths_large_jumps(self) -> None:
        from motion import TargetTracker

        tr = TargetTracker()
        m0 = tr.observe(100.0, 100.0, 0.0)
        m1 = tr.observe(400.0, 350.0, 0.05)
        jump = np.hypot(m1.x - m0.x, m1.y - m0.y)
        raw_jump = np.hypot(300.0, 250.0)
        self.assertLess(jump, raw_jump * 0.6)


if __name__ == "__main__":
    unittest.main()
