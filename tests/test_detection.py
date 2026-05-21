"""Tests for HSV targeting: reject sky/viewmodel false positives, prefer humanoid."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

import detection


def _red_humanoid(frame: np.ndarray, cx: int, cy: int, w: int = 36, h: int = 90) -> None:
    x1, y1 = cx - w // 2, cy - h
    cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 220), -1)


def _red_dot(frame: np.ndarray, cx: int, cy: int, r: int = 6) -> None:
    cv2.circle(frame, (cx, cy), r, (0, 0, 255), -1)


HSV_RED = [{"lower": [0, 120, 120], "upper": [12, 255, 255]}]


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

    def test_prefers_center_humanoid_over_side_blob(self) -> None:
        frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        _red_humanoid(frame, int(self.cx), int(self.cy + 40))
        _red_dot(frame, int(self.cx - 120), int(self.cy - 30))
        result = self._detect(frame)
        self.assertIsNotNone(result.target)
        assert result.target is not None
        self.assertLess(abs(result.target.centroid_x - self.cx), 40.0)

    def test_rejects_viewmodel_red_in_bottom_band(self) -> None:
        frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        _red_dot(frame, int(self.cx), self.h - 40, r=10)
        result = self._detect(frame)
        self.assertIsNone(result.target)

    def test_rejects_sky_blob_above_crosshair(self) -> None:
        frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        _red_dot(frame, int(self.cx), int(self.cy - self.fov * 0.5), r=8)
        result = self._detect(frame)
        self.assertIsNone(result.target)

    def test_finds_humanoid_in_fov(self) -> None:
        frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        _red_humanoid(frame, int(self.cx + 30), int(self.cy + 50))
        result = self._detect(frame)
        self.assertIsNotNone(result.target)
        assert result.target is not None
        self.assertGreater(result.target.confidence, detection._MIN_CONFIDENCE)

    def test_vertical_penalty_scores_center_lower_than_sky(self) -> None:
        low = detection.Target(self.cx, self.cy + 30, 2000.0, 30.0, 0.5, 30, 80, 0.6)
        high = detection.Target(self.cx, self.cy - 80, 2000.0, 80.0, 0.5, 30, 80, 0.6)
        score_low = detection.score_target(
            low, float(self.fov), 2.0, 0.02, center_y=self.cy
        )
        score_high = detection.score_target(
            high, float(self.fov), 2.0, 0.02, center_y=self.cy
        )
        self.assertGreater(score_low, score_high)


class MotionTests(unittest.TestCase):
    def test_tracker_smooths_large_jumps(self) -> None:
        tracker = detection  # noqa: ensure import works
        from motion import TargetTracker

        tr = TargetTracker()
        m0 = tr.observe(100.0, 100.0, 0.0)
        m1 = tr.observe(400.0, 350.0, 0.05)
        jump = np.hypot(m1.x - m0.x, m1.y - m0.y)
        raw_jump = np.hypot(300.0, 250.0)
        self.assertLess(jump, raw_jump * 0.6)


if __name__ == "__main__":
    unittest.main()
