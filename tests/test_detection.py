"""Tests for HSV targeting: humanoid only, reject walls/stripes/sky."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

import detection


def _red_humanoid(frame: np.ndarray, cx: int, cy: int, w: int = 34, h: int = 92) -> None:
    x1, y1 = cx - w // 2, cy - h
    cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 220), -1)


def _red_horizontal_stripe(frame: np.ndarray, y: int, h: int = 22) -> None:
    frame[y : y + h, :] = (0, 0, 230)


def _red_wall_panel(frame: np.ndarray, cx: int, cy: int, w: int, h: int) -> None:
    x1, y1 = cx - w // 2, cy - h // 2
    cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 240), -1)


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

    def test_finds_standing_humanoid(self) -> None:
        frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        _red_humanoid(frame, int(self.cx), int(self.cy + 50))
        result = self._detect(frame)
        self.assertIsNotNone(result.target)

    def test_rejects_horizontal_red_wall_stripe(self) -> None:
        """Red/black striped wall band — wide, short, not a body."""
        frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        _red_horizontal_stripe(frame, int(self.cy))
        result = self._detect(frame)
        self.assertIsNone(result.target)

    def test_rejects_screen_wide_wall_panel(self) -> None:
        frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        _red_wall_panel(frame, int(self.cx), int(self.cy), w=int(self.w * 0.7), h=40)
        result = self._detect(frame)
        self.assertIsNone(result.target)

    def test_rejects_viewmodel_red_in_bottom_band(self) -> None:
        frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        cv2.circle(frame, (int(self.cx), self.h - 40), 10, (0, 0, 255), -1)
        result = self._detect(frame)
        self.assertIsNone(result.target)

    def test_rejects_sky_blob_above_crosshair(self) -> None:
        frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        cv2.circle(frame, (int(self.cx), int(self.cy - self.fov * 0.5)), 8, (0, 0, 255), -1)
        result = self._detect(frame)
        self.assertIsNone(result.target)

    def test_prefers_humanoid_over_small_side_blob(self) -> None:
        frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        _red_humanoid(frame, int(self.cx), int(self.cy + 45))
        cv2.circle(frame, (int(self.cx - 100), int(self.cy)), 8, (0, 0, 255), -1)
        result = self._detect(frame)
        self.assertIsNotNone(result.target)
        assert result.target is not None
        self.assertLess(abs(result.target.centroid_x - self.cx), 35.0)

    def test_humanoid_gate_rejects_flat_rectangle(self) -> None:
        frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        _red_wall_panel(frame, int(self.cx), int(self.cy), w=200, h=30)
        mask = detection.build_hsv_mask(frame, HSV_RED)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.assertGreater(len(contours), 0)
        ok, *_ = detection._is_humanoid_contour(contours[0], self.w, self.h)
        self.assertFalse(ok)


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
