"""Scenario tests: Apex body structure vs walls/signs/background."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

import detector

# Apex enemy highlight red in BGR
RED = (0, 0, 255)
HSV_RED = [
    {"lower": [0, 100, 100], "upper": [12, 255, 255]},
    {"lower": [170, 100, 100], "upper": [180, 255, 255]},
]


def _apex_dummy(
    frame: np.ndarray,
    cx: int,
    foot_y: int,
    *,
    scale: float = 1.0,
    crouch: bool = False,
    partial: bool = False,
    side_shift: int = 0,
) -> None:
    """Segmented plates like firing-range dummies (head / chest / knees)."""
    s = scale
    if crouch:
        head_h, chest_h, knee_h = int(20 * s), int(36 * s), int(16 * s)
        gaps = (6, 8)
    else:
        head_h, chest_h, knee_h = int(26 * s), int(44 * s), int(22 * s)
        gaps = (10, 12)
    head_w, chest_w, knee_w = int(28 * s), int(42 * s), int(24 * s)
    cx += side_shift
    hx = cx - head_w // 2
    hy = foot_y - int((head_h + gaps[0] + chest_h + gaps[1] + knee_h))
    cv2.rectangle(frame, (hx, hy), (hx + head_w, hy + head_h), RED, -1)
    cx0 = cx - chest_w // 2
    cy0 = hy + head_h + gaps[0]
    cv2.rectangle(frame, (cx0, cy0), (cx0 + chest_w, cy0 + chest_h), RED, -1)
    kx = cx - knee_w // 2
    if partial:
        return
    ky0 = cy0 + chest_h + gaps[1]
    cv2.rectangle(frame, (kx, ky0), (kx + knee_w, ky0 + knee_h), RED, -1)


def _red_building(frame: np.ndarray, x: int, y: int, w: int, h: int) -> None:
    cv2.rectangle(frame, (x, y), (x + w, y + h), RED, -1)


def _horizontal_stripe(frame: np.ndarray, y: int, h: int = 28) -> None:
    frame[y : y + h, :] = RED


def _diamond_sign(frame: np.ndarray, cx: int, cy: int, size: int = 36) -> None:
    pts = np.array(
        [
            [cx, cy - size],
            [cx + size, cy],
            [cx, cy + size],
            [cx - size, cy],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(frame, [pts], RED)


class ScenarioTests(unittest.TestCase):
    def setUp(self) -> None:
        self.h, self.w = 720, 1280
        self.fov = 200
        self.cx = self.w / 2
        self.cy = self.h / 2

    def _detect(self, frame: np.ndarray, *, debug: bool = False):
        return detector.find_best_target(
            frame,
            HSV_RED,
            self.fov,
            min_area=60.0,
            fov_center_x=self.cx,
            fov_center_y=self.cy,
            debug=debug,
        )

    def test_close_dummy_wins_over_red_building(self) -> None:
        frame = np.full((self.h, self.w, 3), 40, dtype=np.uint8)
        _red_building(frame, 40, int(self.cy - 120), 280, 240)
        _apex_dummy(frame, int(self.cx), int(self.cy + 100), scale=1.2)
        r = self._detect(frame, debug=True)
        self.assertTrue(r.active, r.debug_lines)
        assert r.target is not None
        self.assertGreater(r.target.body_shape_score, 0.42)
        self.assertLess(abs(r.target.centroid_x - self.cx), 90)

    def test_two_enemies_picks_closer_valid_body(self) -> None:
        frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        _apex_dummy(frame, int(self.cx - 20), int(self.cy + 90), scale=1.0)
        _apex_dummy(frame, int(self.cx + 200), int(self.cy + 90), scale=0.9)
        r = self._detect(frame)
        self.assertTrue(r.active)
        assert r.target is not None
        self.assertLess(r.target.distance_to_center, 120)

    def test_partial_body_head_and_chest_only(self) -> None:
        frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        _apex_dummy(frame, int(self.cx), int(self.cy + 80), partial=True)
        r = self._detect(frame)
        self.assertTrue(r.active, msg="partial head+chest should still detect")

    def test_crouched_dummy_detects(self) -> None:
        frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        _apex_dummy(frame, int(self.cx), int(self.cy + 60), crouch=True)
        r = self._detect(frame)
        self.assertTrue(r.active)

    def test_side_angle_dummy_detects(self) -> None:
        frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        _apex_dummy(frame, int(self.cx), int(self.cy + 85), side_shift=25)
        r = self._detect(frame)
        self.assertTrue(r.active)

    def test_far_small_dummy_detects(self) -> None:
        frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        _apex_dummy(frame, int(self.cx), int(self.cy + 40), scale=0.55)
        r = self._detect(frame)
        self.assertTrue(r.active)

    def test_wall_stripe_no_active(self) -> None:
        frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        _horizontal_stripe(frame, int(self.cy))
        r = self._detect(frame, debug=True)
        self.assertFalse(r.active)
        self.assertIsNone(r.target)

    def test_diamond_sign_no_active(self) -> None:
        frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        _diamond_sign(frame, int(self.cx), int(self.cy))
        r = self._detect(frame)
        self.assertFalse(r.active)

    def test_red_wall_only_no_active(self) -> None:
        frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        _red_building(frame, int(self.cx - 200), int(self.cy - 80), 500, 160)
        r = self._detect(frame)
        self.assertFalse(r.active)

    def test_health_bar_only_no_active(self) -> None:
        frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        cv2.rectangle(frame, (int(self.cx - 55), int(self.cy - 40)), (int(self.cx + 55), int(self.cy - 32)), RED, -1)
        r = self._detect(frame)
        self.assertFalse(r.active)

    def test_solid_block_no_active(self) -> None:
        frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        cv2.rectangle(
            frame,
            (int(self.cx - 80), int(self.cy - 30)),
            (int(self.cx + 80), int(self.cy + 30)),
            RED,
            -1,
        )
        r = self._detect(frame)
        self.assertFalse(r.active)

    def test_lock_stable_across_frames(self) -> None:
        frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        _apex_dummy(frame, int(self.cx + 5), int(self.cy + 88))
        r0 = self._detect(frame)
        self.assertTrue(r0.active)
        assert r0.target is not None
        positions = []
        for dx in (0, 2, 4, 2, 0, -2):
            f = frame.copy()
            _apex_dummy(f, int(self.cx + 5 + dx), int(self.cy + 88))
            r = detector.find_best_target(
                f,
                HSV_RED,
                self.fov,
                60.0,
                self.cx,
                self.cy,
                sticky_target=r0.target,
                stickiness_pixels=80.0,
            )
            self.assertTrue(r.active)
            assert r.target is not None
            positions.append((r.target.centroid_x, r.target.centroid_y))
        jitter = np.std([p[0] for p in positions])
        self.assertLess(jitter, 35.0)

    def test_debug_lines_populated(self) -> None:
        frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        _apex_dummy(frame, int(self.cx), int(self.cy + 80))
        r = self._detect(frame, debug=True)
        self.assertTrue(any("body=" in ln for ln in r.debug_lines))
        self.assertTrue(any("SELECTED" in ln for ln in r.debug_lines) or not r.active)

    def test_body_score_prioritized_over_distance(self) -> None:
        frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        _apex_dummy(frame, int(self.cx), int(self.cy + 90), scale=1.1)
        cv2.circle(frame, (int(self.cx + 30), int(self.cy - 10)), 9, RED, -1)
        r = self._detect(frame)
        self.assertTrue(r.active)
        assert r.target is not None
        self.assertGreaterEqual(r.target.part_count, 2)


class LegacyTests(unittest.TestCase):
    def test_segmentation_striped_vs_solid(self) -> None:
        solid = np.zeros((100, 40), dtype=np.uint8)
        solid[10:90, 12:28] = 255
        striped = np.zeros((100, 40), dtype=np.uint8)
        striped[10:25, 12:28] = 255
        striped[40:60, 12:28] = 255
        striped[75:88, 12:28] = 255
        vs, _ = detector._vertical_profile_score(solid, 0, 0, 40, 100)
        vt, _ = detector._vertical_profile_score(striped, 0, 0, 40, 100)
        self.assertGreater(vt, vs)


if __name__ == "__main__":
    unittest.main()

