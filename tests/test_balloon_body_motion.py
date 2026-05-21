"""Balloon rejection, body-vs-balloon competition, moving-body lag."""

from __future__ import annotations

import time
import unittest

import cv2
import numpy as np

import detector
from motion import TargetTracker
from tests.reference_body_frames import (
    CX,
    CY,
    FRAME_H,
    FRAME_W,
    gen_ads_dummy_with_sky_balloon,
    gen_body_plus_balloon_compete,
    gen_close_vertical_dummy,
    gen_dynamic_pose_dummy,
    gen_firing_range_standing,
)

HSV_RED = [
    {"lower": [0, 100, 100], "upper": [12, 255, 255]},
    {"lower": [170, 100, 100], "upper": [180, 255, 255]},
]
FOV = 200
MIN_AREA = 40.0


class BalloonRejectionTests(unittest.TestCase):
    def _det(self, frame: np.ndarray, *, sticky=None):
        return detector.find_best_target(
            frame,
            HSV_RED,
            FOV,
            MIN_AREA,
            float(CX),
            float(CY),
            sticky_target=sticky,
            stickiness_pixels=90,
            debug=True,
        )

    def test_sky_balloon_inside_fov_no_target(self) -> None:
        frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8) + 25
        cv2.circle(frame, (CX, int(FRAME_H * 0.12)), 34, (0, 0, 255), -1)
        r = self._det(frame)
        self.assertFalse(r.active, "\n".join(r.debug_lines))

    def test_round_red_center_no_target(self) -> None:
        frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8) + 30
        cv2.circle(frame, (CX, CY - 80), 40, (0, 0, 255), -1)
        r = self._det(frame)
        self.assertFalse(r.active)
        joined = "\n".join(r.debug_lines)
        if "cand[" in joined:
            self.assertTrue(
                any(
                    tok in joined
                    for tok in ("round_non_body", "sky_blob", "no_body_stack", "no_torso", "solid_wall")
                ),
                joined,
            )

    def test_body_plus_balloon_body_wins(self) -> None:
        frame = gen_body_plus_balloon_compete()
        r = self._det(frame)
        self.assertTrue(r.active, r.debug_lines)
        assert r.target is not None
        self.assertGreater(r.target.centroid_y, FRAME_H * 0.38)
        self.assertGreaterEqual(r.target.part_count, 2)
        self.assertGreater(r.target.body_shape_score, 0.45)

    def test_reference_frames_detect_body(self) -> None:
        for name, gen in [
            ("standing", gen_firing_range_standing),
            ("close", gen_close_vertical_dummy),
            ("dynamic", gen_dynamic_pose_dummy),
            ("ads", lambda: gen_ads_dummy_with_sky_balloon()[0]),
        ]:
            r = self._det(gen())
            self.assertTrue(r.active, f"{name} failed: {r.debug_lines}")

    def test_wall_stripe_rejected(self) -> None:
        frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        frame[120:150, :] = (0, 0, 255)
        r = self._det(frame)
        self.assertFalse(r.active)


class MovingBodyMotionTests(unittest.TestCase):
    def test_tracker_lag_bounded_on_lateral_motion(self) -> None:
        """Simulated 60 FPS lateral dummy — smoothed trail < 35% frame lag vs raw."""
        tracker = TargetTracker()
        raw_x: list[float] = []
        smooth_x: list[float] = []
        sticky = None
        t0 = time.perf_counter()
        for i in range(20):
            frame = np.full((FRAME_H, FRAME_W, 3), 90, dtype=np.uint8)
            ox = (i - 10) * 14
            foot = CY + 120
            for dx, dy, rw, rh in [(0, -90, 20, 18), (-6, -50, 34, 26), (0, -18, 18, 16)]:
                x, y = CX + dx + ox, foot + dy
                cv2.rectangle(frame, (x - rw // 2, y - rh), (x + rw // 2, y), (0, 0, 255), -1)
            r = detector.find_best_target(
                frame,
                HSV_RED,
                FOV,
                MIN_AREA,
                float(CX),
                float(CY),
                sticky_target=sticky,
                stickiness_pixels=90,
            )
            if r.active and r.target:
                sticky = r.target
                t = t0 + i * (1.0 / 60.0)
                m = tracker.observe_target(
                    r.target.centroid_x,
                    r.target.centroid_y,
                    t,
                    bbox_x=r.target.bbox_x,
                    bbox_y=r.target.bbox_y,
                    bbox_w=r.target.bbox_w,
                    bbox_h=r.target.bbox_h,
                )
                raw_x.append(r.target.centroid_x)
                smooth_x.append(m.x)
        self.assertGreaterEqual(len(smooth_x), 15)
        errs = [abs(smooth_x[i] - raw_x[i]) for i in range(len(smooth_x))]
        self.assertLess(max(errs), 55.0, f"max lag px {max(errs)}")
        # smoothed should follow trend, not crawl: total travel > 70% raw
        raw_travel = abs(raw_x[-1] - raw_x[0])
        smooth_travel = abs(smooth_x[-1] - smooth_x[0])
        self.assertGreater(smooth_travel, raw_travel * 0.72)


if __name__ == "__main__":
    unittest.main()
