"""Body-anchored aim: dot stays inside bbox, no sky drift, motion does not re-blend up."""

from __future__ import annotations

import time
import unittest

import detector
from motion import TargetTracker
from tests.reference_body_frames import (
    CX,
    CY,
    FRAME_H,
    FRAME_W,
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
TORSO_FRAC = 0.36


def _inside_upper_chest(t: detector.Target) -> bool:
    y_lo = t.bbox_y + t.bbox_h * 0.26
    y_hi = t.bbox_y + t.bbox_h * 0.54
    x_lo = t.bbox_x + t.bbox_w * 0.12
    x_hi = t.bbox_x + t.bbox_w * 0.88
    return x_lo <= t.centroid_x <= x_hi and y_lo <= t.centroid_y <= y_hi


class BodyAnchorTests(unittest.TestCase):
    def _det(self, frame, **kw):
        return detector.find_best_target(
            frame,
            HSV_RED,
            FOV,
            MIN_AREA,
            float(CX),
            float(CY),
            torso_aim_fraction=TORSO_FRAC,
            **kw,
        )

    def test_reference_frames_aim_inside_bbox(self) -> None:
        for name, gen in [
            ("standing", gen_firing_range_standing),
            ("close", gen_close_vertical_dummy),
            ("dynamic", gen_dynamic_pose_dummy),
            ("balloon_compete", gen_body_plus_balloon_compete),
        ]:
            r = self._det(gen())
            self.assertTrue(r.active, f"{name}: {r.debug_lines}")
            t = r.target
            assert t is not None
            self.assertTrue(
                _inside_upper_chest(t),
                f"{name} aim=({t.centroid_x:.0f},{t.centroid_y:.0f}) bbox=({t.bbox_x},{t.bbox_y},{t.bbox_w}x{t.bbox_h})",
            )

    def test_motion_does_not_pull_aim_above_detector(self) -> None:
        frame = gen_firing_range_standing()
        r = self._det(frame)
        self.assertTrue(r.active)
        t = r.target
        assert t is not None
        raw_y = t.centroid_y
        tracker = TargetTracker()
        tracker.configure_prediction(True, 0.04, 22.0)
        m = tracker.observe_target(
            t.centroid_x,
            t.centroid_y,
            0.0,
            bbox_x=t.bbox_x,
            bbox_y=t.bbox_y,
            bbox_w=t.bbox_w,
            bbox_h=t.bbox_h,
        )
        for i in range(1, 25):
            m = tracker.observe_target(
                t.centroid_x + i * 2.0,
                t.centroid_y,
                i * (1.0 / 60.0),
                bbox_x=t.bbox_x,
                bbox_y=t.bbox_y,
                bbox_w=t.bbox_w,
                bbox_h=t.bbox_h,
            )
        y_hi = t.bbox_y + t.bbox_h * 0.52
        self.assertLessEqual(m.y, y_hi + 2.0, f"motion y={m.y:.1f} above chest band {y_hi:.1f}")
        self.assertGreaterEqual(m.y, t.bbox_y + t.bbox_h * 0.28 - 2.0)
        self.assertLess(m.y, raw_y + 35.0, "should not leap far above raw anchor")

    def test_observe_target_no_column_reblend(self) -> None:
        """First frame: motion equals detector anchor (no 65% bbox column shift)."""
        frame = gen_close_vertical_dummy()
        r = self._det(frame)
        self.assertTrue(r.active)
        t = r.target
        assert t is not None
        tracker = TargetTracker()
        tracker.configure_prediction(False, 0.0, 0.0)
        m = tracker.observe_target(
            t.centroid_x,
            t.centroid_y,
            1.0,
            bbox_x=t.bbox_x,
            bbox_y=t.bbox_y,
            bbox_w=t.bbox_w,
            bbox_h=t.bbox_h,
        )
        self.assertAlmostEqual(m.x, t.centroid_x, delta=0.5)
        self.assertAlmostEqual(m.y, t.centroid_y, delta=0.5)


if __name__ == "__main__":
    unittest.main()
