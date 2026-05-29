"""YOLO overlay must use follow drag, not raw centroid snaps."""

from __future__ import annotations

import math
import unittest

from detector import Target
from motion import TargetMotion, TargetTracker


def _yolo_overlay_motion(
    tracker: TargetTracker,
    target: Target,
    *,
    last_motion: TargetMotion | None,
    capture_fps: int = 60,
) -> TargetMotion:
    """Mirror AssistRuntime._motion_from_yolo_target overlay path."""
    dt = max(1.0 / max(1, capture_fps), 1e-4)
    tracker._body_bbox = (
        int(target.bbox_x),
        int(target.bbox_y),
        int(target.bbox_w),
        int(target.bbox_h),
    )
    if last_motion is not None:
        tracker._vx = (target.centroid_x - last_motion.x) / dt
        tracker._vy = (target.centroid_y - last_motion.y) / dt
    ox, oy = tracker._advance_overlay_follow(
        target.centroid_x,
        target.centroid_y,
        dt,
    )
    return TargetMotion(
        target.centroid_x,
        target.centroid_y,
        0.0,
        0.0,
        overlay_x=ox,
        overlay_y=oy,
    )


class YoloOverlayMotionTests(unittest.TestCase):
    def test_overlay_follow_dampens_detect_jump(self) -> None:
        tr = TargetTracker()
        tr.configure_overlay_dot_alpha(0.58)
        t0 = Target(
            200.0,
            180.0,
            5000.0,
            10.0,
            0.9,
            bbox_x=170,
            bbox_y=100,
            bbox_w=60,
            bbox_h=120,
        )
        m0 = _yolo_overlay_motion(tr, t0, last_motion=None)
        ox0, oy0 = m0.overlay_xy()

        t1 = Target(
            260.0,
            180.0,
            5000.0,
            10.0,
            0.9,
            bbox_x=230,
            bbox_y=100,
            bbox_w=60,
            bbox_h=80,
        )
        m1 = _yolo_overlay_motion(tr, t1, last_motion=m0)
        ox1, oy1 = m1.overlay_xy()
        step = math.hypot(ox1 - ox0, oy1 - oy0)
        raw_jump = 60.0
        self.assertLess(step, raw_jump, "overlay must not snap full detect jump")
        self.assertGreater(step, 0.5)
        self.assertLess(abs(ox1 - 260.0), raw_jump)


if __name__ == "__main__":
    unittest.main()
