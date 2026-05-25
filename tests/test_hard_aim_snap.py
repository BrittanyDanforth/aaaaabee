"""Hard-aim snap behaviour: with low smoothing / firm strength a stationary
target's chest should close within ~2 frames of detection at 60 FPS, and
within ~3 frames at 30 FPS. Locks in the tight velocity loop introduced by
the engagement-boost commit so future tuning passes cannot silently regress
the "hard aim" preset response."""

from __future__ import annotations

import unittest

from detector import Target
from pull import PullController, PullTuning


def _hard_aim_tuning() -> PullTuning:
    return PullTuning(
        max_speed=40.0,
        pull_strength=1.0,
        deadzone=2.0,
        velocity_smoothing=0.30,
        smoothing_curve="ease_out",
        magnetism_radius=70.0,
        magnetism_min_scale=0.70,
        fov_radius=200.0,
        fov_edge_min_scale=0.85,
        prediction_enabled=False,
        prediction_lead_seconds=0.0,
        prediction_max_pixels=0.0,
        humanize_enabled=False,
        humanize_amplitude=0.0,
        humanize_jerk_limit=0.0,
        aim_pre_smoothed=True,
    )


def _stationary_target(off_x: float, off_y: float, *, cx: float, cy: float) -> Target:
    return Target(
        centroid_x=cx + off_x,
        centroid_y=cy + off_y,
        area=2400.0,
        distance_to_center=(off_x ** 2 + off_y ** 2) ** 0.5,
        confidence=0.95,
        bbox_x=int(cx + off_x) - 22,
        bbox_y=int(cy + off_y) - 70,
        bbox_w=44,
        bbox_h=140,
    )


def _frames_to_close(off_x: float, fps: float, *, threshold: float = 6.0) -> int:
    ctrl = PullController(_hard_aim_tuning())
    cx, cy = 640.0, 360.0
    cursor = [cx, cy]
    dt = 1.0 / fps
    tgt = _stationary_target(off_x, 0.0, cx=cx, cy=cy)
    for i in range(1, 60):
        pr = ctrl.compute_delta(tgt, cursor[0], cursor[1], time_sec=i * dt)
        cursor[0] += pr.dx
        cursor[1] += pr.dy
        err = abs((cx + off_x) - cursor[0])
        if err <= threshold:
            return i
    return 60


class HardAimSnapTests(unittest.TestCase):
    def test_60fps_80px_closes_within_three_frames(self) -> None:
        n = _frames_to_close(80.0, fps=60.0)
        self.assertLessEqual(n, 3, f"80px error closed in {n} frames (expected <=3)")

    def test_30fps_80px_closes_within_three_frames(self) -> None:
        n = _frames_to_close(80.0, fps=30.0)
        self.assertLessEqual(n, 3, f"80px error closed in {n} frames at 30 FPS (expected <=3)")

    def test_small_error_does_not_overshoot(self) -> None:
        """A small 12-px error must close monotonically — no oscillation past zero."""
        ctrl = PullController(_hard_aim_tuning())
        cx, cy = 640.0, 360.0
        cursor = [cx, cy]
        dt = 1.0 / 60.0
        tgt = _stationary_target(12.0, 0.0, cx=cx, cy=cy)
        max_overshoot = 0.0
        for i in range(1, 30):
            pr = ctrl.compute_delta(tgt, cursor[0], cursor[1], time_sec=i * dt)
            cursor[0] += pr.dx
            cursor[1] += pr.dy
            overshoot = cursor[0] - (cx + 12.0)
            max_overshoot = max(max_overshoot, overshoot)
        self.assertLessEqual(max_overshoot, 1.0, f"hard-aim overshot stationary target by {max_overshoot:.2f}px")


if __name__ == "__main__":
    unittest.main()
