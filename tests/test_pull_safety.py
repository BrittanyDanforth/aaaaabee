"""Pull controller safety: NaN/Inf inputs must not poison state or crash."""

from __future__ import annotations

import math
import unittest

from detector import Target
from pull import PullController, PullTuning


def _tuning(**kwargs) -> PullTuning:
    base = dict(
        max_speed=15.0,
        pull_strength=0.56,
        deadzone=2.0,
        velocity_smoothing=0.5,
        smoothing_curve="ease_out",
        magnetism_radius=70.0,
        magnetism_min_scale=0.35,
        fov_radius=140.0,
        fov_edge_min_scale=0.5,
        prediction_enabled=False,
        prediction_lead_seconds=0.04,
        prediction_max_pixels=24.0,
        humanize_enabled=False,
        humanize_amplitude=0.0,
        humanize_jerk_limit=0.0,
        aim_pre_smoothed=True,
    )
    base.update(kwargs)
    return PullTuning(**base)


def _tgt(x: float, y: float, *, w: int = 40, h: int = 100) -> Target:
    bx = int(x) - w // 2 if math.isfinite(x) else 0
    by = int(y) - h // 2 if math.isfinite(y) else 0
    return Target(x, y, 400.0, 50.0, 0.9, bbox_x=bx, bbox_y=by, bbox_w=w, bbox_h=h)


class PullNaNSafetyTests(unittest.TestCase):
    def test_nan_target_returns_zero(self) -> None:
        ctrl = PullController(_tuning())
        nan = float("nan")
        pr = ctrl.compute_delta(_tgt(nan, 200.0), 200.0, 200.0, time_sec=1.0)
        self.assertEqual((pr.dx, pr.dy), (0, 0))
        self.assertEqual(pr.magnitude, 0.0)

    def test_nan_center_returns_zero_and_preserves_state(self) -> None:
        ctrl = PullController(_tuning())
        nan = float("nan")
        ctrl.compute_delta(_tgt(260.0, 200.0), 200.0, 200.0, time_sec=0.0)
        vel_before = (ctrl._vel_x, ctrl._vel_y)
        pr = ctrl.compute_delta(_tgt(260.0, 200.0), nan, 200.0, time_sec=0.016)
        self.assertEqual((pr.dx, pr.dy), (0, 0))
        # Velocity state should not have been corrupted to NaN.
        self.assertTrue(math.isfinite(ctrl._vel_x))
        self.assertTrue(math.isfinite(ctrl._vel_y))
        self.assertEqual((ctrl._vel_x, ctrl._vel_y), vel_before)

    def test_inf_center_returns_zero(self) -> None:
        ctrl = PullController(_tuning())
        pr = ctrl.compute_delta(_tgt(260.0, 200.0), float("inf"), 200.0, time_sec=0.0)
        self.assertEqual((pr.dx, pr.dy), (0, 0))

    def test_nan_time_does_not_poison_last_time(self) -> None:
        ctrl = PullController(_tuning())
        ctrl.compute_delta(_tgt(260.0, 200.0), 200.0, 200.0, time_sec=float("nan"))
        # Subsequent finite time call must produce a usable dt.
        self.assertTrue(ctrl._last_time is None or math.isfinite(ctrl._last_time))
        pr = ctrl.compute_delta(_tgt(260.0, 200.0), 200.0, 200.0, time_sec=0.10)
        # Pull should engage (dist=60, well past deadzone=2).
        self.assertTrue(math.isfinite(ctrl._vel_x) and math.isfinite(ctrl._vel_y))

    def test_repeated_same_time_no_division_by_zero(self) -> None:
        ctrl = PullController(_tuning())
        now = 1.234
        for _ in range(5):
            pr = ctrl.compute_delta(_tgt(260.0, 200.0), 200.0, 200.0, time_sec=now)
            self.assertTrue(math.isfinite(pr.magnitude))
            self.assertTrue(math.isfinite(ctrl._vel_x))

    def test_negative_dt_does_not_break_state(self) -> None:
        ctrl = PullController(_tuning())
        ctrl.compute_delta(_tgt(260.0, 200.0), 200.0, 200.0, time_sec=10.0)
        # perf_counter going backward by 1ms.
        pr = ctrl.compute_delta(_tgt(260.0, 200.0), 200.0, 200.0, time_sec=9.999)
        self.assertTrue(math.isfinite(ctrl._vel_x))
        self.assertTrue(math.isfinite(pr.magnitude))


if __name__ == "__main__":
    unittest.main()
