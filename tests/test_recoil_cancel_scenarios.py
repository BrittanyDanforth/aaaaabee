"""Recoil cancel scenarios — straight on-target spray, ramp, lateral assist."""

from __future__ import annotations

import unittest

from motion import RecoilCompensator
from pull import PullController
from tests.test_recoil_jitter import _on_target_tgt, _tgt, _tuning


class RecoilCancelScenarioTests(unittest.TestCase):
    def test_on_target_spray_stays_straight_no_sideways(self) -> None:
        ctrl = PullController(
            _tuning(
                recoil_compensation_enabled=True,
                recoil_pull_down_pixels_per_second=60.0,
                jitter_enabled=True,
                jitter_amplitude_pixels=2.0,
                jitter_frequency_hz=8.0,
            )
        )
        deltas = []
        t = 0.0
        for _ in range(60):
            pr = ctrl.compute_delta(
                _on_target_tgt(200.0, 200.0),
                200.0,
                200.0,
                time_sec=t,
                is_firing=True,
            )
            deltas.append((pr.dx, pr.dy))
            t += 1.0 / 60.0
        self.assertEqual(sum(dx for dx, _ in deltas), 0, "on-target must not shake sideways")
        total_dy = sum(dy for _, dy in deltas)
        self.assertGreater(total_dy, 20, "pull-down should fight vertical recoil")

    def test_pull_down_ramps_in_during_spray(self) -> None:
        rc = RecoilCompensator(
            recoil_enabled=True,
            pull_down_px_per_s=120.0,
            jitter_enabled=False,
            jitter_amplitude_px=0.0,
            jitter_frequency_hz=8.0,
        )
        first_by = rc.compute_bias(is_firing=True, dt=1.0 / 60.0, err_x=0.0)[1]
        for _ in range(25):
            rc.compute_bias(is_firing=True, dt=1.0 / 60.0, err_x=0.0)
        late_by = rc.compute_bias(is_firing=True, dt=1.0 / 60.0, err_x=0.0)[1]
        self.assertGreater(late_by, first_by * 2.5, "pull-down should ramp up during burst")

    def test_lateral_hold_assists_when_aim_off_center(self) -> None:
        rc = RecoilCompensator(
            recoil_enabled=False,
            pull_down_px_per_s=0.0,
            jitter_enabled=True,
            jitter_amplitude_px=2.0,
            jitter_frequency_hz=10.0,
        )
        bx, _ = rc.compute_bias(is_firing=True, dt=1.0 / 60.0, err_x=18.0)
        self.assertGreater(bx, 0.0, "positive err_x → mouse moves toward target")
        self.assertLessEqual(bx, 2.0)
        bx0, _ = rc.compute_bias(is_firing=True, dt=1.0 / 60.0, err_x=0.0)
        self.assertEqual(bx0, 0.0)

    def test_offset_target_pull_and_lateral_same_sign(self) -> None:
        ctrl = PullController(
            _tuning(
                recoil_compensation_enabled=True,
                recoil_pull_down_pixels_per_second=40.0,
                jitter_enabled=True,
                jitter_amplitude_pixels=2.0,
                jitter_frequency_hz=8.0,
            )
        )
        total_dy = 0
        t = 0.0
        for _ in range(20):
            pr = ctrl.compute_delta(
                _tgt(240.0, 200.0),
                200.0,
                200.0,
                time_sec=t,
                is_firing=True,
            )
            if _ == 0:
                self.assertGreater(pr.dx, 0, "target right of crosshair → pull right")
            total_dy += pr.dy
            t += 1.0 / 60.0
        self.assertGreater(total_dy, 3, "recoil cancel still pulls down while tracking")


if __name__ == "__main__":
    unittest.main()
