"""Lateral recoil hold — mouse pull only, never the overlay dot."""

from __future__ import annotations

import unittest

from motion import RecoilCompensator
from pull import PullController
from tests.test_recoil_jitter import _on_target_tgt, _tuning


class JitterMouseOnlyTests(unittest.TestCase):
    def test_jitter_only_without_recoil_pull_down(self) -> None:
        rc = RecoilCompensator(
            recoil_enabled=False,
            pull_down_px_per_s=0.0,
            jitter_enabled=True,
            jitter_amplitude_px=2.0,
            jitter_frequency_hz=8.0,
        )
        self.assertTrue(rc.active)
        bx, by = rc.compute_bias(is_firing=True, dt=1.0 / 60.0, err_x=15.0)
        self.assertEqual(by, 0.0)
        self.assertGreater(bx, 0.0)
        self.assertLessEqual(abs(bx), 2.0)
        bx0, _ = rc.compute_bias(is_firing=True, dt=1.0 / 60.0, err_x=0.0)
        self.assertEqual(bx0, 0.0)

    def test_lateral_hold_only_when_aim_off_center(self) -> None:
        ctrl = PullController(
            _tuning(
                jitter_enabled=True,
                jitter_amplitude_pixels=2.0,
                jitter_frequency_hz=10.0,
                recoil_compensation_enabled=False,
                recoil_pull_down_pixels_per_second=0.0,
            )
        )
        on_target = []
        off_target = []
        t = 0.0
        for i in range(40):
            tgt = _on_target_tgt(200.0, 200.0) if i < 20 else _on_target_tgt(230.0, 200.0)
            pr = ctrl.compute_delta(
                tgt,
                200.0,
                200.0,
                time_sec=t,
                is_firing=True,
            )
            if i < 20:
                on_target.append(pr.dx)
            else:
                off_target.append(pr.dx)
            t += 1.0 / 60.0
        self.assertEqual(sum(on_target), 0, "centered aim → no sideways correction")
        self.assertGreater(sum(off_target), 0, "off-center → lateral hold assists pull")


if __name__ == "__main__":
    unittest.main()
