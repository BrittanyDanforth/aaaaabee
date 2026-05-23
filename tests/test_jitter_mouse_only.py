"""Recoil-break jitter must move mouse via pull only — never the overlay dot."""

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
        bx, by = rc.compute_bias(is_firing=True, dt=1.0 / 60.0)
        self.assertEqual(by, 0.0)
        self.assertLessEqual(abs(bx), 2.0)

    def test_jitter_enable_produces_mouse_dx_over_time(self) -> None:
        ctrl = PullController(
            _tuning(
                jitter_enabled=True,
                jitter_amplitude_pixels=2.0,
                jitter_frequency_hz=10.0,
                recoil_compensation_enabled=False,
                recoil_pull_down_pixels_per_second=0.0,
            )
        )
        seen_nonzero_x = False
        t = 0.0
        for _ in range(50):
            pr = ctrl.compute_delta(
                _on_target_tgt(300.0, 300.0),
                200.0,
                200.0,
                time_sec=t,
                is_firing=True,
            )
            if pr.dx != 0:
                seen_nonzero_x = True
            t += 1.0 / 60.0
        self.assertTrue(seen_nonzero_x, "mouse shake must emit horizontal pull deltas")


if __name__ == "__main__":
    unittest.main()
