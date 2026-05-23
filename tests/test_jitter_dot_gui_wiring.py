"""GUI + pipeline wiring for dot smoothness and engagement jitter."""

from __future__ import annotations

import unittest
from pathlib import Path

from motion import TargetTracker
from pull import PullController, PullTuning
from tests.test_recoil_jitter import _on_target_tgt, _tuning


class DotSmoothnessWiringTests(unittest.TestCase):
    def test_runtime_reads_overlay_dot_smooth_alpha(self) -> None:
        text = Path("runtime.py").read_text(encoding="utf-8")
        self.assertIn("overlay_dot_smooth_alpha", text)
        self.assertIn("set_dot_render_alpha", text)
        self.assertIn("smooth_overlay_point", text)

    def test_gui_has_red_dot_smoothness_slider(self) -> None:
        text = Path("aba_gui.py").read_text(encoding="utf-8")
        self.assertIn('"Red dot smoothness"', text)
        self.assertIn('"overlay_dot_smooth_alpha"', text)

    def test_overlay_window_uses_float_coords_not_int_round(self) -> None:
        text = Path("overlay_window.py").read_text(encoding="utf-8")
        self.assertNotIn("int(round(target[0]))", text)
        self.assertIn("_render_x", text)

    def test_smoothness_tau_affects_deadband_overlay_drift(self) -> None:
        """Higher smoothing_tau_still should damp overlay more in deadband."""
        dt = 1.0 / 60.0

        def spread(tau_still: float) -> float:
            tr = TargetTracker()
            tr.configure_smoothing_tau(tau_still, 0.015)
            tr.observe(500.0, 400.0, 0.0)
            xs: list[float] = []
            for i in range(1, 30):
                jx = 500.0 + (1.0 if i % 2 == 0 else -1.0)
                jy = 400.0 + (1.0 if i % 3 == 0 else -1.0)
                m = tr.observe(jx, jy, i * dt)
                ox, _ = m.overlay_xy()
                xs.append(ox)
            return max(xs) - min(xs)

        self.assertLess(
            spread(0.12),
            spread(0.03),
            "higher Smoothness tau should reduce overlay swim",
        )

    def test_jitter_amplitude_zero_is_inactive(self) -> None:
        from motion import RecoilCompensator

        rc = RecoilCompensator(
            recoil_enabled=False,
            pull_down_px_per_s=0.0,
            jitter_enabled=True,
            jitter_amplitude_px=0.0,
            jitter_frequency_hz=8.0,
        )
        self.assertFalse(rc.active)

    def test_jitter_hot_reload_changes_horizontal_delta(self) -> None:
        ctrl = PullController(_tuning())
        ctrl.update_tuning(
            jitter_enabled=True,
            jitter_amplitude_pixels=2.5,
            jitter_frequency_hz=10.0,
        )
        deltas: list[int] = []
        t = 0.0
        for _ in range(40):
            pr = ctrl.compute_delta(
                _on_target_tgt(200.0, 200.0),
                200.0,
                200.0,
                time_sec=t,
                is_firing=True,
            )
            deltas.append(pr.dx)
            t += 1.0 / 60.0
        self.assertGreater(max(abs(d) for d in deltas), 0)


if __name__ == "__main__":
    unittest.main()
