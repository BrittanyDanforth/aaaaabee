"""
RecoilCompensator + jitter integration with PullController.

Verifies engagement gating, hot-reload of new tuning keys, residual safety,
and that the default (disabled) configuration is bit-identical to the
pre-feature pull output.
"""

from __future__ import annotations

import math
import unittest

from detector import Target
from motion import RecoilCompensator
from pull import PullController, PullTuning


def _tuning(**overrides) -> PullTuning:
    base = dict(
        max_speed=20.0,
        pull_strength=0.85,
        deadzone=2.0,
        velocity_smoothing=0.45,
        smoothing_curve="ease_out",
        magnetism_radius=80.0,
        magnetism_min_scale=0.35,
        fov_radius=160.0,
        fov_edge_min_scale=0.6,
        prediction_enabled=False,
        prediction_lead_seconds=0.04,
        prediction_max_pixels=24.0,
        humanize_enabled=False,
        humanize_amplitude=0.0,
        humanize_jerk_limit=0.0,
        aim_pre_smoothed=True,
    )
    base.update(overrides)
    return PullTuning(**base)


def _tgt(x: float, y: float, *, w: int = 40, h: int = 110) -> Target:
    return Target(
        x, y, 400.0, 50.0, 0.9,
        bbox_x=int(x) - w // 2,
        bbox_y=int(y) - h // 2,
        bbox_w=w,
        bbox_h=h,
    )


def _on_target_tgt(center_x: float, center_y: float) -> Target:
    """A target sitting exactly on the crosshair — pull would otherwise be 0."""
    return _tgt(center_x, center_y)


class RecoilCompensatorUnitTests(unittest.TestCase):
    """Pure unit tests on the compensator without involving PullController."""

    def test_disabled_returns_zero(self) -> None:
        rc = RecoilCompensator(
            recoil_enabled=False,
            pull_down_px_per_s=120.0,
            jitter_enabled=False,
            jitter_amplitude_px=2.0,
            jitter_frequency_hz=8.0,
        )
        self.assertFalse(rc.active)
        bx, by = rc.compute_bias(is_firing=True, dt=0.016)
        self.assertEqual((bx, by), (0.0, 0.0))

    def test_not_firing_returns_zero_even_when_armed(self) -> None:
        rc = RecoilCompensator(
            recoil_enabled=True,
            pull_down_px_per_s=80.0,
            jitter_enabled=True,
            jitter_amplitude_px=2.0,
            jitter_frequency_hz=6.0,
        )
        self.assertTrue(rc.active)
        for _ in range(10):
            self.assertEqual(rc.compute_bias(is_firing=False, dt=0.016), (0.0, 0.0))

    def test_firing_bias_within_expected_bounds(self) -> None:
        rc = RecoilCompensator(
            recoil_enabled=True,
            pull_down_px_per_s=60.0,
            jitter_enabled=True,
            jitter_amplitude_px=2.0,
            jitter_frequency_hz=8.0,
        )
        max_bx = 0.0
        max_by = 0.0
        # Sweep one full second so the sine wave samples its full envelope.
        for _ in range(60):
            bx, by = rc.compute_bias(is_firing=True, dt=1.0 / 60.0)
            max_bx = max(max_bx, abs(bx))
            max_by = max(max_by, by)
        # Horizontal jitter must stay below the amplitude (≤ 2.0 px).
        self.assertLess(max_bx, 2.0 + 1e-6)
        # Per-frame Y bias at 60 Hz ≈ 60 / 60 = 1.0 px — well under "per-frame
        # < expected" sanity bound. We assert it's strictly less than 60/60 + 0.5.
        self.assertLess(max_by, 1.5)
        # And strictly greater than zero (we did fire).
        self.assertGreater(max_by, 0.5)

    def test_phase_resets_on_release(self) -> None:
        rc = RecoilCompensator(
            recoil_enabled=False,
            pull_down_px_per_s=0.0,
            jitter_enabled=True,
            jitter_amplitude_px=2.0,
            jitter_frequency_hz=8.0,
        )
        # Advance the phase mid-fire.
        for _ in range(20):
            rc.compute_bias(is_firing=True, dt=1.0 / 60.0)
        # Release.
        rc.compute_bias(is_firing=False, dt=1.0 / 60.0)
        # The very next firing frame must start from phase ≈ 0 → bias ≈ 0.
        bx, _ = rc.compute_bias(is_firing=True, dt=1.0 / 60.0)
        self.assertLess(abs(bx), 0.4)


class PullControllerRecoilTests(unittest.TestCase):
    """End-to-end integration through PullController.compute_delta."""

    def _run_frames(
        self,
        ctrl: PullController,
        n: int,
        *,
        is_firing: bool,
        target_xy: tuple[float, float] = (200.0, 200.0),
        center_xy: tuple[float, float] = (200.0, 200.0),
        dt: float = 1.0 / 60.0,
    ) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        t = 0.0
        for _ in range(n):
            pr = ctrl.compute_delta(
                _tgt(*target_xy),
                center_xy[0],
                center_xy[1],
                time_sec=t,
                is_firing=is_firing,
            )
            out.append((pr.dx, pr.dy))
            t += dt
        return out

    def test_disabled_by_default_zero_contribution(self) -> None:
        """Default tuning with no recoil/jitter keys must emit zero on a
        centred target — same as before the feature existed."""
        ctrl = PullController(_tuning())
        deltas = self._run_frames(ctrl, 30, is_firing=True)
        # On-crosshair target inside deadzone → zero pull, zero bias.
        for dx, dy in deltas:
            self.assertEqual((dx, dy), (0, 0))

    def test_enabled_but_not_firing_zero_contribution(self) -> None:
        ctrl = PullController(
            _tuning(
                recoil_compensation_enabled=True,
                recoil_pull_down_pixels_per_second=60.0,
                jitter_enabled=True,
                jitter_amplitude_pixels=2.0,
                jitter_frequency_hz=8.0,
            )
        )
        deltas = self._run_frames(ctrl, 30, is_firing=False)
        for dx, dy in deltas:
            self.assertEqual((dx, dy), (0, 0))

    def test_enabled_and_firing_emits_bounded_delta(self) -> None:
        ctrl = PullController(
            _tuning(
                recoil_compensation_enabled=True,
                recoil_pull_down_pixels_per_second=60.0,
                jitter_enabled=True,
                jitter_amplitude_pixels=2.0,
                jitter_frequency_hz=8.0,
            )
        )
        deltas = self._run_frames(ctrl, 60, is_firing=True)
        moved = [(dx, dy) for dx, dy in deltas if dx != 0 or dy != 0]
        self.assertGreater(len(moved), 10, "Expected the compensator to emit deltas while firing")
        # Per-frame magnitude must stay small:
        #   jitter ≤ amplitude+1 px, recoil ≤ 60 px/s × 1/60 s + 1 = 2 px.
        for dx, dy in deltas:
            self.assertLess(abs(dx), 2 + 1)  # amplitude + 1
            self.assertLess(abs(dy), 2 + 1)  # per-frame recoil + 1 (residual edge)
        # Cumulative Y bias should be net downward over a second.
        total_dy = sum(dy for _, dy in deltas)
        self.assertGreater(total_dy, 30, "Recoil pull-down should accumulate downward")
        # Cumulative X bias of a pure sine should be close to zero.
        total_dx = sum(dx for dx, _ in deltas)
        self.assertLess(abs(total_dx), 6, "Jitter mean should be near zero over a full second")

    def test_hot_reload_via_update_tuning(self) -> None:
        ctrl = PullController(_tuning())
        # Initially off — no bias.
        deltas_off = self._run_frames(ctrl, 10, is_firing=True)
        self.assertTrue(all(d == (0, 0) for d in deltas_off))
        # Hot-reload to enable.
        ctrl.update_tuning(
            recoil_compensation_enabled=True,
            recoil_pull_down_pixels_per_second=120.0,
            jitter_enabled=True,
            jitter_amplitude_pixels=2.0,
            jitter_frequency_hz=8.0,
        )
        deltas_on = self._run_frames(ctrl, 60, is_firing=True)
        moved_on = [d for d in deltas_on if d != (0, 0)]
        self.assertGreater(len(moved_on), 20)
        total_dy_on = sum(dy for _, dy in deltas_on)
        self.assertGreater(total_dy_on, 80)
        # Hot-reload back to off mid-session — bias should stop on the next frame.
        ctrl.update_tuning(
            recoil_compensation_enabled=False,
            recoil_pull_down_pixels_per_second=0.0,
            jitter_enabled=False,
            jitter_amplitude_pixels=0.0,
        )
        deltas_off2 = self._run_frames(ctrl, 10, is_firing=True)
        self.assertTrue(all(d == (0, 0) for d in deltas_off2))

    def test_reset_clears_firing_state(self) -> None:
        ctrl = PullController(
            _tuning(
                recoil_compensation_enabled=False,
                recoil_pull_down_pixels_per_second=0.0,
                jitter_enabled=True,
                jitter_amplitude_pixels=2.0,
                jitter_frequency_hz=8.0,
            )
        )
        self._run_frames(ctrl, 20, is_firing=True)
        ctrl.reset()
        # After reset the compensator phase is back to zero. First firing
        # frame after reset must emit ≈ 0 horizontal jitter (sin(0)=0) and
        # because the residual was also cleared, the integer delta is 0.
        pr = ctrl.compute_delta(
            _on_target_tgt(200.0, 200.0),
            200.0,
            200.0,
            time_sec=0.0,
            is_firing=True,
        )
        self.assertEqual(pr.dx, 0)

    def test_does_not_corrupt_velocity_smoother(self) -> None:
        """
        Recoil pull-down must NOT bleed into _vel_y. Confirm by enabling
        recoil at a strong value, firing for 30 frames at a deadzone-locked
        target, then disabling firing and checking that _vel_y is still 0.
        If the bias were inside the smoother, _vel_y would have accumulated
        and the next non-firing frame would still emit deltas.
        """
        ctrl = PullController(
            _tuning(
                recoil_compensation_enabled=True,
                recoil_pull_down_pixels_per_second=120.0,
                jitter_enabled=False,
                jitter_amplitude_pixels=0.0,
            )
        )
        self._run_frames(ctrl, 30, is_firing=True)
        # _vel_x / _vel_y are unchanged by the bias path because the target
        # is in the deadzone — the smoother decay path zeros them.
        self.assertAlmostEqual(ctrl._vel_x, 0.0, places=6)
        self.assertAlmostEqual(ctrl._vel_y, 0.0, places=6)
        # Now stop firing — no further deltas.
        deltas_after = self._run_frames(ctrl, 10, is_firing=False)
        self.assertTrue(all(d == (0, 0) for d in deltas_after))

    def test_integer_truncation_and_residual_safety(self) -> None:
        """
        With a sub-pixel-per-frame recoil rate (10 px/s @ 60 fps → 0.166
        px/frame), the integer-emit must accumulate residual and emit at
        least one 1-pixel step within 12 frames, but never emit fractional
        runaway (i.e. cumulative delta within ±1 px of the expected float).
        """
        ctrl = PullController(
            _tuning(
                recoil_compensation_enabled=True,
                recoil_pull_down_pixels_per_second=10.0,
                jitter_enabled=False,
                jitter_amplitude_pixels=0.0,
            )
        )
        deltas = self._run_frames(ctrl, 120, is_firing=True)
        total_dy = sum(dy for _, dy in deltas)
        # Expected float total: 10 px/s × 2 s = 20 px. Residual should keep
        # us within 1 px of that.
        self.assertTrue(
            19 <= total_dy <= 21,
            f"Residual drain off: total_dy={total_dy}, expected ~20",
        )

    def test_concurrent_pull_and_bias_on_offset_target(self) -> None:
        """
        Off-crosshair target → normal pull engages. With recoil enabled, the
        Y output should be MORE downward than pull alone, but X output
        should be approximately the same (jitter zero-mean).
        """
        target = (260.0, 200.0)  # 60 px right of crosshair
        center = (200.0, 200.0)

        ctrl_no_recoil = PullController(_tuning())
        deltas_baseline = self._run_frames(
            ctrl_no_recoil, 30, is_firing=True, target_xy=target, center_xy=center
        )

        ctrl_recoil = PullController(
            _tuning(
                recoil_compensation_enabled=True,
                recoil_pull_down_pixels_per_second=60.0,
                jitter_enabled=False,
                jitter_amplitude_pixels=0.0,
            )
        )
        deltas_with = self._run_frames(
            ctrl_recoil, 30, is_firing=True, target_xy=target, center_xy=center
        )

        total_dy_baseline = sum(dy for _, dy in deltas_baseline)
        total_dy_recoil = sum(dy for _, dy in deltas_with)
        # Recoil branch must drift more downward (positive dy = down in screen
        # coords).
        self.assertGreater(total_dy_recoil, total_dy_baseline + 10)

        total_dx_baseline = sum(dx for dx, _ in deltas_baseline)
        total_dx_recoil = sum(dx for dx, _ in deltas_with)
        # Horizontal pull should be unchanged within rounding.
        self.assertLess(abs(total_dx_recoil - total_dx_baseline), 3)


class ProfileDefaultsTests(unittest.TestCase):
    """The new keys must default to off across every shipped profile."""

    def test_apex_style_live_trace_recoil_off(self) -> None:
        from profiles import PROFILE_APEX_STYLE_LIVE_TRACE, apply_profile

        cfg = apply_profile({"profile": PROFILE_APEX_STYLE_LIVE_TRACE})
        self.assertFalse(cfg.get("recoil_compensation_enabled", True))
        self.assertEqual(cfg.get("recoil_pull_down_pixels_per_second", -1), 0.0)
        self.assertFalse(cfg.get("jitter_enabled", True))
        self.assertEqual(cfg.get("jitter_amplitude_pixels", -1), 0.0)

    def test_apex_style_live_safe_recoil_off(self) -> None:
        from profiles import PROFILE_APEX_STYLE_LIVE_SAFE, apply_profile

        cfg = apply_profile({"profile": PROFILE_APEX_STYLE_LIVE_SAFE})
        self.assertFalse(cfg.get("recoil_compensation_enabled", True))
        self.assertFalse(cfg.get("jitter_enabled", True))

    def test_strong_preset_arms_recoil_and_jitter(self) -> None:
        try:
            from aba_gui import TUNING_PRESETS
        except ModuleNotFoundError as exc:
            # Headless CI may not have tkinter; verify the constant via source
            # inspection instead so the contract is still tested.
            if "tkinter" not in str(exc) and "_tkinter" not in str(exc):
                raise
            from pathlib import Path

            text = Path(__file__).resolve().parent.parent.joinpath("aba_gui.py").read_text()
            self.assertIn('"recoil_compensation_enabled": True', text)
            self.assertIn('"jitter_enabled": True', text)
            return

        preset = TUNING_PRESETS["Strong"]
        self.assertTrue(preset["recoil_compensation_enabled"])
        self.assertGreater(preset["recoil_pull_down_pixels_per_second"], 0.0)
        self.assertTrue(preset["jitter_enabled"])
        self.assertGreater(preset["jitter_amplitude_pixels"], 0.0)
        # Per spec: amplitude=1.5, recoil_pull=25 (the spec said "small value").
        self.assertLessEqual(preset["jitter_amplitude_pixels"], 2.0)
        self.assertLessEqual(preset["recoil_pull_down_pixels_per_second"], 35.0)

    def test_validation_accepts_recoil_keys(self) -> None:
        from config_validation import validate_config
        from profiles import apply_profile

        cfg = validate_config(apply_profile({
            "profile": "apex_style_dry_run",
            "recoil_compensation_enabled": True,
            "recoil_pull_down_pixels_per_second": 45.0,
            "jitter_enabled": True,
            "jitter_amplitude_pixels": 1.2,
            "jitter_frequency_hz": 7.5,
        }))
        self.assertTrue(cfg["recoil_compensation_enabled"])
        self.assertEqual(cfg["recoil_pull_down_pixels_per_second"], 45.0)
        self.assertTrue(cfg["jitter_enabled"])
        self.assertEqual(cfg["jitter_amplitude_pixels"], 1.2)
        self.assertEqual(cfg["jitter_frequency_hz"], 7.5)

    def test_validation_rejects_out_of_range_values(self) -> None:
        from config_validation import ConfigError, validate_config
        from profiles import apply_profile

        # Negative pull-down is rejected (minimum=0.0).
        with self.assertRaises(ConfigError):
            validate_config(apply_profile({
                "profile": "apex_style_dry_run",
                "recoil_pull_down_pixels_per_second": -5.0,
            }))
        # 200 px/s exceeds the documented 180 px/s ceiling.
        with self.assertRaises(ConfigError):
            validate_config(apply_profile({
                "profile": "apex_style_dry_run",
                "recoil_pull_down_pixels_per_second": 200.0,
            }))
        # Jitter amplitude beyond 6 px (per spec) is rejected.
        with self.assertRaises(ConfigError):
            validate_config(apply_profile({
                "profile": "apex_style_dry_run",
                "jitter_amplitude_pixels": 9.0,
            }))


if __name__ == "__main__":
    unittest.main()
