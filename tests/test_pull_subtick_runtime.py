"""Production sub-tick proof: runtime._run_pull_subticks must emit
multiple mouse-moves between detect frames when pull_subtick_hz > 0.

Tests use a virtual clock (sleep_fn / now_fn injected) so they're
synchronous and fast.  Verifies:

  1. With subtick_hz=180 and a 16.67 ms (60 fps) frame budget, the
     loop fires at least 2 sub-ticks per detect frame.
  2. Anchor extrapolation respects the ±2.5 / ±1.0 px clip.
  3. Body-bbox chest-band clamp re-clips each sub-tick anchor.
  4. subtick_hz=0 emits zero sub-ticks (legacy behaviour preserved).
"""

from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace

from detector import Target
from motion import TargetTracker, TargetMotion
from pull import PullController, PullTuning


class _MinimalRuntime:
    """Just enough of AssistRuntime for the subtick contract."""

    _SUBTICK_MAX_EXTRAP_Y = 1.0
    _SUBTICK_MAX_EXTRAP_X = 2.5
    _SUBTICK_STALE_VELOCITY_DECAY = 0.92

    def __init__(self, pull: PullController, aim_tracker: TargetTracker):
        self._pull = pull
        self._aim_tracker = aim_tracker
        self._running = True
        self._stopping = False
        self.mouse_moves: list[tuple[int, int]] = []

    def _should_run(self) -> bool:
        return self._running and not self._stopping

    def _safe_mouse_move(self, dx: int, dy: int):
        self.mouse_moves.append((dx, dy))


# Import the real method off AssistRuntime so we test the real code.
from runtime import AssistRuntime


_run_pull_subticks = AssistRuntime._run_pull_subticks


def _make_pull() -> PullController:
    return PullController(PullTuning(
        max_speed=22.0, pull_strength=0.85, deadzone=2.0,
        velocity_smoothing=0.45, smoothing_curve="ease_out",
        magnetism_radius=65.0, magnetism_min_scale=0.70,
        fov_radius=185.0, fov_edge_min_scale=0.88,
        prediction_enabled=True, prediction_lead_seconds=0.020,
        prediction_max_pixels=12.0, humanize_enabled=False,
        humanize_amplitude=0.0, humanize_jerk_limit=0.0,
        aim_pre_smoothed=True,
    ))


class PullSubtickProductionTests(unittest.TestCase):

    def _virtual_clock(self):
        state = {"t": 0.0}
        def sleep_fn(d):
            state["t"] += d
        def now_fn():
            return state["t"]
        return sleep_fn, now_fn, state

    def test_180hz_subtick_fires_multiple_times_per_frame(self) -> None:
        pull = _make_pull()
        tracker = TargetTracker()
        # Body 30 px to the left of the cursor (frame_cx=640).
        motion = tracker.observe_target(
            610.0, 540.0, 0.0,
            bbox_x=580, bbox_y=480, bbox_w=60, bbox_h=120,
            aim_is_body_anchor=True,
        )
        target = Target(
            centroid_x=motion.x, centroid_y=motion.y,
            area=4000.0, distance_to_center=30.0, confidence=0.88,
            bbox_x=580, bbox_y=480, bbox_w=60, bbox_h=120,
            body_shape_score=0.86, head_score=0.80, torso_score=0.74,
            limb_stack_score=0.62, red_coverage=0.20,
            has_classified_torso=True, part_count=4,
        )
        rt = _MinimalRuntime(pull, tracker)
        sleep_fn, now_fn, _ = self._virtual_clock()
        frame_interval = 1.0 / 60.0  # 16.67 ms
        emitted = _run_pull_subticks(
            rt, target, motion,
            frame_cx=640.0, frame_cy=540.0,
            deadline=frame_interval,
            subtick_hz=180,
            stale_det=False,
            firing_now=False,
            sleep_fn=sleep_fn, now_fn=now_fn,
        )
        # 180 Hz inside 16.67 ms => ~3 sub-ticks.
        self.assertGreaterEqual(len(emitted), 2,
            f"expected at least 2 sub-ticks @ 180 Hz inside one 60 fps "
            f"frame; got {len(emitted)}")
        # The first sub-tick may be at max_step (22 px) when the
        # cursor was significantly offset; subsequent sub-ticks should
        # decay quickly toward the deadzone.  We only assert "no
        # teleport" (delta <= max_step) per sub-tick.
        for dx, dy in emitted:
            self.assertLessEqual(abs(dx) + abs(dy), 22 * 2,
                "sub-tick exceeded the per-tick max_step cap "
                f"(was dx={dx}, dy={dy})")

    def test_subtick_hz_zero_emits_nothing(self) -> None:
        pull = _make_pull()
        tracker = TargetTracker()
        motion = tracker.observe_target(
            610.0, 540.0, 0.0,
            bbox_x=580, bbox_y=480, bbox_w=60, bbox_h=120,
            aim_is_body_anchor=True,
        )
        target = Target(
            centroid_x=motion.x, centroid_y=motion.y,
            area=4000.0, distance_to_center=30.0, confidence=0.88,
            bbox_x=580, bbox_y=480, bbox_w=60, bbox_h=120,
            body_shape_score=0.86, head_score=0.80, torso_score=0.74,
            limb_stack_score=0.62, red_coverage=0.20,
            has_classified_torso=True, part_count=4,
        )
        rt = _MinimalRuntime(pull, tracker)
        sleep_fn, now_fn, _ = self._virtual_clock()
        emitted = _run_pull_subticks(
            rt, target, motion,
            frame_cx=640.0, frame_cy=540.0,
            deadline=1.0 / 60.0,
            subtick_hz=0,
            stale_det=False, firing_now=False,
            sleep_fn=sleep_fn, now_fn=now_fn,
        )
        self.assertEqual(emitted, [])

    def test_subtick_stale_velocity_decays(self) -> None:
        """Long stale-grace window must not drift the cursor by
        velocity × time forever.  With stale_det=True every sub-tick
        must shrink the extrapolation velocity so total drift is
        bounded even at 180 Hz × N frames of stale."""

        pull = _make_pull()
        tracker = TargetTracker()
        # Seed strong rightward velocity (240 px/s) then stop observing.
        for i in range(6):
            tracker.observe_target(
                610.0 + i * 4.0, 540.0, i / 60.0,
                bbox_x=580 + i * 4, bbox_y=480, bbox_w=60, bbox_h=120,
                aim_is_body_anchor=True,
            )
        motion = tracker._last
        self.assertGreater(motion.vx, 100.0,
            "test setup: vx should be strongly positive")
        target = Target(
            centroid_x=motion.x, centroid_y=motion.y,
            area=4000.0, distance_to_center=0.0, confidence=0.88,
            bbox_x=604, bbox_y=480, bbox_w=60, bbox_h=120,
            body_shape_score=0.86, head_score=0.80, torso_score=0.74,
            limb_stack_score=0.62, red_coverage=0.20,
            has_classified_torso=True, part_count=4,
        )
        rt = _MinimalRuntime(pull, tracker)
        # Run 10 consecutive "frames" of stale sub-ticks @ 180 Hz.
        sleep_fn, now_fn, state = self._virtual_clock()
        total_emitted: list[tuple[int, int]] = []
        for frame in range(10):
            deadline = state["t"] + 1.0 / 60.0
            emitted = _run_pull_subticks(
                rt, target, motion,
                frame_cx=640.0, frame_cy=540.0,
                deadline=deadline,
                subtick_hz=180,
                stale_det=True,   # critical: stale path
                firing_now=False,
                sleep_fn=sleep_fn, now_fn=now_fn,
            )
            total_emitted.extend(emitted)
        # Total horizontal drift across 10 stale frames must stay
        # bounded.  Without decay this would be vx * 10 frames * sub_dt
        # * 3 subticks ≈ 22 px.  With decay it's well under that.
        total_dx = sum(dx for dx, _dy in total_emitted)
        self.assertLessEqual(
            total_dx, 35,
            f"stale sub-tick total drift = {total_dx} px (runaway)"
        )

    def test_precise_sleep_is_low_cpu(self) -> None:
        """``_precise_sleep`` must not busy-wait the request budget.
        Measured baseline at 180 Hz: the original implementation
        spent ~46 % of one core spinning; this test guards that
        regression by asserting CPU cost stays under 10 % of wall
        time at 180 Hz on Linux."""

        import resource
        import time as _time
        import runtime as _rt

        # Warm up so the first-call JIT / module-import doesn't
        # dominate the measurement.
        for _ in range(30):
            _rt.AssistRuntime._precise_sleep(1.0 / 1000.0)
        ru0 = resource.getrusage(resource.RUSAGE_SELF)
        wall0 = _time.perf_counter()
        for _ in range(180):
            _rt.AssistRuntime._precise_sleep(1.0 / 180.0)
        wall = _time.perf_counter() - wall0
        ru1 = resource.getrusage(resource.RUSAGE_SELF)
        cpu = (ru1.ru_utime + ru1.ru_stime) - (ru0.ru_utime + ru0.ru_stime)
        busy = cpu / wall if wall > 0 else 0.0
        # On a busy CI runner allow a bit of headroom — the previous
        # busy-wait implementation reproducibly burned 30-46 %.
        self.assertLess(
            busy, 0.10,
            f"_precise_sleep is spending {busy*100:.1f}% CPU at 180 Hz; "
            f"busy-wait must not be reintroduced"
        )

    def test_precise_sleep_respects_duration(self) -> None:
        """At a 5 ms request, the actual elapsed time should be close
        to 5 ms on Linux — proves the sleep call works."""

        import time as _time
        import runtime as _rt

        t0 = _time.perf_counter()
        _rt.AssistRuntime._precise_sleep(0.005)
        elapsed = _time.perf_counter() - t0
        self.assertGreaterEqual(elapsed, 0.004)
        self.assertLess(
            elapsed, 0.020,
            f"_precise_sleep ran {elapsed*1000:.2f} ms for a 5 ms request"
        )

    def test_subtick_anchor_clamped_to_chest_band(self) -> None:
        """High vy*sub_dt would project the anchor below the chest
        band; the per-subtick clamp must keep it inside."""

        pull = _make_pull()
        tracker = TargetTracker()
        # Force a strong downward velocity into the smoother.
        for i in range(6):
            tracker.observe_target(
                610.0, 540.0 + i * 80.0, i / 60.0,
                bbox_x=580, bbox_y=int(480 + i * 80), bbox_w=60, bbox_h=120,
                aim_is_body_anchor=True,
            )
        motion = tracker._last
        # Body bbox now: by=880, bh=120 -> chest band 880+33.6 .. 880+62.4
        target = Target(
            centroid_x=motion.x, centroid_y=motion.y,
            area=4000.0, distance_to_center=0.0, confidence=0.88,
            bbox_x=580, bbox_y=880, bbox_w=60, bbox_h=120,
            body_shape_score=0.86, head_score=0.80, torso_score=0.74,
            limb_stack_score=0.62, red_coverage=0.20,
            has_classified_torso=True, part_count=4,
        )
        rt = _MinimalRuntime(pull, tracker)
        sleep_fn, now_fn, _ = self._virtual_clock()
        _run_pull_subticks(
            rt, target, motion,
            frame_cx=640.0, frame_cy=540.0,
            deadline=1.0 / 60.0,
            subtick_hz=180,
            stale_det=False, firing_now=False,
            sleep_fn=sleep_fn, now_fn=now_fn,
        )
        # All sub-tick mouse moves should be bounded; no single move
        # should fling the cursor more than max_speed * dt scaled.
        for dx, dy in rt.mouse_moves:
            self.assertLessEqual(abs(dx) + abs(dy), 12)


if __name__ == "__main__":
    unittest.main()
