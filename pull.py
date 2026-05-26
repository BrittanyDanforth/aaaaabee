"""Smoothed mouse pull: deadzone, FOV scaling, dt-aware velocity, humanized motion."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

from detector import Target
from motion import (
    HumanizedMotion,
    RecoilCompensator,
    TargetTracker,
    alpha_from_tau,
    apply_smoothing_curve,
    fov_distance_scale,
)

_MIN_DT = 0.001
_MAX_DT = 0.12
_REF_FPS = 60.0
# Aim already smoothed in runtime — pull should correct quickly, not stack another
# heavy EMA. Pre-smoothed tau dropped further (16ms -> 11ms) so the cursor closes
# the gap to the smoothed body anchor with minimal additional lag.
_TAU_VEL_PRE_SMOOTHED = 0.011
_TAU_VEL_STANDALONE = 0.032


@dataclass
class PullTuning:
    max_speed: float
    pull_strength: float
    deadzone: float
    velocity_smoothing: float
    smoothing_curve: str
    magnetism_radius: float
    magnetism_min_scale: float
    fov_radius: float
    fov_edge_min_scale: float
    prediction_enabled: bool
    prediction_lead_seconds: float
    prediction_max_pixels: float
    humanize_enabled: bool
    humanize_amplitude: float
    humanize_jerk_limit: float
    aim_pre_smoothed: bool = True
    # Seconds to reach ~63% of desired pull velocity (lower = snappier mouse).
    velocity_tau_seconds: float = 0.0
    # Recoil compensator (engagement-gated, applied only when is_firing=True).
    recoil_compensation_enabled: bool = False
    recoil_pull_down_pixels_per_second: float = 0.0
    jitter_enabled: bool = False
    jitter_amplitude_pixels: float = 0.0
    jitter_frequency_hz: float = 6.0
    # Match mouse_gate_stale_grace_frames / may_assist_pull_target grace window.
    stale_grace_frames: int = 12


@dataclass
class PullResult:
    dx: int
    dy: int
    magnitude: float
    effective_strength: float
    distance: float
    vel_x: float = 0.0
    vel_y: float = 0.0
    desired_x: float = 0.0
    desired_y: float = 0.0




def pull_fov_distance_scale(
    dist: float,
    fov_radius: float,
    edge_min_scale: float,
) -> float:
    """
    Pull strength vs distance — edge targets must still get strong pull.
    Linear falloff is capped so FOV-edge enemies (near off-screen) are followed.
    """
    if not math.isfinite(dist) or not math.isfinite(fov_radius) or fov_radius <= 0.0:
        return 1.0
    base = fov_distance_scale(dist, fov_radius, edge_min_scale)
    t = min(1.25, max(0.0, dist / fov_radius))
    if t >= 0.85:
        return max(base, 0.82)
    return base


class PullController:
    """Convert aim error into mouse deltas — dt-aware, no second aim smooth when pre-smoothed."""

    def __init__(self, tuning: PullTuning) -> None:
        self._tuning = tuning
        self._vel_x = 0.0
        self._vel_y = 0.0
        self._residual_x = 0.0
        self._residual_y = 0.0
        # Track previous integer-emit input for sign-flip noise
        # detection in _emit_integer_delta.
        self._prev_out_x = 0.0
        self._prev_out_y = 0.0
        self._stale_count = 0
        self._last_time: float | None = None
        self._tracker = TargetTracker()
        self._humanize = HumanizedMotion(
            tuning.humanize_amplitude,
            tuning.humanize_jerk_limit,
        )
        self._recoil = self._build_recoil(tuning)

    @staticmethod
    def _build_recoil(tuning: PullTuning) -> RecoilCompensator:
        return RecoilCompensator(
            recoil_enabled=tuning.recoil_compensation_enabled,
            pull_down_px_per_s=tuning.recoil_pull_down_pixels_per_second,
            jitter_enabled=tuning.jitter_enabled,
            jitter_amplitude_px=tuning.jitter_amplitude_pixels,
            jitter_frequency_hz=tuning.jitter_frequency_hz,
        )

    def update_tuning(self, **kwargs: Any) -> None:
        """Hot-update tuning fields without resetting pull state."""
        for k, v in kwargs.items():
            if hasattr(self._tuning, k):
                object.__setattr__(self._tuning, k, v)
        if "humanize_amplitude" in kwargs or "humanize_jerk_limit" in kwargs:
            self._humanize = HumanizedMotion(
                self._tuning.humanize_amplitude,
                self._tuning.humanize_jerk_limit,
            )
        recoil_keys = {
            "recoil_compensation_enabled",
            "recoil_pull_down_pixels_per_second",
            "jitter_enabled",
            "jitter_amplitude_pixels",
            "jitter_frequency_hz",
        }
        if recoil_keys.intersection(kwargs):
            self._recoil = self._build_recoil(self._tuning)

    def reset(self) -> None:
        self._vel_x = 0.0
        self._vel_y = 0.0
        self._residual_x = 0.0
        self._residual_y = 0.0
        self._prev_out_x = 0.0
        self._prev_out_y = 0.0
        self._stale_count = 0
        self._last_time = None
        self._tracker.reset()
        self._humanize.reset()
        self._recoil.reset()

    def reset_assist_velocity(self) -> None:
        """Clear aim-pull smoothing only — preserve recoil ramp across brief target gaps."""
        self._vel_x = 0.0
        self._vel_y = 0.0
        self._residual_x = 0.0
        self._residual_y = 0.0
        self._prev_out_x = 0.0
        self._prev_out_y = 0.0
        self._stale_count = 0

    def recoil_pull_down_active(self) -> bool:
        return (
            self._tuning.recoil_compensation_enabled
            and self._tuning.recoil_pull_down_pixels_per_second > 0.0
        )

    def compute_recoil_only(
        self,
        *,
        time_sec: float | None = None,
        is_firing: bool = False,
        err_x: float = 0.0,
    ) -> PullResult:
        """Engagement recoil cancel without a lock — pull-down always fires when armed."""
        now = time.perf_counter() if time_sec is None else time_sec
        if not math.isfinite(now):
            now = time.perf_counter()
        dt = self._frame_dt(now)
        self._last_time = now
        if not is_firing or not self._recoil.active:
            self._recoil_bias(is_firing=False, dt=dt, err_x=0.0)
            return PullResult(0, 0, 0.0, 0.0, 0.0)
        bias_x, bias_y = self._recoil_bias(is_firing=True, dt=dt, err_x=err_x)
        move_x, move_y = self._emit_integer_delta(bias_x, bias_y)
        mag = math.hypot(move_x, move_y)
        return PullResult(move_x, move_y, mag, 0.0, 0.0)

    def _magnetism_scale(self, dist: float) -> float:
        radius = self._tuning.magnetism_radius
        floor = self._tuning.magnetism_min_scale
        if radius <= 0.0:
            return 1.0
        t = min(1.0, dist / radius)
        return floor + (1.0 - floor) * t

    def _frame_dt(self, now: float) -> float:
        if self._last_time is None:
            return 1.0 / _REF_FPS
        dt = now - self._last_time
        if dt <= 0.0 or not math.isfinite(dt):
            return 1.0 / _REF_FPS
        return max(_MIN_DT, min(dt, _MAX_DT))

    def _max_step_for_dt(self, dt: float) -> float:
        """
        Per-frame cap scaled so 30 FPS can move ~2x px/frame vs 60 FPS reference.
        Pre-smoothed cap raised so the cursor isn't bottlenecked when closing on
        a moving target at low capture FPS (30/45) where each frame must cover
        more distance.
        """
        cap = 4.5 if self._tuning.aim_pre_smoothed else 3.0
        scale = max(0.5, min(cap, dt * _REF_FPS))
        return self._tuning.max_speed * scale

    def _velocity_tau(self) -> float:
        if self._tuning.velocity_tau_seconds > 0.0:
            return self._tuning.velocity_tau_seconds
        return _TAU_VEL_PRE_SMOOTHED if self._tuning.aim_pre_smoothed else _TAU_VEL_STANDALONE

    def _effective_strength_multiplier(self) -> float:
        # Pre-smoothed aim has already stripped the noise; let the pull err
        # slightly toward "firmer" so it actually closes the residual gap to the
        # body anchor instead of crawling.
        if self._tuning.aim_pre_smoothed:
            return 1.15
        return 1.0

    def _aim_point(self, target: Target, now: float, *, stale_detection: bool) -> tuple[float, float]:
        if self._tuning.aim_pre_smoothed:
            return target.centroid_x, target.centroid_y

        if stale_detection:
            return target.centroid_x, target.centroid_y

        if target.bbox_w > 0 and target.bbox_h > 0:
            motion = self._tracker.observe_target(
                target.centroid_x,
                target.centroid_y,
                now,
                bbox_x=target.bbox_x,
                bbox_y=target.bbox_y,
                bbox_w=target.bbox_w,
                bbox_h=target.bbox_h,
            )
        else:
            motion = self._tracker.observe(target.centroid_x, target.centroid_y, now)

        if not self._tuning.prediction_enabled:
            return motion.x, motion.y
        return motion.predict(
            self._tuning.prediction_lead_seconds,
            self._tuning.prediction_max_pixels,
        )

    def _emit_integer_delta(self, out_x: float, out_y: float) -> tuple[int, int]:
        # SELF-AUDIT FIX (caught in pull_trace TSV F35-F46): a previous
        # patch dropped the residual-drain threshold 0.55 → 0.40 so slow
        # targets emit every other frame.  On STALE-grace frames where
        # the controller is ticked with tiny near-zero out_* (~±0.1 px)
        # the residual crossed 0.40 and emitted alternating ±1 px ticks
        # every frame — the "chunky pull at idle" feel the user
        # reported.
        #
        # Two discriminators, both required to be safe:
        #   1. ``_stale_count > 0`` — we're in stale grace, motion
        #      smoother is frozen, anything reaching the integer
        #      stage is by definition noise.  Decay residual 50 %
        #      and require the classic 0.55 drain threshold so a
        #      one-shot ±0.1 spike can't bank a 1-px tick.
        #   2. Sign-flip on |out|<0.30 — protect non-stale paths
        #      where the controller might still see oscillating
        #      detector noise.  (Sign-stable small out_* — recoil
        #      bias 0.166 px/frame, slow target tracking — still
        #      drain promptly at 0.40 so they emit correctly.)
        is_stale_noise = self._stale_count > 0
        flip_x = (
            not is_stale_noise
            and (out_x * self._prev_out_x) < 0.0
            and abs(out_x) < 0.30
        )
        flip_y = (
            not is_stale_noise
            and (out_y * self._prev_out_y) < 0.0
            and abs(out_y) < 0.30
        )
        if is_stale_noise:
            # Aggressive residual leak during stale grace.
            self._residual_x *= 0.50
            self._residual_y *= 0.50
        elif flip_x:
            self._residual_x *= 0.70
        elif flip_y:
            self._residual_y *= 0.70
        self._prev_out_x = out_x
        self._prev_out_y = out_y

        self._residual_x += out_x
        self._residual_y += out_y
        move_x = int(self._residual_x)
        move_y = int(self._residual_y)
        self._residual_x -= move_x
        self._residual_y -= move_y
        # Higher drain threshold when in stale-noise or sign-flip
        # mode; snappy 0.40 otherwise.
        if is_stale_noise or flip_x:
            drain_thresh_x = 0.55
        else:
            drain_thresh_x = 0.40
        if is_stale_noise or flip_y:
            drain_thresh_y = 0.55
        else:
            drain_thresh_y = 0.40
        if move_x == 0 and abs(self._residual_x) >= drain_thresh_x:
            move_x = 1 if self._residual_x > 0 else -1
            self._residual_x -= move_x
        if move_y == 0 and abs(self._residual_y) >= drain_thresh_y:
            move_y = 1 if self._residual_y > 0 else -1
            self._residual_y -= move_y
        return move_x, move_y

    def _recoil_bias(self, *, is_firing: bool, dt: float, err_x: float) -> tuple[float, float]:
        """Engagement-gated recoil cancel; resets compensator state when not firing."""
        if is_firing and self._recoil.active:
            return self._recoil.compute_bias(is_firing=True, dt=dt, err_x=err_x)
        if not is_firing:
            self._recoil.compute_bias(is_firing=False, dt=dt, err_x=0.0)
        return 0.0, 0.0

    def _pull_result_from_bias_only(
        self,
        *,
        is_firing: bool,
        dt: float,
        err_x: float,
        dist: float,
    ) -> PullResult:
        """Deadzone path: emit recoil cancel without aim-pull velocity."""
        bias_x, bias_y = self._recoil_bias(is_firing=is_firing, dt=dt, err_x=err_x)
        move_x, move_y = self._emit_integer_delta(bias_x, bias_y)
        return PullResult(
            move_x,
            move_y,
            math.hypot(move_x, move_y),
            0.0,
            dist,
        )

    def compute_delta(
        self,
        target: Target,
        center_x: float,
        center_y: float,
        *,
        time_sec: float | None = None,
        stale_detection: bool = False,
        is_firing: bool = False,
    ) -> PullResult:
        if not (math.isfinite(target.centroid_x) and math.isfinite(target.centroid_y)):
            return PullResult(0, 0, 0.0, 0.0, 0.0)
        # Guard caller-supplied center against NaN/Inf — would otherwise poison
        # err_x/err_y -> dist -> velocity state for the rest of the session.
        if not (math.isfinite(center_x) and math.isfinite(center_y)):
            return PullResult(0, 0, 0.0, 0.0, 0.0)
        now = time.perf_counter() if time_sec is None else time_sec
        # Never persist a non-finite timestamp; would make every subsequent dt
        # collapse to the 1/_REF_FPS fallback and prevent the smoother from ever
        # advancing again until reset().
        if not math.isfinite(now):
            now = time.perf_counter()
        dt = self._frame_dt(now)
        self._last_time = now

        if stale_detection:
            self._stale_count += 1
            if not self._tuning.aim_pre_smoothed and self._stale_count == 1:
                self._tracker.reset()
            grace = max(0, int(self._tuning.stale_grace_frames))
            if grace > 0 and self._stale_count > grace:
                decay = alpha_from_tau(dt, 0.08)
                self._vel_x *= 1.0 - decay
                self._vel_y *= 1.0 - decay
        else:
            self._stale_count = 0

        aim_x, aim_y = self._aim_point(target, now, stale_detection=stale_detection)

        if not (math.isfinite(aim_x) and math.isfinite(aim_y)):
            return PullResult(0, 0, 0.0, 0.0, 0.0)

        err_x = aim_x - center_x
        err_y = aim_y - center_y
        # Only clamp absurd errors (detector teleport); normal FOV offset must pull through.
        # Floor was a hard 48 px regardless of FOV size, which clipped legitimate
        # engagements when a small/distant target sat near the rim of a wide
        # detection FOV (e.g. bbox_w=15 at err_x=130 with fov_radius=140 was
        # being clamped to 48 — pull strength fell from "edge target" to "near
        # centre" levels). Scale the floor with fov_radius so edge engagement
        # is proportional regardless of bbox size; teleport-defense above the
        # detection circle is still handled by the dist > fov_radius * 1.02 cutoff.
        if target.bbox_w > 0 and target.bbox_h > 0:
            fov_floor = self._tuning.fov_radius * 0.6
            max_ex = max(fov_floor, target.bbox_w * 2.5)
            max_ey = max(fov_floor * 0.85, target.bbox_h * 2.2)
            if abs(err_x) > max_ex:
                err_x = max(-max_ex, min(max_ex, err_x))
            if abs(err_y) > max_ey:
                err_y = max(-max_ey, min(max_ey, err_y))
        dist = math.hypot(err_x, err_y)
        if not math.isfinite(dist):
            # NaN aim_x/aim_y already early-returned, but defend against the
            # extremely rare case where err_* arithmetic overflows to inf.
            self._vel_x = 0.0
            self._vel_y = 0.0
            self._residual_x = 0.0
            self._residual_y = 0.0
            self._recoil.reset()
            return PullResult(0, 0, 0.0, 0.0, 0.0)
        if dist > self._tuning.fov_radius * 1.02:
            self._vel_x *= 0.0
            self._vel_y *= 0.0
            self._residual_x = 0.0
            self._residual_y = 0.0
            self._recoil.reset()
            return PullResult(0, 0, 0.0, 0.0, dist)

        dead = max(0.0, float(self._tuning.deadzone))
        in_deadzone = dead > 0.0 and dist <= dead
        if in_deadzone and not self._tuning.aim_pre_smoothed:
            decay = apply_smoothing_curve(
                self._tuning.velocity_smoothing,
                self._tuning.smoothing_curve,
            )
            decay = alpha_from_tau(dt, self._velocity_tau() * max(0.5, decay))
            self._vel_x *= 1.0 - decay
            self._vel_y *= 1.0 - decay
            # Recoil compensation is engagement-gated, not aim-gated: when the
            # user is firing at a centred (in-deadzone) target the gun is still
            # recoiling, so pull-down / lateral-hold biases must still emit.
            # Applied as additive float bias through the residual accumulator,
            # bypassing the velocity smoother (otherwise bias bleeds into _vel_*).
            if is_firing and self._recoil.active:
                return self._pull_result_from_bias_only(
                    is_firing=True, dt=dt, err_x=err_x, dist=dist
                )
            self._residual_x = 0.0
            self._residual_y = 0.0
            self._recoil.compute_bias(is_firing=False, dt=dt, err_x=0.0)
            return PullResult(0, 0, 0.0, 0.0, dist)

        deadzone_scale = 1.0
        if in_deadzone and self._tuning.aim_pre_smoothed:
            if dist <= 0.5:
                deadzone_scale = 0.0
            else:
                ramp = (dist - 0.5) / max(dead - 0.5, 1.0)
                deadzone_scale = max(0.20, min(1.0, ramp))
            if deadzone_scale <= 0.0 and is_firing and self._recoil.active:
                return self._pull_result_from_bias_only(
                    is_firing=True, dt=dt, err_x=err_x, dist=dist
                )
            if deadzone_scale <= 0.0:
                self._residual_x = 0.0
                self._residual_y = 0.0
                self._recoil.compute_bias(is_firing=False, dt=dt, err_x=0.0)
                return PullResult(0, 0, 0.0, 0.0, dist)

        magnet = self._magnetism_scale(dist)
        fov_scale = pull_fov_distance_scale(
            dist,
            self._tuning.fov_radius,
            self._tuning.fov_edge_min_scale,
        )
        strength = (
            self._tuning.pull_strength
            * magnet
            * fov_scale
            * deadzone_scale
            * self._effective_strength_multiplier()
        )
        desired_x = err_x * strength
        desired_y = err_y * strength

        tau = self._velocity_tau()
        norm_dist = min(1.0, dist / max(self._tuning.fov_radius, 1.0))
        smooth_weight = self._tuning.velocity_smoothing * (0.35 + 0.65 * norm_dist)
        if self._tuning.aim_pre_smoothed:
            smooth_weight *= 0.50
        tau_eff = tau * (0.5 + 0.5 * smooth_weight)
        alpha = alpha_from_tau(dt, tau_eff)
        alpha = apply_smoothing_curve(alpha, self._tuning.smoothing_curve)
        alpha = max(alpha, alpha_from_tau(dt, tau))
        # Closing-distance engagement: when the cursor is meaningfully off-anchor,
        # tighten the velocity filter so the gap closes within a couple frames.
        if self._tuning.aim_pre_smoothed and dist > 4.0:
            alpha = max(
                alpha,
                alpha_from_tau(dt, 0.006) * min(1.0, max(0.35, dist / 28.0)),
            )

        if not self._tuning.aim_pre_smoothed:
            vel_mag = math.hypot(self._vel_x, self._vel_y)
            engage_threshold = self._tuning.max_speed * 0.35
            if vel_mag < engage_threshold:
                alpha = min(1.0, alpha * 1.35)

        self._vel_x += alpha * (desired_x - self._vel_x)
        self._vel_y += alpha * (desired_y - self._vel_y)

        max_step = self._max_step_for_dt(dt)
        mag = math.hypot(self._vel_x, self._vel_y)
        if mag > max_step:
            scale = max_step / mag
            self._vel_x *= scale
            self._vel_y *= scale

        out_x, out_y = self._vel_x, self._vel_y
        if self._tuning.humanize_enabled:
            out_x, out_y = self._humanize.apply(out_x, out_y)

        # Recoil cancel is added AFTER the velocity smoother (no _vel_* bleed)
        # but BEFORE integer truncation (residual drains sub-pixel bias).
        bias_x, bias_y = self._recoil_bias(is_firing=is_firing, dt=dt, err_x=err_x)
        out_x += bias_x
        out_y += bias_y

        move_x, move_y = self._emit_integer_delta(out_x, out_y)
        mag = math.hypot(move_x, move_y)
        return PullResult(
            move_x,
            move_y,
            mag,
            strength,
            dist,
            vel_x=self._vel_x,
            vel_y=self._vel_y,
            desired_x=desired_x,
            desired_y=desired_y,
        )
