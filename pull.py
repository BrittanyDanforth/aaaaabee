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

    def set_runtime_fov_radius(self, radius: float) -> None:
        """Update runtime-computed detection FOV radius safely."""
        r = float(radius)
        if not math.isfinite(r) or r <= 0.0:
            return
        object.__setattr__(self._tuning, "fov_radius", r)

    def reset(self) -> None:
        self._vel_x = 0.0
        self._vel_y = 0.0
        self._residual_x = 0.0
        self._residual_y = 0.0
        self._stale_count = 0
        self._last_time = None
        self._tracker.reset()
        self._humanize.reset()
        self._recoil.reset()

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
        self._residual_x += out_x
        self._residual_y += out_y
        move_x = int(self._residual_x)
        move_y = int(self._residual_y)
        self._residual_x -= move_x
        self._residual_y -= move_y
        # Drain fractional bank so slow correction is not stuck at 0 for many frames.
        if move_x == 0 and abs(self._residual_x) >= 0.55:
            move_x = 1 if self._residual_x > 0 else -1
            self._residual_x -= move_x
        if move_y == 0 and abs(self._residual_y) >= 0.55:
            move_y = 1 if self._residual_y > 0 else -1
            self._residual_y -= move_y
        return move_x, move_y

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
            if self._stale_count > 12:
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

        if dist <= self._tuning.deadzone:
            decay = apply_smoothing_curve(
                self._tuning.velocity_smoothing,
                self._tuning.smoothing_curve,
            )
            decay = alpha_from_tau(dt, self._velocity_tau() * max(0.5, decay))
            self._vel_x *= 1.0 - decay
            self._vel_y *= 1.0 - decay
            # Recoil compensation is engagement-gated, not aim-gated: when the
            # user is firing at a centred (in-deadzone) target the gun is still
            # recoiling, so the pull-down / horizontal-jitter biases must still
            # emit. They are applied as a separate additive integer delta,
            # bypassing the velocity smoother and the residual accumulator
            # (otherwise the bias would bleed into _vel_y and the cursor would
            # keep drifting downward for several frames after release).
            if is_firing and self._recoil.active:
                bias_x, bias_y = self._recoil.compute_bias(is_firing=True, dt=dt)
                self._residual_x += bias_x
                self._residual_y += bias_y
                move_x = int(self._residual_x)
                move_y = int(self._residual_y)
                self._residual_x -= move_x
                self._residual_y -= move_y
                if move_x == 0 and abs(self._residual_x) >= 0.55:
                    move_x = 1 if self._residual_x > 0 else -1
                    self._residual_x -= move_x
                if move_y == 0 and abs(self._residual_y) >= 0.55:
                    move_y = 1 if self._residual_y > 0 else -1
                    self._residual_y -= move_y
                return PullResult(
                    move_x,
                    move_y,
                    math.hypot(move_x, move_y),
                    0.0,
                    dist,
                )
            self._residual_x = 0.0
            self._residual_y = 0.0
            self._recoil.compute_bias(is_firing=False, dt=dt)
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
        if self._tuning.aim_pre_smoothed and dist > 14.0:
            alpha = max(alpha, alpha_from_tau(dt, 0.006) * min(1.0, dist / 40.0))

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

        # Recoil + jitter bias is added AFTER the velocity smoother (so it
        # doesn't bleed into _vel_x/_vel_y and cause oscillation post-fire)
        # but BEFORE integer truncation (so the fractional pixel residual
        # accumulator drains the bias correctly across frames).
        if is_firing and self._recoil.active:
            bias_x, bias_y = self._recoil.compute_bias(is_firing=True, dt=dt)
            out_x += bias_x
            out_y += bias_y
        elif not is_firing:
            self._recoil.compute_bias(is_firing=False, dt=dt)

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
