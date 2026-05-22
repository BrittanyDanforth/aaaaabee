"""Smoothed mouse pull: deadzone, FOV scaling, dt-aware velocity, humanized motion."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from detector import Target
from motion import (
    HumanizedMotion,
    TargetTracker,
    alpha_from_tau,
    apply_smoothing_curve,
    fov_distance_scale,
)

_MIN_DT = 0.001
_MAX_DT = 0.12
_REF_FPS = 60.0
# Aim already smoothed in runtime — pull should correct quickly, not stack another heavy EMA.
_TAU_VEL_PRE_SMOOTHED = 0.016
_TAU_VEL_STANDALONE = 0.038


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

    def reset(self) -> None:
        self._vel_x = 0.0
        self._vel_y = 0.0
        self._residual_x = 0.0
        self._residual_y = 0.0
        self._stale_count = 0
        self._last_time = None
        self._tracker.reset()
        self._humanize.reset()

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
        """Per-frame cap scaled so 30 FPS can move ~2x px/frame vs 60 FPS reference."""
        cap = 3.6 if self._tuning.aim_pre_smoothed else 2.5
        scale = max(0.5, min(cap, dt * _REF_FPS))
        return self._tuning.max_speed * scale

    def _velocity_tau(self) -> float:
        if self._tuning.velocity_tau_seconds > 0.0:
            return self._tuning.velocity_tau_seconds
        return _TAU_VEL_PRE_SMOOTHED if self._tuning.aim_pre_smoothed else _TAU_VEL_STANDALONE

    def _effective_strength_multiplier(self) -> float:
        if self._tuning.aim_pre_smoothed:
            return 1.05
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
    ) -> PullResult:
        if not (math.isfinite(target.centroid_x) and math.isfinite(target.centroid_y)):
            return PullResult(0, 0, 0.0, 0.0, 0.0)
        now = time.perf_counter() if time_sec is None else time_sec
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
        if target.bbox_w > 0 and target.bbox_h > 0:
            max_ex = max(48.0, target.bbox_w * 2.5)
            max_ey = max(40.0, target.bbox_h * 2.2)
            if abs(err_x) > max_ex:
                err_x = max(-max_ex, min(max_ex, err_x))
            if abs(err_y) > max_ey:
                err_y = max(-max_ey, min(max_ey, err_y))
        dist = math.hypot(err_x, err_y)
        if dist > self._tuning.fov_radius * 1.02:
            self._vel_x *= 0.0
            self._vel_y *= 0.0
            self._residual_x = 0.0
            self._residual_y = 0.0
            return PullResult(0, 0, 0.0, 0.0, dist)

        if dist <= self._tuning.deadzone:
            decay = apply_smoothing_curve(
                self._tuning.velocity_smoothing,
                self._tuning.smoothing_curve,
            )
            decay = alpha_from_tau(dt, self._velocity_tau() * max(0.5, decay))
            self._vel_x *= 1.0 - decay
            self._vel_y *= 1.0 - decay
            self._residual_x = 0.0
            self._residual_y = 0.0
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
            smooth_weight *= 0.55
        tau_eff = tau * (0.5 + 0.5 * smooth_weight)
        alpha = alpha_from_tau(dt, tau_eff)
        alpha = apply_smoothing_curve(alpha, self._tuning.smoothing_curve)
        alpha = max(alpha, alpha_from_tau(dt, tau))
        if self._tuning.aim_pre_smoothed and dist > 18.0:
            alpha = max(alpha, alpha_from_tau(dt, 0.008) * min(1.0, dist / 50.0))

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
