"""Smoothed mouse pull: deadzone, FOV scaling, easing curves, prediction, humanized motion."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from detector import Target
from motion import (
    HumanizedMotion,
    TargetTracker,
    apply_smoothing_curve,
    fov_distance_scale,
)


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
    # Runtime already runs motion.observe_target(bbox) — avoid a second tracker lag stack.
    aim_pre_smoothed: bool = True


@dataclass
class PullResult:
    dx: int
    dy: int
    magnitude: float
    effective_strength: float
    distance: float


class PullController:
    """Velocity smoothing with easing, magnetism, FOV scaling, lead prediction, humanization."""

    def __init__(self, tuning: PullTuning) -> None:
        self._tuning = tuning
        self._vel_x = 0.0
        self._vel_y = 0.0
        self._residual_x = 0.0
        self._residual_y = 0.0
        self._stale_count = 0
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
        self._tracker.reset()
        self._humanize.reset()

    def _magnetism_scale(self, dist: float) -> float:
        radius = self._tuning.magnetism_radius
        floor = self._tuning.magnetism_min_scale
        if radius <= 0.0:
            return 1.0
        t = min(1.0, dist / radius)
        return floor + (1.0 - floor) * t

    def _aim_point(self, target: Target, now: float, *, stale_detection: bool) -> tuple[float, float]:
        """Aim anchor: pre-smoothed centroid from runtime, or internal tracker when standalone."""
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

        if stale_detection:
            self._stale_count += 1
            if not self._tuning.aim_pre_smoothed and self._stale_count == 1:
                self._tracker.reset()
            if self._stale_count > 8:
                self._vel_x *= 0.85
                self._vel_y *= 0.85
        else:
            self._stale_count = 0

        aim_x, aim_y = self._aim_point(target, now, stale_detection=stale_detection)

        if not (math.isfinite(aim_x) and math.isfinite(aim_y)):
            return PullResult(0, 0, 0.0, 0.0, 0.0)

        dx = aim_x - center_x
        dy = aim_y - center_y
        dist = math.hypot(dx, dy)
        if dist <= self._tuning.deadzone:
            decay = apply_smoothing_curve(
                self._tuning.velocity_smoothing,
                self._tuning.smoothing_curve,
            )
            self._vel_x *= 1.0 - decay
            self._vel_y *= 1.0 - decay
            self._residual_x = 0.0
            self._residual_y = 0.0
            return PullResult(0, 0, 0.0, 0.0, dist)

        magnet = self._magnetism_scale(dist)
        fov_scale = fov_distance_scale(
            dist,
            self._tuning.fov_radius,
            self._tuning.fov_edge_min_scale,
        )
        strength = self._tuning.pull_strength * magnet * fov_scale
        desired_x = dx * strength
        desired_y = dy * strength

        norm_dist = min(1.0, dist / max(self._tuning.fov_radius, 1.0))
        base_smoothing = self._tuning.velocity_smoothing * (0.5 + 0.5 * norm_dist)
        vel_mag = math.hypot(self._vel_x, self._vel_y)
        engage_threshold = self._tuning.max_speed * 0.35
        if vel_mag < engage_threshold:
            base_smoothing = min(1.0, base_smoothing * 2.5)
        alpha = apply_smoothing_curve(base_smoothing, self._tuning.smoothing_curve)
        self._vel_x += alpha * (desired_x - self._vel_x)
        self._vel_y += alpha * (desired_y - self._vel_y)

        mag = math.hypot(self._vel_x, self._vel_y)
        if mag > self._tuning.max_speed:
            scale = self._tuning.max_speed / mag
            self._vel_x *= scale
            self._vel_y *= scale

        out_x, out_y = self._vel_x, self._vel_y
        if self._tuning.humanize_enabled:
            out_x, out_y = self._humanize.apply(out_x, out_y)

        self._residual_x += out_x
        self._residual_y += out_y
        move_x = int(self._residual_x)
        move_y = int(self._residual_y)
        self._residual_x -= move_x
        self._residual_y -= move_y
        mag = math.hypot(move_x, move_y)
        return PullResult(move_x, move_y, mag, strength, dist)
