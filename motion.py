"""Easing curves, FOV distance scaling, target prediction, and humanized micro-motion."""

from __future__ import annotations

import math
from dataclasses import dataclass

_MAX_VELOCITY = 2800.0
_MAX_DT = 0.5
_MIN_DT = 0.005
_JUMP_THRESHOLD = 180.0
_VELOCITY_NOISE_FLOOR = 3.5
_POSITION_SMOOTH_ALPHA = 0.38
_SOFT_JUMP_ALPHA = 0.22


def _finite(v: float, fallback: float = 0.0) -> float:
    return v if math.isfinite(v) else fallback


def apply_smoothing_curve(t: float, curve: str) -> float:
    """Map blend factor t in [0, 1] through an easing curve."""
    if not math.isfinite(t):
        return 0.0
    t = max(0.0, min(1.0, t))
    if curve == "ease_out":
        return 1.0 - (1.0 - t) ** 2
    if curve == "ease_in_out":
        return t * t * (3.0 - 2.0 * t)
    return t


def fov_distance_scale(dist: float, fov_radius: float, edge_min_scale: float) -> float:
    """Weaker pull near the FOV edge, full strength at the crosshair."""
    if not math.isfinite(dist) or not math.isfinite(fov_radius):
        return 1.0
    if fov_radius <= 0.0:
        return 1.0
    t = min(1.0, max(0.0, dist / fov_radius))
    floor = max(0.0, min(1.0, _finite(edge_min_scale, 0.0)))
    return floor + (1.0 - floor) * (1.0 - t)


@dataclass
class TargetMotion:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0

    def predict(
        self,
        lead_seconds: float,
        max_lead_pixels: float,
    ) -> tuple[float, float]:
        if lead_seconds <= 0.0 or max_lead_pixels <= 0.0:
            return self.x, self.y
        ls = min(_finite(lead_seconds, 0.0), 0.12)
        mlp = min(_finite(max_lead_pixels, 0.0), 120.0)
        vx = _finite(self.vx, 0.0)
        vy = _finite(self.vy, 0.0)
        ox = vx * ls
        oy = vy * ls
        lead_dist = math.hypot(ox, oy)
        if not math.isfinite(lead_dist) or lead_dist <= 0.0:
            return self.x, self.y
        if lead_dist > mlp:
            scale = mlp / lead_dist
            ox *= scale
            oy *= scale
        px = _finite(self.x, 0.0) + ox
        py = _finite(self.y, 0.0) + oy
        if not (math.isfinite(px) and math.isfinite(py)):
            return self.x, self.y
        return px, py


class TargetTracker:
    """Per-target velocity estimate from successive centroid samples."""

    def __init__(self) -> None:
        self._last: TargetMotion | None = None
        self._last_time: float | None = None
        self._smooth_x: float | None = None
        self._smooth_y: float | None = None
        self._last_raw_x: float | None = None
        self._last_raw_y: float | None = None

    def reset(self) -> None:
        self._last = None
        self._last_time = None
        self._smooth_x = None
        self._smooth_y = None
        self._last_raw_x = None
        self._last_raw_y = None

    def _smooth_position(self, x: float, y: float, *, soft: bool = False) -> tuple[float, float]:
        alpha = _SOFT_JUMP_ALPHA if soft else _POSITION_SMOOTH_ALPHA
        if self._smooth_x is None or self._smooth_y is None:
            self._smooth_x = x
            self._smooth_y = y
        else:
            self._smooth_x = alpha * x + (1.0 - alpha) * self._smooth_x
            self._smooth_y = alpha * y + (1.0 - alpha) * self._smooth_y
        return self._smooth_x, self._smooth_y

    def observe(self, x: float, y: float, time_sec: float) -> TargetMotion:
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(time_sec)):
            if self._last is not None:
                return self._last
            return TargetMotion(
                x if math.isfinite(x) else 0.0,
                y if math.isfinite(y) else 0.0,
                0.0,
                0.0,
            )

        if self._last is None or self._last_time is None:
            sx, sy = self._smooth_position(x, y)
            motion = TargetMotion(sx, sy, 0.0, 0.0)
        else:
            dt = time_sec - self._last_time
            if dt < 0:
                sx, sy = self._smooth_position(x, y, soft=True)
                motion = TargetMotion(sx, sy, 0.0, 0.0)
            elif dt < _MIN_DT:
                motion = TargetMotion(self._last.x, self._last.y, self._last.vx, self._last.vy)
                self._last_raw_x = x
                self._last_raw_y = y
                return motion
            else:
                dt = min(dt, _MAX_DT)
                lx = self._last_raw_x if self._last_raw_x is not None else self._last.x
                ly = self._last_raw_y if self._last_raw_y is not None else self._last.y
                jump = math.hypot(x - lx, y - ly)
                if jump > _JUMP_THRESHOLD:
                    sx, sy = self._smooth_position(x, y, soft=True)
                    motion = TargetMotion(sx, sy, 0.0, 0.0)
                else:
                    vx = (x - lx) / dt
                    vy = (y - ly) / dt
                    blend = min(1.0, 0.22 + dt * 18.0)
                    raw_vx = self._last.vx + blend * (vx - self._last.vx)
                    raw_vy = self._last.vy + blend * (vy - self._last.vy)
                    vmag = math.hypot(raw_vx, raw_vy)
                    if vmag > _MAX_VELOCITY:
                        scale = _MAX_VELOCITY / vmag
                        raw_vx *= scale
                        raw_vy *= scale
                    if vmag < _VELOCITY_NOISE_FLOOR:
                        raw_vx = 0.0
                        raw_vy = 0.0
                    sx, sy = self._smooth_position(x, y)
                    motion = TargetMotion(sx, sy, raw_vx, raw_vy)

        self._last = motion
        self._last_time = time_sec
        self._last_raw_x = x
        self._last_raw_y = y
        return motion


class HumanizedMotion:
    """Bounded micro-variation and jerk limiting so steps are not perfectly linear."""

    def __init__(self, amplitude: float, jerk_limit: float) -> None:
        self._amplitude = max(0.0, min(amplitude, 2.0))
        self._jerk_limit = max(0.0, jerk_limit)
        self._prev_dx = 0.0
        self._prev_dy = 0.0
        self._frame = 0

    def reset(self) -> None:
        self._prev_dx = 0.0
        self._prev_dy = 0.0
        self._frame = 0

    def apply(self, dx: float, dy: float) -> tuple[float, float]:
        dx = _finite(dx, 0.0)
        dy = _finite(dy, 0.0)
        if self._amplitude <= 0.0 and self._jerk_limit <= 0.0:
            self._prev_dx = dx
            self._prev_dy = dy
            return dx, dy

        self._frame += 1
        mag = math.hypot(dx, dy)
        wobble_scale = min(0.35, mag / max(self._amplitude * 6.0, 1.0))
        if mag < 2.0:
            wobble_scale = 0.0
        n = self._frame
        wobble_x = self._amplitude * wobble_scale * math.sin(n * 0.73) * math.cos(n * 0.19)
        wobble_y = self._amplitude * wobble_scale * math.cos(n * 0.61) * math.sin(n * 0.23)

        out_x = dx + wobble_x
        out_y = dy + wobble_y

        if self._jerk_limit > 0.0:
            jx = out_x - self._prev_dx
            jy = out_y - self._prev_dy
            jmag = math.hypot(jx, jy)
            if jmag > self._jerk_limit:
                scale = self._jerk_limit / jmag
                out_x = self._prev_dx + jx * scale
                out_y = self._prev_dy + jy * scale

        if not (math.isfinite(out_x) and math.isfinite(out_y)):
            out_x, out_y = dx, dy

        self._prev_dx = out_x
        self._prev_dy = out_y
        return out_x, out_y
