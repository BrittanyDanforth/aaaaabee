"""Dt-aware target smoothing and prediction — responsive on moving bodies, stable on lock."""

from __future__ import annotations

import math
from dataclasses import dataclass

_MAX_VELOCITY = 3200.0
_MAX_DT = 0.12
_MIN_DT = 0.001
_TAU_POS_STILL = 0.055
_TAU_POS_MOVING = 0.022
_TAU_VEL = 0.040
_TAU_PRED_BLEND = 0.018
_MAX_PRED_LEAD_S = 0.045
_MAX_PRED_PX = 28.0
_SPEED_MOVING_PX_S = 85.0


def _finite(v: float, fallback: float = 0.0) -> float:
    return v if math.isfinite(v) else fallback


def alpha_from_tau(dt: float, tau: float) -> float:
    if tau <= 0.0:
        return 1.0
    dt = max(_MIN_DT, min(dt, _MAX_DT))
    return 1.0 - math.exp(-dt / tau)


def apply_smoothing_curve(t: float, curve: str) -> float:
    if not math.isfinite(t):
        return 0.0
    t = max(0.0, min(1.0, t))
    if curve == "ease_out":
        return 1.0 - (1.0 - t) ** 2
    if curve == "ease_in_out":
        return t * t * (3.0 - 2.0 * t)
    return t


def fov_distance_scale(dist: float, fov_radius: float, edge_min_scale: float) -> float:
    if not math.isfinite(dist) or not math.isfinite(fov_radius) or fov_radius <= 0.0:
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
        ls = min(_finite(lead_seconds, 0.0), _MAX_PRED_LEAD_S)
        mlp = min(_finite(max_lead_pixels, 0.0), _MAX_PRED_PX)
        ox = _finite(self.vx, 0.0) * ls
        oy = _finite(self.vy, 0.0) * ls
        lead_dist = math.hypot(ox, oy)
        if lead_dist <= 0.0 or not math.isfinite(lead_dist):
            return self.x, self.y
        if lead_dist > mlp:
            s = mlp / lead_dist
            ox *= s
            oy *= s
        px = _finite(self.x, 0.0) + ox
        py = _finite(self.y, 0.0) + oy
        if math.isfinite(px) and math.isfinite(py):
            return px, py
        return self.x, self.y


class TargetTracker:
    """Smooth aim anchor with frame-time normalization (30–144 FPS stable)."""

    def __init__(self) -> None:
        self._last: TargetMotion | None = None
        self._last_time: float | None = None
        self._smooth_x: float | None = None
        self._smooth_y: float | None = None
        self._last_meas_x: float | None = None
        self._last_meas_y: float | None = None
        self._vx: float = 0.0
        self._vy: float = 0.0

    def reset(self) -> None:
        self._last = None
        self._last_time = None
        self._smooth_x = None
        self._smooth_y = None
        self._last_meas_x = None
        self._last_meas_y = None
        self._vx = 0.0
        self._vy = 0.0

    def _effective_tau(self, dt: float, speed: float) -> float:
        t = max(0.0, min(1.0, speed / _SPEED_MOVING_PX_S))
        return _TAU_POS_STILL + (_TAU_POS_MOVING - _TAU_POS_STILL) * t

    def observe_target(
        self,
        x: float,
        y: float,
        time_sec: float,
        *,
        bbox_x: int | None = None,
        bbox_y: int | None = None,
        bbox_w: int | None = None,
        bbox_h: int | None = None,
    ) -> TargetMotion:
        """Prefer bbox upper-chest column; raw centroid weighted lightly to avoid plate hopping."""
        if (
            bbox_x is not None
            and bbox_y is not None
            and bbox_w is not None
            and bbox_h is not None
            and bbox_w > 0
            and bbox_h > 0
        ):
            col_x = bbox_x + bbox_w * 0.5
            col_y = bbox_y + bbox_h * 0.38
            x = 0.35 * x + 0.65 * col_x
            y = 0.35 * y + 0.65 * col_y
        return self.observe(x, y, time_sec)

    def observe(self, x: float, y: float, time_sec: float) -> TargetMotion:
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(time_sec)):
            if self._last is not None:
                return self._last
            return TargetMotion(
                _finite(x, 0.0),
                _finite(y, 0.0),
                0.0,
                0.0,
            )

        if self._smooth_x is None or self._smooth_y is None:
            self._smooth_x = x
            self._smooth_y = y
            self._last_meas_x = x
            self._last_meas_y = y
            self._last_time = time_sec
            self._last = TargetMotion(x, y, 0.0, 0.0)
            return self._last

        dt = time_sec - (self._last_time or time_sec)
        if dt < 0.0:
            dt = _MIN_DT
        dt = max(_MIN_DT, min(dt, _MAX_DT))

        if self._last_meas_x is not None and self._last_meas_y is not None:
            inst_vx = (x - self._last_meas_x) / dt
            inst_vy = (y - self._last_meas_y) / dt
            va = alpha_from_tau(dt, _TAU_VEL)
            self._vx = _finite(self._vx + va * (inst_vx - self._vx), 0.0)
            self._vy = _finite(self._vy + va * (inst_vy - self._vy), 0.0)
            vmag = math.hypot(self._vx, self._vy)
            if vmag > _MAX_VELOCITY:
                s = _MAX_VELOCITY / vmag
                self._vx *= s
                self._vy *= s

        speed = math.hypot(self._vx, self._vy)
        tau = self._effective_tau(dt, speed)
        alpha = alpha_from_tau(dt, tau)

        self._smooth_x = self._smooth_x + alpha * (x - self._smooth_x)
        self._smooth_y = self._smooth_y + alpha * (y - self._smooth_y)

        pred_x = self._smooth_x + self._vx * min(dt, _MAX_PRED_LEAD_S)
        pred_y = self._smooth_y + self._vy * min(dt, _MAX_PRED_LEAD_S)
        pa = alpha_from_tau(dt, _TAU_PRED_BLEND)
        out_x = self._smooth_x + pa * (pred_x - self._smooth_x)
        out_y = self._smooth_y + pa * (pred_y - self._smooth_y)

        self._last_meas_x = x
        self._last_meas_y = y
        self._last_time = time_sec
        self._last = TargetMotion(out_x, out_y, self._vx, self._vy)
        return self._last


class HumanizedMotion:
    """Bounded micro-variation — disable (amplitude=0) for tracking tests."""

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
