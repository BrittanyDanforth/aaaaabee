"""Dt-aware target smoothing and prediction — responsive on moving bodies, stable on lock."""

from __future__ import annotations

import math
from dataclasses import dataclass

_MAX_VELOCITY = 3200.0
_MAX_DT = 0.12
_MIN_DT = 0.001
# Apex-tuned smoothing time constants. Tighter than the previous defaults so the
# aim point keeps up with strafing enemies — overall feel is snappier without
# inducing jitter (verified by moving_body_lag_bounded tests).
_TAU_POS_STILL = 0.042
_TAU_POS_MOVING = 0.018
_TAU_VEL = 0.032
_TAU_PRED_BLEND = 0.016
_MAX_PRED_LEAD_S = 0.040
_MAX_PRED_PX = 22.0
_MAX_UPWARD_LEAD_PX = 4.0
_BODY_Y_LO_FRAC = 0.28
_BODY_Y_HI_FRAC = 0.50
_body_y_lo_frac = _BODY_Y_LO_FRAC
_body_y_hi_frac = _BODY_Y_HI_FRAC
_tau_still = _TAU_POS_STILL
_tau_moving = _TAU_POS_MOVING
_max_upward_lead_px = _MAX_UPWARD_LEAD_PX
_BODY_X_MARGIN_FRAC = 0.18
# Velocity threshold (px/s) where smoothing slides from "still" tau to "moving" tau.
# Lower threshold = quicker response to small movements (peeking, strafing).
_SPEED_MOVING_PX_S = 60.0


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


def clamp_aim_to_display_fov(
    x: float,
    y: float,
    center_x: float,
    center_y: float,
    detect_fov: float,
    display_fov: float,
    *,
    scale: float = 0.96,
) -> tuple[float, float]:
    """Frame-space clamp shared by overlay dot and pull (inside visible FOV ring)."""
    if not (
        math.isfinite(x)
        and math.isfinite(y)
        and math.isfinite(center_x)
        and math.isfinite(center_y)
    ):
        return x, y
    fov_limit = max(1.0, min(float(detect_fov), float(display_fov))) * scale
    dx = x - center_x
    dy = y - center_y
    dist = math.hypot(dx, dy)
    if dist <= fov_limit or dist <= 0.0:
        return x, y
    s = fov_limit / dist
    return center_x + dx * s, center_y + dy * s


def fov_distance_scale(dist: float, fov_radius: float, edge_min_scale: float) -> float:
    if not math.isfinite(dist) or not math.isfinite(fov_radius) or fov_radius <= 0.0:
        return 1.0
    t = min(1.0, max(0.0, dist / fov_radius))
    floor = max(0.0, min(1.0, _finite(edge_min_scale, 0.0)))
    return floor + (1.0 - floor) * (1.0 - t)


@dataclass
class TargetMotion:
    """``x``/``y`` match ``overlay_xy()`` — pull moves crosshair toward the visible dot."""

    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    overlay_x: float | None = None
    overlay_y: float | None = None

    def overlay_xy(self) -> tuple[float, float]:
        if self.overlay_x is not None and self.overlay_y is not None:
            return self.overlay_x, self.overlay_y
        return self.x, self.y

    def predict(
        self,
        lead_seconds: float,
        max_lead_pixels: float,
    ) -> tuple[float, float]:
        if lead_seconds <= 0.0 or max_lead_pixels <= 0.0:
            return self.x, self.y
        ls = min(max(0.0, _finite(lead_seconds, 0.0)), 0.12)
        mlp = min(max(0.0, _finite(max_lead_pixels, 0.0)), 48.0)
        ox = _finite(self.vx, 0.0) * ls
        oy = _finite(self.vy, 0.0) * ls
        if oy < 0.0:
            oy = max(oy, -_max_upward_lead_px)
        lead_dist = math.hypot(ox, oy)
        if lead_dist <= 0.0 or not math.isfinite(lead_dist):
            return self.x, self.y
        if lead_dist > mlp:
            s = mlp / lead_dist
            ox *= s
            oy *= s
            if oy < 0.0:
                oy = max(oy, -_max_upward_lead_px)
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
        self._pull_x: float | None = None
        self._pull_y: float | None = None
        self._last_meas_x: float | None = None
        self._last_meas_y: float | None = None
        self._vx: float = 0.0
        self._vy: float = 0.0
        self._prediction_enabled: bool = True
        self._prediction_lead_s: float = _MAX_PRED_LEAD_S
        self._prediction_max_px: float = _MAX_PRED_PX
        self._body_bbox: tuple[int, int, int, int] | None = None
        # PHASE-7 AUDIT FIX (CRIT3): track the last stable body bbox so we
        # can refuse a single-frame fragment (e.g. the detector returns
        # just the helmet because the enemy crouched or slid behind
        # cover). When the new bbox jumps significantly upward OR shrinks
        # dramatically we treat it as a likely fragment and keep using
        # the previous stable bbox for the chest-band clamp for a few
        # frames, giving the detector time to re-acquire the full body.
        # Without this, a 40-px head fragment replacing a 140-px torso
        # bbox would jump the chest-band clamp upward by ~30 px in one
        # frame, manifesting as visible aim drift.
        self._last_stable_bbox: tuple[int, int, int, int] | None = None
        self._stable_bbox_hold_frames: int = 0
        self._aim_is_body_anchor: bool = True
        self._fov_cx: float | None = None
        self._fov_cy: float | None = None
        self._fov_radius: float | None = None
        self._last_pred_offset: tuple[float, float] = (0.0, 0.0)
        self._last_pre_predict: tuple[float, float] | None = None
        self._overlay_smooth: tuple[float, float] | None = None
        # M5 (audit): hysteresis state for the stationary jitter deadband.
        # Enter when meas_drift < 2.5 AND speed < 90; exit only when
        # meas_drift > 5.0 for 2 consecutive frames AND the instantaneous
        # velocity also exceeds the threshold (not just smoothed speed).
        self._in_deadband: bool = False
        self._deadband_exit_frames: int = 0

    def configure_prediction(
        self,
        enabled: bool,
        lead_seconds: float,
        max_pixels: float,
        *,
        vertical_cap_pixels: float | None = None,
    ) -> None:
        global _max_upward_lead_px
        self._prediction_enabled = bool(enabled)
        self._prediction_lead_s = max(0.0, float(lead_seconds))
        self._prediction_max_px = max(0.0, float(max_pixels))
        if vertical_cap_pixels is not None:
            _max_upward_lead_px = max(0.0, float(vertical_cap_pixels))

    def configure_body_clamp(self, y_min_fraction: float, y_max_fraction: float) -> None:
        global _body_y_lo_frac, _body_y_hi_frac
        lo = max(0.1, min(0.5, float(y_min_fraction)))
        hi = max(lo + 0.05, min(0.7, float(y_max_fraction)))
        _body_y_lo_frac = lo
        _body_y_hi_frac = hi

    def configure_fov_clamp(
        self,
        center_x: float,
        center_y: float,
        radius: float,
    ) -> None:
        self._fov_cx = float(center_x)
        self._fov_cy = float(center_y)
        self._fov_radius = max(1.0, float(radius))

    def _clamp_to_fov(self, x: float, y: float) -> tuple[float, float]:
        """Keep smoothed aim inside detection FOV circle."""
        if self._fov_cx is None or self._fov_cy is None or self._fov_radius is None:
            return x, y
        dx = x - self._fov_cx
        dy = y - self._fov_cy
        dist = math.hypot(dx, dy)
        r = self._fov_radius * 0.95
        if dist <= r or dist <= 0.0:
            return x, y
        s = r / dist
        return self._fov_cx + dx * s, self._fov_cy + dy * s

    def configure_smoothing_tau(self, still: float, moving: float) -> None:
        global _tau_still, _tau_moving
        _tau_still = max(0.01, float(still))
        _tau_moving = max(0.005, min(_tau_still, float(moving)))

    def smooth_overlay_point(
        self,
        x: float,
        y: float,
        *,
        alpha: float = 0.45,
    ) -> tuple[float, float]:
        """Mild EMA smoothing applied to the post-FOV-clamp overlay point.

        The overlay dot is clamped to the visible FOV ring (96 % radius). When
        the detected aim point oscillates around the ring boundary the dot can
        pop in and out — applying a light EMA here keeps the dot's visible
        position stable without affecting the underlying tracked motion.
        """
        if not (math.isfinite(x) and math.isfinite(y)):
            return x, y
        if self._overlay_smooth is None:
            self._overlay_smooth = (x, y)
            return x, y
        a = max(0.05, min(1.0, float(alpha)))
        ay = a
        if y < self._overlay_smooth[1]:
            ay = min(a, 0.48)
        sx = self._overlay_smooth[0] + a * (x - self._overlay_smooth[0])
        sy = self._overlay_smooth[1] + ay * (y - self._overlay_smooth[1])
        self._overlay_smooth = (sx, sy)
        return sx, sy

    def reset_overlay_smoothing(self) -> None:
        self._overlay_smooth = None

    def reset(self) -> None:
        self._last = None
        self._last_time = None
        self._smooth_x = None
        self._smooth_y = None
        self._pull_x = None
        self._pull_y = None
        self._last_meas_x = None
        self._last_meas_y = None
        self._vx = 0.0
        self._vy = 0.0
        self._body_bbox = None
        self._last_stable_bbox = None
        self._stable_bbox_hold_frames = 0
        self._aim_is_body_anchor = True
        self._fov_cx = None
        self._fov_cy = None
        self._fov_radius = None
        self._last_pred_offset = (0.0, 0.0)
        self._last_pre_predict = None
        self._overlay_smooth = None
        # M5 deadband memory must also reset when the lock is fully torn down.
        self._in_deadband = False
        self._deadband_exit_frames = 0

    def soft_reset(self) -> None:
        """
        PHASE-6 AUDIT FIX (D-HIGH5): partial reset that PRESERVES the
        last smoothed position and last measurement so the next aim
        sample is step-capped relative to where the dot already is.

        ``reset()`` clears ``_smooth_x/y`` and ``_last_meas_x/y`` to
        None, which means the next observation has no anchor — the
        step-cap branch in ``update()`` is bypassed and the dot
        teleports to the new measurement in a single frame.  When the
        runtime drops a target after the grace window, calling
        ``soft_reset()`` instead keeps the dot at the last known
        position; the velocity and motion-validated memory are
        cleared so the smoother does not fight a stale prediction,
        but the geometric anchor survives.
        """
        self._last = None
        self._last_time = None
        # Keep: _smooth_x/y, _pull_x/y, _last_meas_x/y
        self._vx = 0.0
        self._vy = 0.0
        self._body_bbox = None
        # PHASE-7 (CRIT3): soft_reset also clears the stable-bbox cache
        # because the lock has been released — no continuity assumption
        # holds between the old target and whatever new body the
        # next observation will refer to.
        self._last_stable_bbox = None
        self._stable_bbox_hold_frames = 0
        self._last_pred_offset = (0.0, 0.0)
        self._last_pre_predict = None
        self._in_deadband = False
        self._deadband_exit_frames = 0

    @staticmethod
    def _clamp_to_body_bbox(
        x: float,
        y: float,
        bbox_x: int,
        bbox_y: int,
        bbox_w: int,
        bbox_h: int,
    ) -> tuple[float, float]:
        """Keep smoothed aim inside upper-chest band — prevents sky/side drift."""
        mx = bbox_w * _BODY_X_MARGIN_FRAC
        y_lo = bbox_y + bbox_h * _body_y_lo_frac
        y_hi = bbox_y + bbox_h * _body_y_hi_frac
        x_lo = bbox_x + mx
        x_hi = bbox_x + bbox_w - mx
        return (
            max(x_lo, min(x_hi, x)),
            max(y_lo, min(y_hi, y)),
        )


    @staticmethod
    def _cap_measurement_step(
        x: float,
        y: float,
        last_x: float,
        last_y: float,
        bbox_h: int,
        dt: float,
    ) -> tuple[float, float]:
        """
        Limit per-frame detector jumps so overlay dot does not teleport across the
        screen, but allow generous travel proportional to bbox height so the
        smoother keeps up with strafing/sliding enemies. The floor is kept low
        (6 px) so slow strafes pass through without being clipped.
        """
        # M6 (audit): a 6-px floor lets 5-6 px snaps through on a locked
        # stationary target; the user perceives those as glitch. The caller
        # passes locked_slow=True when smoothed speed < 30 px/s AND the
        # tracker is locked so we tighten the floor to 2 px. Fast-moving /
        # unlocked / re-acquire stays at the 6 px floor (responsive snaps).
        max_step = max(6.0, min(34.0, bbox_h * 0.24)) * max(0.35, min(2.2, dt * 60.0))
        dx = x - last_x
        dy = y - last_y
        dist = math.hypot(dx, dy)
        if dist <= max_step or dist <= 0.0:
            return x, y
        s = max_step / dist
        nx = last_x + dx * s
        ny = last_y + dy * s
        up_cap = max(2.0, max_step * 0.42)
        if ny < last_y - up_cap:
            ny = last_y - up_cap
        return nx, ny

    @staticmethod
    def _cap_measurement_step_locked_slow(
        x: float,
        y: float,
        last_x: float,
        last_y: float,
        dt: float,
    ) -> tuple[float, float]:
        """Tighter 2-px floor used when the lock is stationary (M6 audit)."""
        max_step = 2.0 * max(0.35, min(2.2, dt * 60.0))
        dx = x - last_x
        dy = y - last_y
        dist = math.hypot(dx, dy)
        if dist <= max_step or dist <= 0.0:
            return x, y
        s = max_step / dist
        nx = last_x + dx * s
        ny = last_y + dy * s
        up_cap = max(1.5, max_step * 0.35)
        if ny < last_y - up_cap:
            ny = last_y - up_cap
        return nx, ny

    def _effective_tau(self, dt: float, speed: float) -> float:
        t = max(0.0, min(1.0, speed / _SPEED_MOVING_PX_S))
        return _tau_still + (_tau_moving - _tau_still) * t

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
        aim_is_body_anchor: bool = True,
    ) -> TargetMotion:
        """
        When aim_is_body_anchor=True (runtime default), x/y are detector aim_x/aim_y.
        Never re-blend toward bbox_y + 0.38*h. Clamp inside chest band before smooth.
        """
        self._aim_is_body_anchor = bool(aim_is_body_anchor)
        if (
            bbox_x is not None
            and bbox_y is not None
            and bbox_w is not None
            and bbox_h is not None
            and bbox_w > 0
            and bbox_h > 0
        ):
            bx, by, bw, bh = int(bbox_x), int(bbox_y), int(bbox_w), int(bbox_h)
            # PHASE-7 AUDIT FIX (CRIT3): stabilise the bbox used for the
            # chest-band clamp against single-frame fragment hits.
            # ``bx, by, bw, bh`` is the bbox the detector returned this
            # frame; ``cx, cy, cw, ch`` is what we ACTUALLY use for the
            # clamp.  When the new bbox jumps upward (top above the prior
            # stable top by more than 20% of stable height) OR shrinks
            # dramatically (h < 55% of stable h), we treat it as a likely
            # fragment and keep using the prior stable bbox for the next
            # few frames.  After ``_STABLE_BBOX_HOLD_MAX`` consecutive
            # fragment frames the detector has clearly committed to the
            # smaller bbox, so we accept it as the new baseline.
            cx, cy, cw, ch = bx, by, bw, bh
            _STABLE_BBOX_HOLD_MAX = 4
            prev = self._last_stable_bbox
            if prev is not None:
                pbx, pby, pbw, pbh = prev
                upward_jump = by < pby - pbh * 0.10
                shrunk = bh < pbh * 0.62
                if (upward_jump or shrunk) and self._stable_bbox_hold_frames < _STABLE_BBOX_HOLD_MAX:
                    cx, cy, cw, ch = pbx, pby, pbw, pbh
                    self._stable_bbox_hold_frames += 1
                else:
                    self._last_stable_bbox = (bx, by, bw, bh)
                    self._stable_bbox_hold_frames = 0
            else:
                self._last_stable_bbox = (bx, by, bw, bh)
                self._stable_bbox_hold_frames = 0
            self._body_bbox = (cx, cy, cw, ch)
            # Recompute bx/by/bw/bh to refer to the STABILISED bbox so
            # the rest of this method (clamp, column blend, last_meas
            # step cap) operates against the stable baseline.
            bx, by, bw, bh = cx, cy, cw, ch
            if self._aim_is_body_anchor:
                x, y = self._clamp_to_body_bbox(x, y, bx, by, bw, bh)
            else:
                col_x = bx + bw * 0.5
                col_y = by + bh * 0.38
                x = 0.35 * x + 0.65 * col_x
                y = 0.35 * y + 0.65 * col_y
            if self._last_meas_x is not None and self._last_meas_y is not None:
                dt_cap = 1.0 / 60.0
                if self._last_time is not None and time_sec > self._last_time:
                    dt_cap = min(0.12, time_sec - self._last_time)
                # M6 (audit): tighter 2-px step when the lock is already
                # in the stationary deadband. Outside the deadband the
                # 6-px floor preserves responsive snaps. Gating on the
                # deadband flag avoids pinning the smoother on a fresh
                # observation chain where smoothed velocity is briefly
                # zero by construction.
                meas_step = math.hypot(
                    x - self._last_meas_x, y - self._last_meas_y
                )
                lateral_rate = meas_step / max(_MIN_DT, dt_cap)
                # Strafing targets need the generous step cap even when the
                # smoother is in the stationary deadband (velocity briefly low).
                if self._in_deadband and lateral_rate < 22.0 and meas_step < 3.5:
                    x, y = self._cap_measurement_step_locked_slow(
                        x, y, self._last_meas_x, self._last_meas_y, dt_cap
                    )
                else:
                    x, y = self._cap_measurement_step(
                        x, y, self._last_meas_x, self._last_meas_y, bh, dt_cap
                    )
        else:
            self._body_bbox = None
        return self.observe(x, y, time_sec)

    @property
    def last_prediction_offset(self) -> tuple[float, float]:
        return self._last_pred_offset

    @property
    def last_pre_predict_point(self) -> tuple[float, float] | None:
        return self._last_pre_predict

    def _clamp_aim_output(self, x: float, y: float) -> tuple[float, float]:
        if self._body_bbox is not None:
            bx, by, bw, bh = self._body_bbox
            x, y = self._clamp_to_body_bbox(x, y, bx, by, bw, bh)
        if self._fov_radius is not None and self._fov_cx is not None and self._fov_cy is not None:
            x, y = self._clamp_to_fov(x, y)
            if self._body_bbox is not None:
                bx, by, bw, bh = self._body_bbox
                x, y = self._clamp_to_body_bbox(x, y, bx, by, bw, bh)
        return x, y

    def _finalize_pull_point(
        self,
        pre_x: float,
        pre_y: float,
        dt: float,
        in_deadband: bool,
    ) -> tuple[float, float]:
        use_inline_lead = (
            self._prediction_enabled
            and not (self._aim_is_body_anchor and self._body_bbox is not None)
            and not in_deadband
        )
        if use_inline_lead:
            lead_dt = min(dt, _MAX_PRED_LEAD_S)
            pred_x = pre_x + self._vx * lead_dt
            pred_y = pre_y + self._vy * lead_dt
            if pred_y < pre_y:
                pred_y = max(pred_y, pre_y - _MAX_UPWARD_LEAD_PX)
            if self._body_bbox is not None:
                bx, by, bw, bh = self._body_bbox
                mx = bw * _BODY_X_MARGIN_FRAC
                pred_x = max(bx + mx, min(bx + bw - mx, pred_x))
            pa = alpha_from_tau(dt, _TAU_PRED_BLEND)
            out_x = pre_x + pa * (pred_x - pre_x)
            out_y = pre_y + pa * (pred_y - pre_y)
        else:
            out_x = pre_x
            out_y = pre_y

        motion = TargetMotion(out_x, out_y, self._vx, self._vy)
        use_second_predict = (
            self._prediction_enabled
            and self._prediction_lead_s > 0.0
            and self._prediction_max_px > 0.0
            and not (self._aim_is_body_anchor and self._body_bbox is not None)
            and not in_deadband
        )
        if use_second_predict:
            px, py = motion.predict(self._prediction_lead_s, self._prediction_max_px)
            out_x, out_y = px, py
        return self._clamp_aim_output(out_x, out_y)

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
            self._pull_x = x
            self._pull_y = y
            self._last_meas_x = x
            self._last_meas_y = y
            self._last_time = time_sec
            ox, oy = self._clamp_aim_output(x, y)
            self._last = TargetMotion(x, y, 0.0, 0.0, overlay_x=ox, overlay_y=oy)
            return self._last

        if self._last_time is None:
            dt = 1.0 / 60.0
        else:
            dt = time_sec - self._last_time
        if dt < 0.0:
            dt = _MIN_DT
        dt = max(_MIN_DT, min(dt, _MAX_DT))

        if self._last_meas_x is not None and self._last_meas_y is not None:
            inst_vx = (x - self._last_meas_x) / dt
            inst_vy = (y - self._last_meas_y) / dt
            # Velocity ceiling scaled to bbox height — bigger (closer) targets move
            # more screen pixels per second, so cap accordingly. Raised from
            # 200 -> 600 px/s baseline to accommodate Apex strafing/slide speeds.
            cap_v = 360.0
            if self._body_bbox is not None:
                cap_v = max(120.0, min(800.0, self._body_bbox[3] * 4.5))
            ivmag = math.hypot(inst_vx, inst_vy)
            if ivmag > cap_v and ivmag > 0.0:
                s = cap_v / ivmag
                inst_vx *= s
                inst_vy *= s
            va = alpha_from_tau(dt, _TAU_VEL)
            self._vx = _finite(self._vx + va * (inst_vx - self._vx), 0.0)
            self._vy = _finite(self._vy + va * (inst_vy - self._vy), 0.0)
            if self._vy < 0.0:
                self._vy *= 0.55
            vmag = math.hypot(self._vx, self._vy)
            if vmag > _MAX_VELOCITY:
                s = _MAX_VELOCITY / vmag
                self._vx *= s
                self._vy *= s

        speed = math.hypot(self._vx, self._vy)
        tau = self._effective_tau(dt, speed)
        alpha = alpha_from_tau(dt, tau)

        # M5 (audit): hysteresis deadband. Enter when speed < 90 AND
        # meas_drift < 2.5; exit only when (meas_drift > 5.0 AND
        # instantaneous velocity > 30 px/s) for 2 consecutive frames.
        # The binary 2.0-px threshold previously oscillated against mask
        # noise (1.5-3 px). Hysteresis stabilises the dot — small drift
        # stays frozen, real movement triggers a clean exit.
        meas_drift = math.hypot(x - self._smooth_x, y - self._smooth_y)
        inst_speed = 0.0
        inst_vx = 0.0
        if self._last_meas_x is not None and self._last_meas_y is not None:
            inst_vx = (x - self._last_meas_x) / max(_MIN_DT, dt)
            iy = (y - self._last_meas_y) / max(_MIN_DT, dt)
            inst_speed = math.hypot(inst_vx, iy)
        if self._in_deadband:
            exits_now = (
                (meas_drift > 5.0 and inst_speed > 30.0)
                or (meas_drift > 2.5 and inst_speed > 50.0)
                or (abs(inst_vx) > 38.0 and meas_drift > 2.8)
            )
            if exits_now:
                self._deadband_exit_frames += 1
                if self._deadband_exit_frames >= 2:
                    self._in_deadband = False
                    self._deadband_exit_frames = 0
            else:
                self._deadband_exit_frames = 0
        else:
            if speed < 90.0 and meas_drift < 2.5:
                self._in_deadband = True
                self._deadband_exit_frames = 0
        in_deadband = self._in_deadband
        if self._pull_x is None or self._pull_y is None:
            self._pull_x = self._smooth_x
            self._pull_y = self._smooth_y
        if not in_deadband:
            self._smooth_x = self._smooth_x + alpha * (x - self._smooth_x)
            self._smooth_y = self._smooth_y + alpha * (y - self._smooth_y)
            self._pull_x = self._pull_x + alpha * (x - self._pull_x)
            self._pull_y = self._pull_y + alpha * (y - self._pull_y)
        else:
            # Overlay: gentle follow in deadband so Basic "Smoothness" (tau_still)
            # still damps detector 1–3 px noise without freeing the pull anchor.
            ov_cap = max(0.025, min(0.085, 0.12 - (_tau_still - 0.02) * 1.2))
            ov_alpha = min(alpha_from_tau(dt, max(_tau_still * 2.2, 0.036)), ov_cap)
            nx = self._smooth_x + ov_alpha * (x - self._smooth_x)
            ny = self._smooth_y + ov_alpha * (y - self._smooth_y)
            # Do not creep the overlay anchor upward on fragment hits (sky steal).
            if y < self._smooth_y - 6.0:
                ny = self._smooth_y
            self._smooth_x, self._smooth_y = nx, ny
            self._pull_x = self._smooth_x
            self._pull_y = self._smooth_y

        pre_pull_x, pre_pull_y = self._pull_x, self._pull_y
        self._last_pre_predict = (pre_pull_x, pre_pull_y)

        pull_x, pull_y = self._finalize_pull_point(pre_pull_x, pre_pull_y, dt, in_deadband)
        overlay_x, overlay_y = self._clamp_aim_output(self._smooth_x, self._smooth_y)
        # Pull assists toward the visible dot anchor, not a faster internal point.
        pull_x, pull_y = overlay_x, overlay_y

        self._last_meas_x = x
        self._last_meas_y = y
        self._last_time = time_sec
        self._last_pred_offset = (pull_x - pre_pull_x, pull_y - pre_pull_y)
        self._last = TargetMotion(
            pull_x,
            pull_y,
            self._vx,
            self._vy,
            overlay_x=overlay_x,
            overlay_y=overlay_y,
        )
        return self._last


class HumanizedMotion:
    """Bounded micro-variation — disable (amplitude=0) for tracking tests."""

    def __init__(self, amplitude: float, jerk_limit: float) -> None:
        self._amplitude = max(0.0, min(amplitude, 2.0))
        self._jerk_limit = max(0.0, jerk_limit)
        self._prev_wobble_x = 0.0
        self._prev_wobble_y = 0.0
        self._frame = 0

    def reset(self) -> None:
        self._prev_wobble_x = 0.0
        self._prev_wobble_y = 0.0
        self._frame = 0

    def apply(self, dx: float, dy: float) -> tuple[float, float]:
        dx = _finite(dx, 0.0)
        dy = _finite(dy, 0.0)
        if self._amplitude <= 0.0 and self._jerk_limit <= 0.0:
            return dx, dy

        self._frame += 1
        mag = math.hypot(dx, dy)
        wobble_scale = min(0.35, mag / max(self._amplitude * 6.0, 1.0))
        if mag < 2.0:
            wobble_scale = 0.0
        n = self._frame
        wobble_x = self._amplitude * wobble_scale * math.sin(n * 0.73) * math.cos(n * 0.19)
        wobble_y = self._amplitude * wobble_scale * math.cos(n * 0.61) * math.sin(n * 0.23)

        if self._jerk_limit > 0.0:
            jx = wobble_x - self._prev_wobble_x
            jy = wobble_y - self._prev_wobble_y
            jmag = math.hypot(jx, jy)
            if jmag > self._jerk_limit:
                scale = self._jerk_limit / jmag
                wobble_x = self._prev_wobble_x + jx * scale
                wobble_y = self._prev_wobble_y + jy * scale

        self._prev_wobble_x = wobble_x
        self._prev_wobble_y = wobble_y
        out_x = dx + wobble_x
        out_y = dy + wobble_y
        if not (math.isfinite(out_x) and math.isfinite(out_y)):
            return dx, dy
        return out_x, out_y


_RECOIL_RAMP_TAU_S = 0.22


class RecoilCompensator:
    """
    Engagement-gated recoil-helper bias.

    While the caller signals ``is_firing=True``:
      * a ramped downward Y bias (``pull_down_px_per_s``) eases in over
        ~``_RECOIL_RAMP_TAU_S`` so sustained fire feels smooth, not a snap
      * horizontal ``jitter_*`` keys drive lateral *stabilization*: an extra
        pull toward reducing ``err_x`` (passed from ``PullController``),
        capped by ``jitter_amplitude_px`` and scaled by ``jitter_frequency_hz``
        as response rate — not a sinusoidal shake

    Bias is *additive* on top of the pull velocity — it is NOT fed back into
    the velocity smoother. That's important: it would otherwise leak into the
    EMA state and the cursor would keep drifting downward for several frames
    after the user stops firing.

    When ``is_firing=False`` the ramp and firing timer reset so the next burst
    starts from zero pull-down and zero lateral correction (on-target err_x=0).
    """

    def __init__(
        self,
        *,
        recoil_enabled: bool,
        pull_down_px_per_s: float,
        jitter_enabled: bool,
        jitter_amplitude_px: float,
        jitter_frequency_hz: float,
    ) -> None:
        self._recoil_enabled = bool(recoil_enabled)
        self._pull_down = max(0.0, min(_finite(pull_down_px_per_s, 0.0), 180.0))
        self._jitter_enabled = bool(jitter_enabled)
        self._jitter_amp = max(0.0, min(_finite(jitter_amplitude_px, 0.0), 6.0))
        self._jitter_hz = max(0.0, min(_finite(jitter_frequency_hz, 6.0), 20.0))
        self._fire_duration = 0.0
        self._was_firing = False

    @property
    def active(self) -> bool:
        """True iff any compensation channel would emit a non-zero bias."""
        recoil_on = self._recoil_enabled and self._pull_down > 0.0
        jitter_on = self._jitter_enabled and self._jitter_amp > 0.0 and self._jitter_hz > 0.0
        return recoil_on or jitter_on

    def reset(self) -> None:
        self._fire_duration = 0.0
        self._was_firing = False

    def compute_bias(
        self,
        *,
        is_firing: bool,
        dt: float,
        err_x: float = 0.0,
    ) -> tuple[float, float]:
        """
        Returns (bias_x, bias_y) in pixels for this frame.

        - bias_y is positive-down (matches the screen-coordinate convention
          used by ``compute_delta``).
        - bias_x stabilizes lateral aim by correcting toward ``err_x → 0``;
          on-target (``err_x ≈ 0``) produces no horizontal bias.
        """
        if not is_firing:
            if self._was_firing:
                self._fire_duration = 0.0
            self._was_firing = False
            return 0.0, 0.0

        dt = _finite(dt, 0.0)
        if dt <= 0.0 or dt > 0.5:
            dt = 1.0 / 60.0
        self._was_firing = True
        self._fire_duration += dt

        bias_y = 0.0
        if self._recoil_enabled and self._pull_down > 0.0:
            ramp = 1.0 - math.exp(-self._fire_duration / _RECOIL_RAMP_TAU_S)
            bias_y = self._pull_down * dt * ramp

        bias_x = 0.0
        if self._jitter_enabled and self._jitter_amp > 0.0 and self._jitter_hz > 0.0:
            ex = _finite(err_x, 0.0)
            # Higher jitter_frequency_hz → snappier lateral hold (legacy key).
            alpha = 1.0 - math.exp(-self._jitter_hz * dt * 0.35)
            correct = ex * alpha
            bias_x = max(-self._jitter_amp, min(self._jitter_amp, correct))

        if not (math.isfinite(bias_x) and math.isfinite(bias_y)):
            return 0.0, 0.0
        return bias_x, bias_y
