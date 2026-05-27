"""
Canonical detection + aim wiring for tests and artifact scripts.

Production live play uses ``AssistRuntime`` in ``runtime.py`` (same lock module).

This module runs ``find_best_target`` + ``apply_target_lock`` + motion with
the same stale-freeze rules as ``AssistRuntime._smooth_aim``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import detector
from detector import DetectionResult, Target
from motion import TargetMotion, TargetTracker
from profiles import (
    effective_detection_fov_radius,
    effective_fov_radius,
    effective_overlay_fov_radius,
    is_yolo_detection,
)
from target_lock import (
    TargetLockState,
    apply_target_lock,
    apply_yolo_target_lock,
    detection_sticky_context,
    may_assist_pull_target,
    viewmodel_exclude_bottom,
)


@dataclass
class AimState:
    """Frame result after production-equivalent detect + lock + motion."""

    aim_x: float
    aim_y: float
    vx: float
    vy: float
    active: bool
    overlay_x: float = 0.0
    overlay_y: float = 0.0
    pull_x: float = 0.0
    pull_y: float = 0.0
    is_stale: bool = False
    inside_body_overlay: bool = False
    inside_body_pull: bool = False
    target: Target | None = None
    detection: DetectionResult | None = None
    debug_lines: list[str] = field(default_factory=list)
    bbox_used: tuple[int, int, int, int] | None = None
    wired_observe_target: bool = True
    observe_called: bool = False


def _ring_clamp_frame(
    ox: float,
    oy: float,
    cx: float,
    cy: float,
    fov_radius: float,
) -> tuple[float, float]:
    """Same 96% FOV ring clamp as ``AssistRuntime._frame_overlay_point`` (frame space)."""
    dx = ox - cx
    dy = oy - cy
    dist = math.hypot(dx, dy)
    lim = max(1.0, float(fov_radius)) * 0.96
    if dist > lim and dist > 0.0:
        s = lim / dist
        return cx + dx * s, cy + dy * s
    return ox, oy


def resolve_runtime_fov(
    config: dict[str, Any], *, ads_active: bool
) -> tuple[int, int]:
    """Same FOV pair as ``AssistRuntime`` main loop (unified by default)."""
    user_fov = int(effective_fov_radius(config, ads_active=ads_active))
    detect_fov = int(
        effective_detection_fov_radius(config, ads_active=ads_active)
    )
    if bool(config.get("unified_fov", True)) and not is_yolo_detection(config):
        detect_fov = user_fov
    ring_inner = int(float(user_fov) * 0.96)
    config["_runtime_fov"] = user_fov
    config["_runtime_detect_fov"] = float(detect_fov)
    config["_runtime_overlay_fov"] = float(ring_inner)
    return user_fov, detect_fov


def _motion_body_bbox(tracker: TargetTracker) -> tuple[int, int, int, int] | None:
    """Chest-band clamp bbox (stable hold), same as pull_trace ``motion_body_bbox``."""
    bb = tracker._body_bbox
    if bb is not None and len(bb) == 4 and bb[2] > 0 and bb[3] > 0:
        return int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
    return None


def _inside_body(
    t: Target,
    x: float,
    y: float,
    *,
    tracker: TargetTracker | None = None,
) -> bool:
    bb = _motion_body_bbox(tracker) if tracker is not None else None
    if bb is not None:
        bx, by, bw, bh = bb
    else:
        bx, by, bw, bh = t.bbox_x, t.bbox_y, t.bbox_w, t.bbox_h
    return TargetTracker.point_inside_body_bbox(x, y, bx, by, bw, bh)


class TargetingRuntime:
    """
    End-to-end targeting session for tests/artifacts.

    Uses the same frame lock + stale motion freeze as ``AssistRuntime``.
    """

    # Visible-bbox smoothing alphas.  alpha_pos governs how quickly the
    # displayed bbox follows the detector's centroid (higher = snappier);
    # alpha_size governs width/height blending (lower = less visible
    # stretch frame-to-frame).  These only smooth the *displayed* bbox
    # (and the bbox passed to TargetTracker.observe_target); the
    # underlying detector candidates and lock-state targets remain
    # untouched so scoring / refine gates see the raw detection.
    _BBOX_ALPHA_POS = 0.65
    _BBOX_ALPHA_SIZE = 0.50
    # IoU below this between consecutive detections is treated as an
    # identity break and the smoother resets (snap to new bbox).  This
    # mirrors how the user perceives a "fresh lock" — a totally new
    # bbox in a different place should not gradually drift in from the
    # previous lock location.
    _BBOX_SMOOTH_RESET_IOU = 0.10

    def __init__(self) -> None:
        self.tracker = TargetTracker()
        self._lock_state = TargetLockState()
        self._last_observe_bbox: tuple[int, int, int, int] | None = None
        self._smoothed_bbox: tuple[float, float, float, float] | None = None
        self._observe_target_calls: int = 0
        self._detect_ctx = detector.DetectionContext()
        self._yolo_engine = None

    def reset(self) -> None:
        self.tracker.reset()
        self._lock_state.reset()
        self._last_observe_bbox = None
        self._smoothed_bbox = None
        self._observe_target_calls = 0
        self._detect_ctx.reset()

    @staticmethod
    def _bbox_iou(
        a: tuple[float, float, float, float],
        b: tuple[float, float, float, float],
    ) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x0 = max(ax, bx)
        y0 = max(ay, by)
        x1 = min(ax + aw, bx + bw)
        y1 = min(ay + ah, by + bh)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        inter = (x1 - x0) * (y1 - y0)
        union = aw * ah + bw * bh - inter
        return inter / max(union, 1.0)

    def _smooth_visible_bbox(
        self, new_bb: tuple[int, int, int, int]
    ) -> tuple[int, int, int, int]:
        nx, ny, nw, nh = new_bb
        if self._smoothed_bbox is None or self._BBOX_ALPHA_POS >= 1.0:
            self._smoothed_bbox = (float(nx), float(ny), float(nw), float(nh))
            return new_bb
        prev = self._smoothed_bbox
        if self._bbox_iou(prev, (nx, ny, nw, nh)) < self._BBOX_SMOOTH_RESET_IOU:
            # Identity break — snap so we don't visually slide in from
            # the previous (unrelated) lock's location.
            self._smoothed_bbox = (float(nx), float(ny), float(nw), float(nh))
            return new_bb
        ap = self._BBOX_ALPHA_POS
        asz = self._BBOX_ALPHA_SIZE
        sx = ap * nx + (1.0 - ap) * prev[0]
        sy = ap * ny + (1.0 - ap) * prev[1]
        sw = asz * nw + (1.0 - asz) * prev[2]
        sh = asz * nh + (1.0 - asz) * prev[3]
        self._smoothed_bbox = (sx, sy, sw, sh)
        # Clamp minimum size 6 so a fully-collapsed box doesn't render.
        return (
            int(round(sx)),
            int(round(sy)),
            max(6, int(round(sw))),
            max(6, int(round(sh))),
        )

    @property
    def last_observe_bbox(self) -> tuple[int, int, int, int] | None:
        return self._last_observe_bbox

    @property
    def observe_target_call_count(self) -> int:
        return self._observe_target_calls

    @property
    def lock_state(self) -> TargetLockState:
        return self._lock_state

    def _apply_motion_config(self, config: dict[str, Any], cx: float, cy: float) -> None:
        fov_r = float(
            config.get("_runtime_overlay_fov")
            or config.get("_runtime_detect_fov")
            or config.get("_runtime_fov")
            or 200
        )
        self.tracker.configure_fov_clamp(cx, cy, fov_r)
        self.tracker.configure_prediction(
            bool(config.get("prediction_enabled", True)),
            float(config.get("prediction_lead_seconds", 0.03)),
            float(config.get("prediction_max_pixels", 20)),
            vertical_cap_pixels=float(config.get("prediction_vertical_cap_pixels", 4.0)),
        )
        self.tracker.configure_body_clamp(
            float(config.get("aim_body_y_min_fraction", 0.28)),
            float(config.get("aim_body_y_max_fraction", 0.52)),
        )
        self.tracker.configure_smoothing_tau(
            float(config.get("smoothing_tau_still", 0.062)),
            float(config.get("smoothing_tau_moving", 0.028)),
        )
        self.tracker.configure_overlay_dot_alpha(
            float(config.get("overlay_dot_smooth_alpha", 0.52))
        )

    def process_frame(
        self,
        frame_bgr: Any,
        config: dict[str, Any],
        *,
        time_sec: float | None = None,
        debug: bool = False,
    ) -> AimState:
        h, w = frame_bgr.shape[:2]
        cx = float(config.get("fov_center_x", w / 2.0))
        cy = float(config.get("fov_center_y", h / 2.0))
        ads_active = bool(config.get("_ads_active", True))
        _user_fov, fov = resolve_runtime_fov(config, ads_active=ads_active)
        min_area = float(
            config.get("min_target_area_pixels")
            or config.get("min_target_area")
            or 40.0
        )
        config.setdefault("target_lost_frames_before_unlock", 18)
        config.setdefault("viewmodel_exclude_bottom_frac", 0.28)
        self._apply_motion_config(config, cx, cy)

        sticky, currently_locked, _ = detection_sticky_context(self._lock_state, config)
        det_mode = str(config.get("detection_mode", "apex")).strip().lower()
        yolo_engine = None
        if det_mode == "yolo":
            from yolo_detector import get_yolo_engine

            yolo_engine = get_yolo_engine(config)
            self._yolo_engine = yolo_engine
        if det_mode == "yolo":
            from yolo_detector import find_best_yolo_target

            if yolo_engine is None:
                raw = DetectionResult(
                    None,
                    0,
                    0.0,
                    debug_lines=["yolo_mode but engine not loaded"],
                    active=False,
                )
            else:
                raw = find_best_yolo_target(
                    frame_bgr,
                    fov,
                    min_area,
                    cx,
                    cy,
                    engine=yolo_engine,
                    sticky_target=sticky,
                    stickiness_pixels=float(
                        config.get("target_stickiness_pixels", 90.0)
                    ),
                    min_height_px=float(config.get("humanoid_min_height_pixels", 16)),
                    min_confidence=float(config.get("yolo_confidence_min", 0.5)),
                    currently_locked=currently_locked,
                    ads_active=ads_active,
                    debug=debug,
                )
        else:
            raw = detector.find_best_target(
                frame_bgr,
                config.get("hsv_ranges"),
                fov,
                min_area,
                cx,
                cy,
                sticky_target=sticky,
                stickiness_pixels=float(config.get("target_stickiness_pixels", 90.0)),
                distance_weight=float(config.get("distance_score_weight", 1.0)),
                area_weight=float(config.get("area_score_weight", 0.5)),
                currently_locked=currently_locked,
                exclude_bottom_frac=viewmodel_exclude_bottom(config),
                min_height_px=float(config.get("humanoid_min_height_pixels", 16)),
                min_aspect=float(config.get("humanoid_min_aspect", 1.2)),
                max_aspect=float(config.get("humanoid_max_aspect", 4.5)),
                min_solidity=float(config.get("humanoid_min_solidity", 0.25)),
                torso_aim_fraction=float(config.get("torso_aim_fraction", 0.38)),
                body_shape_min_score=float(config.get("body_shape_min_score", 0.40)),
                detection_mode=det_mode,
                context=self._detect_ctx,
                debug=debug,
                display_fov_radius=float(effective_overlay_fov_radius(config)),
                yolo_engine=None,
                ads_active=ads_active,
            )
        if det_mode == "yolo" and bool(config.get("yolo_apex_nearest_lock", True)):
            result, is_stale = apply_yolo_target_lock(
                self._lock_state,
                raw,
                cfg=config,
                on_lock_expired=self.tracker.soft_reset,
            )
        else:
            result, is_stale = apply_target_lock(
                self._lock_state,
                raw,
                center_y=cy,
                cfg=config,
                on_lock_expired=self.tracker.soft_reset,
                fov_cx=cx,
                fov_cy=cy,
                frame_size=(w, h),
            )

        tsec = time.perf_counter() if time_sec is None else time_sec
        detection_fresh = (
            bool(result.active)
            and result.target is not None
            and not is_stale
        )

        if result.target is None:
            self.tracker.reset()
            return AimState(
                aim_x=cx,
                aim_y=cy,
                vx=0.0,
                vy=0.0,
                active=False,
                target=None,
                detection=result,
                debug_lines=list(result.debug_lines),
                bbox_used=None,
                observe_called=False,
            )

        t = result.target
        motion: TargetMotion | None
        observe_called = False
        if is_stale:
            motion = self.tracker._last
            if motion is None:
                return AimState(
                    aim_x=cx,
                    aim_y=cy,
                    vx=0.0,
                    vy=0.0,
                    active=False,
                    is_stale=True,
                    target=t,
                    detection=result,
                    debug_lines=list(result.debug_lines),
                    bbox_used=None,
                    observe_called=False,
                )
        else:
            if det_mode == "yolo" and bool(config.get("yolo_skip_motion_smooth", True)):
                motion = TargetMotion(
                    t.centroid_x,
                    t.centroid_y,
                    0.0,
                    0.0,
                    overlay_x=t.centroid_x,
                    overlay_y=t.centroid_y,
                )
                self._last_observe_bbox = (t.bbox_x, t.bbox_y, t.bbox_w, t.bbox_h)
            else:
                self._observe_target_calls += 1
                observe_called = True
                motion = self.tracker.observe_target(
                    t.centroid_x,
                    t.centroid_y,
                    tsec,
                    bbox_x=t.bbox_x,
                    bbox_y=t.bbox_y,
                    bbox_w=t.bbox_w,
                    bbox_h=t.bbox_h,
                    aim_is_body_anchor=True,
                )
                tracker_bbox = self.tracker._body_bbox
                if tracker_bbox is not None:
                    bx_t, by_t, bw_t, bh_t = tracker_bbox
                    self._last_observe_bbox = (
                        int(round(bx_t)),
                        int(round(by_t)),
                        int(round(bw_t)),
                        int(round(bh_t)),
                    )
                else:
                    self._last_observe_bbox = (
                        t.bbox_x,
                        t.bbox_y,
                        t.bbox_w,
                        t.bbox_h,
                    )

        ox, oy = motion.overlay_xy()
        px, py = _ring_clamp_frame(ox, oy, cx, cy, float(fov))
        inside_o = _inside_body(t, ox, oy, tracker=self.tracker)
        inside_p = _inside_body(t, px, py, tracker=self.tracker)

        return AimState(
            aim_x=motion.x,
            aim_y=motion.y,
            vx=motion.vx,
            vy=motion.vy,
            active=detection_fresh,
            overlay_x=ox,
            overlay_y=oy,
            pull_x=px,
            pull_y=py,
            is_stale=is_stale,
            inside_body_overlay=inside_o,
            inside_body_pull=inside_p,
            target=t,
            detection=result,
            debug_lines=list(result.debug_lines),
            bbox_used=self._last_observe_bbox,
            wired_observe_target=True,
            observe_called=observe_called,
        )


def process_frame(
    frame_bgr: Any,
    config: dict[str, Any],
    runtime: TargetingRuntime,
    *,
    time_sec: float | None = None,
) -> AimState:
    """Functional wrapper used by artifact scripts."""
    return runtime.process_frame(frame_bgr, config, time_sec=time_sec)
