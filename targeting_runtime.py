"""
TEST / SCRIPT HARNESS — not production.

``AssistRuntime`` in ``runtime.py`` owns capture, overlay Tk, mouse gate, pull/PID,
and ``sync_config_subsystems``. This module replays detect→lock→motion on a single
frame for unit tests and audit scripts.

Guarantees (when using shared ``targeting_shared`` helpers):
  - Same CV ``find_best_target`` kwargs as production
  - Same 96% ring clamp: min(detect_fov, display_fov)
  - Same ``apply_target_lock`` / ``yolo_detect_and_lock`` entry points

Does NOT guarantee: mss capture crop, Tk overlay hold-last, mouse gate, Apex PID
subticks, idle lock ticks without a frame, or full ``sync_config_subsystems`` cleanup.
See ``docs/TARGETING_RUNTIME_HARNESS.md``.
"""

from __future__ import annotations

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
    detection_sticky_context,
    locked_target_may_refresh_motion_memory,
    may_assist_pull_target,
    viewmodel_exclude_bottom,
)
from targeting_shared import cv_find_best_target_from_config, ring_clamp_frame_point


@dataclass
class AimState:
    """Frame result after detect + lock + motion (harness DTO)."""

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
    may_assist_pull: bool = False
    show_for_overlay: bool = False


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
    overlay_fov = int(effective_overlay_fov_radius(config, ads_active=ads_active))
    ring_inner = float(overlay_fov) * 0.96
    config["_runtime_fov"] = user_fov
    config["_runtime_detect_fov"] = float(detect_fov)
    config["_runtime_overlay_fov"] = ring_inner
    return user_fov, detect_fov


def _motion_body_bbox(tracker: TargetTracker) -> tuple[int, int, int, int] | None:
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
    """Offline detect→lock→motion session for tests and artifact scripts."""

    def __init__(self) -> None:
        self.tracker = TargetTracker()
        self._lock_state = TargetLockState()
        self._last_observe_bbox: tuple[int, int, int, int] | None = None
        self._observe_target_calls: int = 0
        self._detect_ctx: detector.DetectionContext | None = detector.DetectionContext()
        self._yolo_engine = None

    def reset(self) -> None:
        self.tracker.reset()
        self._lock_state.reset()
        self._last_observe_bbox = None
        self._observe_target_calls = 0
        if self._detect_ctx is not None:
            self._detect_ctx.reset()
        self._yolo_engine = None

    @property
    def last_observe_bbox(self) -> tuple[int, int, int, int] | None:
        return self._last_observe_bbox

    @property
    def observe_target_call_count(self) -> int:
        return self._observe_target_calls

    @property
    def lock_state(self) -> TargetLockState:
        return self._lock_state

    def _ensure_detect_ctx(self, config: dict[str, Any]) -> None:
        det_mode = str(config.get("detection_mode", "apex")).strip().lower()
        if det_mode == "yolo":
            if self._detect_ctx is not None:
                self._detect_ctx.reset()
            self._detect_ctx = None
            return
        if self._detect_ctx is None:
            self._detect_ctx = detector.DetectionContext(
                motion_assist=bool(config.get("detection_motion_assist", True)),
                motion_threshold=int(config.get("detection_motion_threshold", 10)),
            )
        else:
            self._detect_ctx.motion_assist = bool(
                config.get("detection_motion_assist", True)
            )
            self._detect_ctx.motion_threshold = int(
                config.get("detection_motion_threshold", 10)
            )

    def _apply_motion_config(self, config: dict[str, Any], cx: float, cy: float) -> None:
        ads_active = bool(config.get("_ads_active", True))
        display_fov = float(
            effective_overlay_fov_radius(config, ads_active=ads_active)
        )
        self.tracker.configure_fov_clamp(cx, cy, display_fov)
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
        user_fov, detect_fov = resolve_runtime_fov(config, ads_active=ads_active)
        display_fov = float(
            config.get("_runtime_overlay_fov") or effective_overlay_fov_radius(config)
        )
        min_area = float(
            config.get("min_target_area_pixels")
            or config.get("min_target_area")
            or 40.0
        )
        config.setdefault("target_lost_frames_before_unlock", 18)
        config.setdefault("viewmodel_exclude_bottom_frac", 0.28)
        config["_ads_active"] = ads_active
        self._ensure_detect_ctx(config)
        self._apply_motion_config(config, cx, cy)

        sticky, currently_locked, _ = detection_sticky_context(self._lock_state, config)
        det_mode = str(config.get("detection_mode", "apex")).strip().lower()

        def _on_lock_expired() -> None:
            self.tracker.soft_reset()

        if det_mode == "yolo":
            from yolo_targeting import resolve_yolo_engine, yolo_detect_and_lock

            self._yolo_engine = resolve_yolo_engine(config, self._yolo_engine)
            result, _box = yolo_detect_and_lock(
                config,
                frame_bgr,
                self._yolo_engine,
                self._lock_state,
                fov_radius=detect_fov,
                center_x=cx,
                center_y=cy,
                frame_size=(w, h),
                on_lock_expired=_on_lock_expired,
                debug=debug,
            )
            is_stale = (
                bool(result.active)
                and result.target is not None
                and self._lock_state.target_lost_frames > 0
            )
        else:
            raw = cv_find_best_target_from_config(
                frame_bgr,
                config,
                fov_radius=detect_fov,
                min_area=min_area,
                center_x=cx,
                center_y=cy,
                sticky_target=sticky,
                currently_locked=currently_locked,
                context=self._detect_ctx,
                external_boxes=None,
                debug=debug,
            )
            result, is_stale = apply_target_lock(
                self._lock_state,
                raw,
                center_y=cy,
                cfg=config,
                on_lock_expired=_on_lock_expired,
                fov_cx=cx,
                fov_cy=cy,
                frame_size=(w, h),
            )
            locked = self._lock_state.locked_target
            if (
                self._detect_ctx is not None
                and locked is not None
                and result.target is not None
                and self._lock_state.target_lost_frames == 0
                and locked_target_may_refresh_motion_memory(
                    locked,
                    self._lock_state,
                    center_y=cy,
                )
            ):
                self._detect_ctx.note_motion_validated(
                    locked.bbox_x,
                    locked.bbox_y,
                    locked.bbox_w,
                    locked.bbox_h,
                )

        tsec = time.perf_counter() if time_sec is None else time_sec
        detection_fresh = (
            bool(result.active)
            and result.target is not None
            and not is_stale
        )
        stale_grace = int(config.get("mouse_gate_stale_grace_frames", 12))
        lost_frames = self._lock_state.target_lost_frames

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
            motion = self._last_yolo_motion if hasattr(self, "_last_yolo_motion") else None
            if motion is None:
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
                self._last_yolo_motion = motion
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
        px, py = ring_clamp_frame_point(
            ox,
            oy,
            cx,
            cy,
            detect_fov=float(detect_fov),
            display_fov=display_fov,
        )
        inside_o = _inside_body(t, ox, oy, tracker=self.tracker)
        inside_p = _inside_body(t, px, py, tracker=self.tracker)
        may_pull = may_assist_pull_target(
            t,
            detection_fresh=detection_fresh,
            center_y=cy,
            target_lost_frames=lost_frames,
            stale_grace_frames=stale_grace,
            frame_w=w,
            frame_h=h,
            fov_cx=cx,
            fov_cy=cy,
        )

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
            may_assist_pull=may_pull,
            show_for_overlay=detection_fresh or (
                lost_frames > 0 and lost_frames <= stale_grace
            ),
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
