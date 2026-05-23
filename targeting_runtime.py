"""
Canonical detection + aim wiring for tests and artifact scripts.

Production live play uses ``AssistRuntime`` in ``runtime.py`` (same lock module).

This module runs ``find_best_target`` + ``apply_target_lock`` — not a separate
sticky-only shortcut. Do not assign ``_sticky = t`` without the frame lock.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import detector
from detector import DetectionResult, Target
from motion import TargetMotion, TargetTracker
from target_lock import (
    TargetLockState,
    apply_target_lock,
    detection_sticky_context,
    viewmodel_exclude_bottom,
)


@dataclass
class AimState:
    """Stable aim anchor after body-shape detection + dt-aware motion."""

    aim_x: float
    aim_y: float
    vx: float
    vy: float
    active: bool
    target: Target | None = None
    detection: DetectionResult | None = None
    debug_lines: list[str] = field(default_factory=list)
    bbox_used: tuple[int, int, int, int] | None = None
    wired_observe_target: bool = True


class TargetingRuntime:
    """
    End-to-end targeting session for tests/artifacts.

    Uses the same frame lock as ``AssistRuntime`` (``target_lock`` module).
    """

    def __init__(self) -> None:
        self.tracker = TargetTracker()
        self._lock_state = TargetLockState()
        self._last_observe_bbox: tuple[int, int, int, int] | None = None
        self._observe_target_calls: int = 0

    def reset(self) -> None:
        self.tracker.reset()
        self._lock_state.reset()
        self._last_observe_bbox = None

    @property
    def last_observe_bbox(self) -> tuple[int, int, int, int] | None:
        return self._last_observe_bbox

    @property
    def observe_target_call_count(self) -> int:
        return self._observe_target_calls

    def process_frame(
        self,
        frame_bgr: Any,
        config: dict[str, Any],
        *,
        time_sec: float | None = None,
        debug: bool = True,
    ) -> AimState:
        h, w = frame_bgr.shape[:2]
        cx = float(config.get("fov_center_x", w / 2.0))
        cy = float(config.get("fov_center_y", h / 2.0))
        fov = int(config.get("fov_radius_pixels", 200))
        min_area = float(config.get("min_target_area", 40.0))
        config.setdefault("_runtime_detect_fov", float(fov))
        config.setdefault("target_lost_frames_before_unlock", 18)
        config.setdefault("viewmodel_exclude_bottom_frac", 0.28)

        sticky, currently_locked, _ = detection_sticky_context(self._lock_state, config)
        raw = detector.find_best_target(
            frame_bgr,
            config["hsv_ranges"],
            fov,
            min_area,
            cx,
            cy,
            sticky_target=sticky,
            stickiness_pixels=float(config.get("stickiness_pixels", 90.0)),
            currently_locked=currently_locked,
            exclude_bottom_frac=viewmodel_exclude_bottom(config),
            min_height_px=float(config.get("humanoid_min_height_pixels", 0)),
            min_aspect=config.get("humanoid_min_aspect"),
            max_aspect=config.get("humanoid_max_aspect"),
            debug=debug,
        )
        result, _stale = apply_target_lock(
            self._lock_state,
            raw,
            center_y=cy,
            cfg=config,
            on_lock_expired=self.tracker.soft_reset,
        )

        tsec = time.perf_counter() if time_sec is None else time_sec

        if result.target is None:
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
            )

        t = result.target
        self._observe_target_calls += 1
        self._last_observe_bbox = (t.bbox_x, t.bbox_y, t.bbox_w, t.bbox_h)
        motion = self.tracker.observe_target(
            t.centroid_x,
            t.centroid_y,
            tsec,
            bbox_x=t.bbox_x,
            bbox_y=t.bbox_y,
            bbox_w=t.bbox_w,
            bbox_h=t.bbox_h,
        )

        return AimState(
            aim_x=motion.x,
            aim_y=motion.y,
            vx=motion.vx,
            vy=motion.vy,
            active=True,
            target=t,
            detection=result,
            debug_lines=list(result.debug_lines),
            bbox_used=self._last_observe_bbox,
            wired_observe_target=True,
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
