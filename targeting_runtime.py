"""
Canonical runtime wiring: detector body target -> motion.observe_target(bbox_*) -> stable aim.

Import this from assist.py / runtime loop instead of calling find_best_target + observe separately.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import detector
from detector import DetectionResult, Target
from motion import TargetMotion, TargetTracker


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
    End-to-end targeting session.

    Guarantees every active frame calls:
        tracker.observe_target(x, y, t, bbox_x=, bbox_y=, bbox_w=, bbox_h=)
    when a body-shaped target is selected.
    """

    def __init__(self) -> None:
        self.tracker = TargetTracker()
        self._sticky: Target | None = None
        self._last_observe_bbox: tuple[int, int, int, int] | None = None
        self._observe_target_calls: int = 0

    def reset(self) -> None:
        self.tracker.reset()
        self._sticky = None
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
        stickiness = float(config.get("stickiness_pixels", 90.0))

        result = detector.find_best_target(
            frame_bgr,
            config["hsv_ranges"],
            fov,
            min_area,
            cx,
            cy,
            sticky_target=self._sticky,
            stickiness_pixels=stickiness,
            min_height_px=float(config.get("humanoid_min_height_pixels", 0)),
            min_aspect=config.get("humanoid_min_aspect"),
            max_aspect=config.get("humanoid_max_aspect"),
            debug=debug,
        )

        tsec = time.perf_counter() if time_sec is None else time_sec

        if not result.active or result.target is None:
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
        self._sticky = t

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
    """Functional wrapper used by assist.py after creating one TargetingRuntime per session."""
    return runtime.process_frame(frame_bgr, config, time_sec=time_sec)
