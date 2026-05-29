"""Shared detection helpers — production runtime and test harness must use the same kwargs."""

from __future__ import annotations

import math
from typing import Any

from detector import DetectionResult, find_best_target
from profiles import effective_overlay_fov_radius
from target_lock import viewmodel_exclude_bottom


def ring_clamp_distance_limit(detect_fov: float, display_fov: float) -> float:
    """96% of min(detect, display) — matches ``AssistRuntime._frame_overlay_point``."""
    return max(1.0, min(float(detect_fov), float(display_fov))) * 0.96


def ring_clamp_frame_point(
    ox: float,
    oy: float,
    cx: float,
    cy: float,
    *,
    detect_fov: float,
    display_fov: float,
) -> tuple[float, float]:
    """Ring-clamp overlay/pull anchor in frame space (harness + scripts)."""
    dx = ox - cx
    dy = oy - cy
    dist = math.hypot(dx, dy)
    lim = ring_clamp_distance_limit(detect_fov, display_fov)
    if dist > lim and dist > 0.0:
        s = lim / dist
        return cx + dx * s, cy + dy * s
    return ox, oy


def cv_find_best_target_from_config(
    frame_bgr: Any,
    cfg: dict[str, Any],
    *,
    fov_radius: int,
    min_area: float,
    center_x: float,
    center_y: float,
    sticky_target: Any | None,
    currently_locked: bool,
    context: Any | None,
    external_boxes: Any | None = None,
    debug: bool = False,
) -> DetectionResult:
    """Single CV detection call shape for ``runtime`` and ``targeting_runtime``."""
    return find_best_target(
        frame_bgr,
        cfg.get("hsv_ranges"),
        fov_radius,
        min_area,
        center_x,
        center_y,
        sticky_target=sticky_target,
        stickiness_pixels=float(cfg.get("target_stickiness_pixels", 90.0)),
        distance_weight=float(cfg.get("distance_score_weight", 1.0)),
        area_weight=float(cfg.get("area_score_weight", 0.5)),
        min_height_px=float(cfg.get("humanoid_min_height_pixels", 16)),
        min_aspect=float(cfg.get("humanoid_min_aspect", 1.2)),
        max_aspect=float(cfg.get("humanoid_max_aspect", 4.5)),
        min_solidity=float(cfg.get("humanoid_min_solidity", 0.25)),
        torso_aim_fraction=float(cfg.get("torso_aim_fraction", 0.38)),
        body_shape_min_score=float(cfg.get("body_shape_min_score", 0.40)),
        head_score_weight=float(cfg.get("head_score_weight", 0.26)),
        torso_score_weight=float(cfg.get("torso_score_weight", 0.26)),
        limb_stack_score_weight=float(cfg.get("limb_stack_score_weight", 0.22)),
        aim_y_min_fraction=float(cfg.get("aim_body_y_min_fraction", 0.28)),
        aim_y_max_fraction=float(cfg.get("aim_body_y_max_fraction", 0.52)),
        debug=debug,
        detection_mode=str(cfg.get("detection_mode", "apex")),
        context=context,
        currently_locked=currently_locked,
        exclude_bottom_frac=viewmodel_exclude_bottom(cfg),
        display_fov_radius=float(effective_overlay_fov_radius(cfg)),
        target_selection_mode=str(cfg.get("target_selection_mode", "apex")),
        external_boxes=external_boxes,
        yolo_fusion_boost=float(cfg.get("yolo_fusion_boost", 0.30)),
        yolo_fusion_min_iou=float(cfg.get("yolo_fusion_min_iou", 0.28)),
        yolo_engine=None,
        ads_active=bool(cfg.get("_ads_active", False)),
    )
