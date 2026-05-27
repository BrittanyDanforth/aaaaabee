"""YOLO primary detection — vendored ApexAimBot YOLOv5 (1:1 detect path from their main.py)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

from apexaimbot_bridge import (
    ApexAimBotRuntime,
    detect_frame,
    get_apexaimbot_runtime,
    reset_apexaimbot_cache,
)
from detector import DetectionResult, Target

logger = logging.getLogger("targeting")

APP_ROOT = Path(__file__).resolve().parent
VENDOR_ROOT = APP_ROOT / "third_party" / "apexaimbot"

# Back-compat aliases for tests / imports
YoloDetection = None  # unused — vendored path returns Target directly
YoloEngine = ApexAimBotRuntime
YoloEngineConfig = None


def reset_yolo_engine_cache() -> None:
    """Clear vendored engine cache (single source: apexaimbot_bridge)."""
    reset_apexaimbot_cache()


def get_yolo_engine(cfg: dict[str, Any]) -> ApexAimBotRuntime | None:
    """Return cached ApexAimBot runtime; cfg should come from normalize_app_config."""
    if str(cfg.get("detection_mode", "apex")).strip().lower() != "yolo":
        return None
    if str(VENDOR_ROOT) not in sys.path:
        sys.path.insert(0, str(VENDOR_ROOT))
    return get_apexaimbot_runtime(cfg)


def reload_yolo_engine(cfg: dict[str, Any]) -> ApexAimBotRuntime | None:
    """Invalidate cache and load engine (hot-reload from GUI/runtime_controller)."""
    reset_yolo_engine_cache()
    return get_yolo_engine(cfg)


def try_create_yolo_engine(cfg: dict[str, Any]) -> ApexAimBotRuntime | None:
    return get_yolo_engine(cfg)


def find_best_yolo_target(
    frame_bgr: np.ndarray,
    fov_radius: int,
    min_area: float,
    fov_center_x: float,
    fov_center_y: float,
    *,
    engine: ApexAimBotRuntime,
    sticky_target: Target | None = None,
    stickiness_pixels: float = 90.0,
    min_height_px: float = 0.0,
    min_confidence: float = 0.30,
    currently_locked: bool = False,
    ads_active: bool = False,
    debug: bool = False,
) -> DetectionResult:
    """Delegate to vendored ApexAimBot detect; optional sticky by centroid distance."""
    result = detect_frame(
        engine,
        frame_bgr,
        fov_center_x=fov_center_x,
        fov_center_y=fov_center_y,
    )
    if debug and result.debug_lines:
        result.debug_lines.append(
            f"sticky={sticky_target is not None} ads={ads_active} fov_r={fov_radius}"
        )
    t = result.target
    if t is None:
        return result
    if min_area > 0 and t.area < min_area:
        return DetectionResult(
            None, result.candidates, 0.0, debug_lines=result.debug_lines + ["below_min_area"],
            active=False,
        )
    if min_height_px > 0 and t.bbox_h < min_height_px:
        return DetectionResult(
            None, result.candidates, 0.0, debug_lines=result.debug_lines + ["below_min_height"],
            active=False,
        )
    # YOLO scores are synthetic — only enforce model confidence, not CV body gates.
    if t.confidence < min_confidence:
        return DetectionResult(
            None,
            result.candidates,
            0.0,
            debug_lines=result.debug_lines + ["below_min_confidence"],
            active=False,
        )
    if sticky_target is not None and stickiness_pixels > 0:
        import math
        dist = math.hypot(
            t.centroid_x - sticky_target.centroid_x,
            t.centroid_y - sticky_target.centroid_y,
        )
        if dist > stickiness_pixels * 2.5 and currently_locked:
            pass  # still return new pick — Apex has no CV-style pool; nearest wins
    return result
