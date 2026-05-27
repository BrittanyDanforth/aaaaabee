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
    reset_apexaimbot_cache()


def _yolo_cache_key(cfg: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(cfg.get("detection_mode", "apex")).lower(),
        str(cfg.get("yolo_weights_path", "")),
        str(cfg.get("yolo_yolov5_root", "")),
        int(cfg.get("yolo_inference_size", 416)),
        float(cfg.get("yolo_confidence_min", 0.5)),
        str(cfg.get("yolo_device", "auto")),
    )


_yolo_engine_cache: tuple[tuple[Any, ...], ApexAimBotRuntime | None] | None = None


def get_yolo_engine(cfg: dict[str, Any]) -> ApexAimBotRuntime | None:
    if str(cfg.get("detection_mode", "apex")).strip().lower() != "yolo":
        return None
    global _yolo_engine_cache
    from apexaimbot_bridge import prepare_apex_cfg

    merged = prepare_apex_cfg(cfg)
    if str(VENDOR_ROOT) not in sys.path:
        sys.path.insert(0, str(VENDOR_ROOT))
    key = _yolo_cache_key(merged)
    if _yolo_engine_cache is None or _yolo_engine_cache[0] != key:
        _yolo_engine_cache = (key, get_apexaimbot_runtime(merged))
    return _yolo_engine_cache[1]


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
    body_shape_min_score: float = 0.40,
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
    if t.area < min_area:
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
