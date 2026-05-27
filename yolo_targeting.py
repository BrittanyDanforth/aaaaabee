"""Single YOLO detect + lock pipeline (runtime, tests, detector delegation)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

from apexaimbot_bridge import (
    ApexAimBotRuntime,
    detect_frame,
    get_apexaimbot_runtime,
    reset_apexaimbot_cache,
)
from detector import DetectionResult, Target
from profiles import is_yolo_detection
from target_lock import (
    TargetLockState,
    apply_target_lock,
    apply_yolo_target_lock,
    detection_sticky_context,
)

logger = logging.getLogger("targeting")

APP_ROOT = Path(__file__).resolve().parent
VENDOR_ROOT = APP_ROOT / "third_party" / "apexaimbot"


def reset_yolo_engine_cache() -> None:
    reset_apexaimbot_cache()


def get_yolo_engine(cfg: dict[str, Any]) -> ApexAimBotRuntime | None:
    """Return cached ApexAimBot runtime when detection_mode is yolo."""
    if str(cfg.get("detection_mode", "apex")).strip().lower() != "yolo":
        return None
    if str(VENDOR_ROOT) not in sys.path:
        sys.path.insert(0, str(VENDOR_ROOT))
    return get_apexaimbot_runtime(cfg)


def reload_yolo_engine(cfg: dict[str, Any]) -> ApexAimBotRuntime | None:
    reset_yolo_engine_cache()
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
    """Vendored infer + area/height/confidence post-filters."""
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
            None,
            result.candidates,
            0.0,
            debug_lines=result.debug_lines + ["below_min_area"],
            active=False,
        )
    if min_height_px > 0 and t.bbox_h < min_height_px:
        return DetectionResult(
            None,
            result.candidates,
            0.0,
            debug_lines=result.debug_lines + ["below_min_height"],
            active=False,
        )
    if t.confidence < min_confidence:
        return DetectionResult(
            None,
            result.candidates,
            0.0,
            debug_lines=result.debug_lines + ["below_min_confidence"],
            active=False,
        )
    if sticky_target is not None and stickiness_pixels > 0 and currently_locked:
        import math

        dist = math.hypot(
            t.centroid_x - sticky_target.centroid_x,
            t.centroid_y - sticky_target.centroid_y,
        )
        if dist > stickiness_pixels * 2.5:
            pass  # nearest pick wins — no CV-style pool
    return result


def resolve_yolo_engine(cfg: dict[str, Any], cached: Any | None) -> Any | None:
    """Return cached engine or load from cfg when detection_mode is yolo."""
    if not is_yolo_detection(cfg):
        return None
    if cached is not None:
        return cached
    return get_yolo_engine(cfg)


def detect_yolo_target(
    cfg: dict[str, Any],
    frame_bgr: np.ndarray,
    engine: Any,
    *,
    fov_radius: int,
    center_x: float,
    center_y: float,
    sticky_target: Target | None = None,
    currently_locked: bool = False,
    debug: bool = False,
) -> DetectionResult:
    """Vendored infer + post-filters — no lock."""
    if engine is None:
        return DetectionResult(
            None,
            0,
            0.0,
            debug_lines=["yolo_mode but engine not loaded — set yolo_weights_path"],
            active=False,
        )
    min_area = float(cfg.get("min_target_area_pixels", 40))
    return find_best_yolo_target(
        frame_bgr,
        int(fov_radius),
        min_area,
        center_x,
        center_y,
        engine=engine,
        sticky_target=sticky_target,
        stickiness_pixels=float(cfg.get("target_stickiness_pixels", 90.0)),
        min_height_px=float(cfg.get("humanoid_min_height_pixels", 0)),
        min_confidence=float(cfg.get("yolo_confidence_min", 0.5)),
        currently_locked=currently_locked,
        ads_active=bool(cfg.get("_ads_active", False)),
        debug=debug,
    )


def lock_yolo_detection(
    cfg: dict[str, Any],
    raw: DetectionResult,
    lock_state: TargetLockState,
    *,
    center_y: float,
    frame_size: tuple[int, int],
    on_lock_expired: Callable[[], None] | None = None,
    on_new_target: Callable[[], None] | None = None,
) -> DetectionResult:
    """Apply YOLO nearest lock or CV lock when yolo_apex_nearest_lock=False."""
    fw, fh = frame_size
    if bool(cfg.get("yolo_apex_nearest_lock", True)):
        result, _is_stale = apply_yolo_target_lock(
            lock_state,
            raw,
            cfg=cfg,
            on_lock_expired=on_lock_expired,
            on_new_target=on_new_target,
        )
        return result
    result, _is_stale = apply_target_lock(
        lock_state,
        raw,
        center_y=center_y,
        cfg=cfg,
        on_lock_expired=on_lock_expired,
        fov_cx=None,
        fov_cy=center_y,
        frame_size=(fw, fh),
    )
    return result


def last_apex_box_from(
    lock_state: TargetLockState, result: DetectionResult
) -> tuple[float, float] | None:
    src = lock_state.locked_target if lock_state.locked_target is not None else result.target
    if src is None:
        return None
    return float(src.bbox_w), float(src.bbox_h)


def yolo_detect_and_lock(
    cfg: dict[str, Any],
    frame_bgr: np.ndarray,
    engine: Any,
    lock_state: TargetLockState,
    *,
    fov_radius: int,
    center_x: float,
    center_y: float,
    frame_size: tuple[int, int],
    on_lock_expired: Callable[[], None] | None = None,
    on_new_target: Callable[[], None] | None = None,
    debug: bool = False,
) -> tuple[DetectionResult, tuple[float, float] | None]:
    """Full production path: sticky context → detect → lock → bbox hint for PID."""
    sticky, currently_locked, _ = detection_sticky_context(lock_state, cfg)
    raw = detect_yolo_target(
        cfg,
        frame_bgr,
        engine,
        fov_radius=fov_radius,
        center_x=center_x,
        center_y=center_y,
        sticky_target=sticky,
        currently_locked=currently_locked,
        debug=debug,
    )
    fh, fw = frame_size
    result = lock_yolo_detection(
        cfg,
        raw,
        lock_state,
        center_y=center_y,
        frame_size=(fw, fh),
        on_lock_expired=on_lock_expired,
        on_new_target=on_new_target,
    )
    return result, last_apex_box_from(lock_state, result)


def tick_yolo_lock_idle(
    cfg: dict[str, Any],
    lock_state: TargetLockState,
    *,
    on_lock_expired: Callable[[], None] | None = None,
) -> None:
    """Advance lock grace / expiry when detect did not run this frame."""
    if not is_yolo_detection(cfg):
        return
    lock_yolo_detection(
        cfg,
        DetectionResult(None, 0, 0.0),
        lock_state,
        center_y=0.0,
        frame_size=(1, 1),
        on_lock_expired=on_lock_expired,
    )
