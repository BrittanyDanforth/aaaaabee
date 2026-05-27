"""Bridge vendored ApexAimBot detect + PID into ABA Target / mouse pull."""

from __future__ import annotations

import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from detector import DetectionResult, Target

logger = logging.getLogger("targeting")

VENDOR_DEFAULT = Path(__file__).resolve().parent / "third_party" / "apexaimbot"


@dataclass
class ApexAimBotRuntime:
    """Holds vendored model + PID controllers (ApexAimBot main.py parity)."""

    model: Any
    config: Any
    pid_x: Any
    pid_y: Any
    mouse_modifier: float = 1.0

    @classmethod
    def from_app_config(cls, cfg: dict[str, Any]) -> ApexAimBotRuntime:
        vendor = _resolve_vendor_root(cfg)
        if str(vendor) not in sys.path:
            sys.path.insert(0, str(vendor))
        from engine import ApexAimBotDetectConfig, load_detect_model
        from PID import PID_PLUS_PLUS

        acfg = ApexAimBotDetectConfig.from_app_config(cfg)
        model = load_detect_model(acfg, device=resolve_yolo_device(cfg))
        px = float(cfg.get("pid_x_p", cfg.get("apexaimbot_pid_x_p", 0.36)))
        ix = float(cfg.get("pid_x_i", cfg.get("apexaimbot_pid_x_i", 0.032)))
        dx = float(cfg.get("pid_x_d", cfg.get("apexaimbot_pid_x_d", 0.01)))
        py = float(cfg.get("pid_y_p", cfg.get("apexaimbot_pid_y_p", 0.2)))
        iy = float(cfg.get("pid_y_i", cfg.get("apexaimbot_pid_y_i", 0.0)))
        dy = float(cfg.get("pid_y_d", cfg.get("apexaimbot_pid_y_d", 0.0)))
        mod = float(cfg.get("apexaimbot_mouse_modifier", 0.8))
        return cls(
            model=model,
            config=acfg,
            pid_x=PID_PLUS_PLUS(0, px, ix, dx),
            pid_y=PID_PLUS_PLUS(0, py, iy, dy),
            mouse_modifier=max(0.05, min(4.0, mod)),
        )


def resolve_yolo_device(cfg: dict[str, Any]) -> str:
    """Map ABA config to YOLOv5 select_device ('' → first CUDA GPU)."""
    raw = str(cfg.get("yolo_device", "") or "").strip().lower()
    if raw in ("", "auto", "cuda"):
        return ""
    if raw in ("cpu", "mps"):
        return raw
    if raw.isdigit():
        return raw
    return ""


def _resolve_vendor_root(cfg: dict[str, Any]) -> Path:
    raw = str(cfg.get("yolo_yolov5_root", "") or "").strip()
    if raw:
        p = Path(raw)
        if p.is_dir():
            return p.resolve()
        alt = Path(__file__).resolve().parent / raw
        if alt.is_dir():
            return alt.resolve()
    if VENDOR_DEFAULT.is_dir() and (VENDOR_DEFAULT / "models" / "common.py").is_file():
        return VENDOR_DEFAULT.resolve()
    raise FileNotFoundError(
        "yolo_yolov5_root must point at third_party/apexaimbot (vendored ApexAimBot yolov5)"
    )


_engine_cache: tuple[tuple[Any, ...], ApexAimBotRuntime | None] | None = None


def reset_apexaimbot_cache() -> None:
    global _engine_cache
    _engine_cache = None


def _cache_key(cfg: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(cfg.get("yolo_weights_path", "")),
        str(cfg.get("yolo_yolov5_root", "")),
        int(cfg.get("yolo_inference_size", 416)),
        float(cfg.get("yolo_confidence_min", 0.5)),
        float(cfg.get("yolo_iou_thres", 0.25)),
    )


def _load_runtime(cfg: dict[str, Any]) -> ApexAimBotRuntime:
    return ApexAimBotRuntime.from_app_config(cfg)


def get_apexaimbot_runtime(cfg: dict[str, Any]) -> ApexAimBotRuntime | None:
    global _engine_cache
    if str(cfg.get("detection_mode", "apex")).strip().lower() != "yolo":
        return None
    key = _cache_key(cfg)
    if _engine_cache is None or _engine_cache[0] != key:
        rt: ApexAimBotRuntime | None = None
        try:
            rt = _load_runtime(cfg)
        except Exception as exc:
            wpath = str(cfg.get("yolo_weights_path", "") or "")
            if wpath.lower().endswith(".engine"):
                logger.warning(
                    "TensorRT engine load failed (%s); trying APEX22W.pt", exc
                )
                fb = dict(cfg)
                fb["yolo_weights_path"] = "third_party/apexaimbot/weights/APEX22W.pt"
                try:
                    rt = _load_runtime(fb)
                    logger.info("ApexAimBot using PyTorch fallback APEX22W.pt")
                except Exception as exc2:
                    logger.error("ApexAimBot PT fallback failed: %s", exc2)
            else:
                logger.error("ApexAimBot runtime load failed: %s", exc)
        _engine_cache = (key, rt)
    return _engine_cache[1]


def detect_frame(
    rt: ApexAimBotRuntime,
    frame_bgr: np.ndarray,
    *,
    fov_center_x: float,
    fov_center_y: float,
) -> DetectionResult:
    """Run vendored detect + nearest pick; map to ABA Target."""
    from detect import interface_img_gpt_plus
    from nearest import send_nearest_pos_to_mouse_ctrl

    h, w = frame_bgr.shape[:2]
    cx = float(fov_center_x)
    cy = float(fov_center_y)
    cfg = rt.config
    dbg = ["mode=apexaimbot_vendored"]

    gw = float(cfg.grab_width) if w >= cfg.grab_width else float(w)
    gh = float(cfg.grab_height) if h >= cfg.grab_height else float(h)
    # Center crop like Apex grab_rect when frame is larger than grab size
    if w > gw or h > gh:
        x0 = int(max(0, (w - gw) / 2))
        y0 = int(max(0, (h - gh) / 2))
        crop = frame_bgr[y0 : y0 + int(gh), x0 : x0 + int(gw)]
        offset_x, offset_y = float(x0), float(y0)
    else:
        crop = frame_bgr
        offset_x, offset_y = 0.0, 0.0
        gw, gh = float(w), float(h)

    try:
        box_list = interface_img_gpt_plus(
            crop,
            model=rt.model,
            model_imgsz=cfg.model_imgsz,
            conf_thres=cfg.conf_thres,
            iou_thres=cfg.iou_thres,
            use_fp_16=cfg.use_fp16,
            max_det=cfg.max_det,
        )
    except Exception as exc:
        dbg.append(f"infer_fail={exc}")
        return DetectionResult(None, 0, 0.0, debug_lines=dbg, active=False)

    dbg.append(f"raw_boxes={len(box_list)}")
    result = send_nearest_pos_to_mouse_ctrl(box_list, grab_width=gw, grab_height=gh)
    if result is None:
        dbg.append("no_nearest")
        return DetectionResult(None, len(box_list), 0.0, debug_lines=dbg, active=False)

    (pos_min, box_width, box_height) = result
    # Apex aim: offset = int(box_height * 0.2) applied on Y error before PID
    aim_x = cx + pos_min[0] + offset_x
    aim_y = cy + pos_min[1] - box_height * cfg.aim_offset_fraction + offset_y
    dist = math.hypot(aim_x - cx, aim_y - cy)
    conf = 0.85
    if box_list:
        conf = max(b[5] for b in box_list) / 100.0

    half_w = box_width / 2
    half_h = box_height / 2
    center_x_crop = gw / 2 + pos_min[0]
    center_y_crop = gh / 2 + pos_min[1]

    t = Target(
        centroid_x=aim_x,
        centroid_y=aim_y,
        area=box_width * box_height,
        distance_to_center=dist,
        confidence=conf,
        bbox_x=int(offset_x + center_x_crop - half_w),
        bbox_y=int(offset_y + center_y_crop - half_h),
        bbox_w=int(max(1, box_width)),
        bbox_h=int(max(1, box_height)),
        solidity=0.75,
        humanoid_score=conf,
        part_count=3,
        body_shape_score=conf,
        head_score=conf,
        torso_score=conf,
        limb_stack_score=0.5,
        red_coverage=0.12,
        fill_ratio=0.65,
        max_circularity=0.45,
        has_classified_torso=True,
    )
    dbg.append(
        f"nearest pos=({pos_min[0]:.0f},{pos_min[1]:.0f}) "
        f"box={box_width:.0f}x{box_height:.0f} aim=({aim_x:.0f},{aim_y:.0f})"
    )
    return DetectionResult(t, len(box_list), conf, debug_lines=dbg, active=True)


def pid_mouse_delta(
    rt: ApexAimBotRuntime,
    *,
    error_x: float,
    error_y: float,
    hip_fire: bool,
) -> tuple[int, int]:
    """ApexAimBot run_ai PID + step caps (min_step when left_down_not_right, else max_step)."""
    step = rt.config.min_step if hip_fire else rt.config.max_step
    pid_x = int(rt.pid_x.getMove(error_x, step))
    pid_y = int(rt.pid_y.getMove(error_y))
    mod = float(rt.mouse_modifier)
    return int(round(pid_x * mod)), int(round(pid_y * mod))


def in_lock_box(
    rt: ApexAimBotRuntime,
    *,
    error_x: float,
    error_y: float,
    box_width: float,
    box_height: float,
    hip_fire: bool,
) -> bool:
    """have_luck from Apex main: _range 1.0 hip / 0.7 ADS, _range_y 0.5."""
    rng_x = 1.0 if hip_fire else 0.7
    rng_y = rt.config.lock_range_y
    return abs(error_x) <= (box_width * rng_x) and abs(error_y) <= (box_height * rng_y)


def validate_yolo_config(cfg: dict[str, Any]) -> Path:
    """Resolve weights path or raise (for setup doctor / self-check)."""
    merged = dict(cfg)
    if not merged.get("yolo_yolov5_root") and VENDOR_DEFAULT.is_dir():
        merged["yolo_yolov5_root"] = str(VENDOR_DEFAULT)
    vendor = _resolve_vendor_root(merged)
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    from engine import ApexAimBotDetectConfig

    return ApexAimBotDetectConfig.from_app_config(merged).weights_path
