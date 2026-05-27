"""Load ApexAimBot DetectMultiBackend — copied from main._init_main."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("targeting")

VENDOR_ROOT = Path(__file__).resolve().parent
APP_ROOT = VENDOR_ROOT.parent.parent


@dataclass
class ApexAimBotDetectConfig:
    weights_path: Path
    model_imgsz: int = 416
    conf_thres: float = 0.5
    iou_thres: float = 0.25
    use_fp16: bool = False
    grab_width: float = 416.0
    grab_height: float = 416.0
    aim_offset_fraction: float = 0.2
    max_det: int = 3
    lock_range_x: float = 1.0
    lock_range_y: float = 0.5
    min_step: int = 10
    max_step: int = 6

    @classmethod
    def from_app_config(cls, cfg: dict[str, Any]) -> ApexAimBotDetectConfig:
        w = str(cfg.get("yolo_weights_path", "") or "").strip()
        if not w:
            raise ValueError("yolo_weights_path required")
        wp = Path(w)
        for base in (APP_ROOT, VENDOR_ROOT, Path.cwd()):
            if not wp.is_file():
                alt = base / w
                if alt.is_file():
                    wp = alt
                    break
        if not wp.is_file():
            alt = VENDOR_ROOT / "weights" / Path(w).name
            if alt.is_file():
                wp = alt
        if not wp.is_file():
            raise FileNotFoundError(f"weights not found: {w}")
        gw = float(cfg.get("yolo_grab_width", cfg.get("yolo_inference_size", 416)))
        gh = float(cfg.get("yolo_grab_height", cfg.get("yolo_inference_size", 416)))
        return cls(
            weights_path=wp.resolve(),
            model_imgsz=int(cfg.get("yolo_inference_size", 416)),
            conf_thres=float(cfg.get("yolo_confidence_min", 0.5)),
            iou_thres=float(cfg.get("yolo_iou_thres", 0.25)),
            use_fp16=bool(cfg.get("yolo_use_fp16", False)),
            grab_width=gw,
            grab_height=gh,
            aim_offset_fraction=float(cfg.get("yolo_aim_fraction", 0.2)),
            max_det=int(cfg.get("yolo_max_det", 3)),
            lock_range_x=float(cfg.get("apexaimbot_lock_range_x", 1.0)),
            lock_range_y=float(cfg.get("apexaimbot_lock_range_y", 0.5)),
            min_step=int(cfg.get("apexaimbot_min_step", 10)),
            max_step=int(cfg.get("apexaimbot_max_step", 6)),
        )


def ensure_vendor_path() -> Path:
    root = VENDOR_ROOT
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def load_detect_model(cfg: ApexAimBotDetectConfig, *, device: str = "") -> Any:
    """Mirror ApexAimBot _init_main model load."""
    ensure_vendor_path()
    from models.common import DetectMultiBackend
    from utils.torch_utils import select_device

    data = VENDOR_ROOT / "yaml" / "coco128.yaml"
    dev = select_device(device)
    fp16 = bool(cfg.use_fp16)
    logger.info("ApexAimBot loading %s fp16=%s device=%s", cfg.weights_path.name, fp16, dev)
    model = DetectMultiBackend(
        str(cfg.weights_path), device=dev, dnn=False, data=data, fp16=fp16
    )
    model.warmup(imgsz=(1, 3, cfg.model_imgsz, cfg.model_imgsz))
    names = [name for name in model.names.values()]
    logger.info("ApexAimBot classes: %s", names)
    return model
