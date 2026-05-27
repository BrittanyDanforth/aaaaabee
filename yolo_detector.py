"""Primary YOLO detection for ABA — replaces CV masks when detection_mode=yolo.

Uses YOLOv5-style weights (.pt via torch hub, or .onnx/.engine with vendored yolov5).
ApexAimBot-compatible: conf/IoU, teammate filter, nearest-to-crosshair pick, chest aim.
"""

from __future__ import annotations

import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from detector import DetectionResult, Target, _bbox_iou
from tracking_fusion import bbox_iou, pick_nearest_crosshair, select_ranked_target

logger = logging.getLogger("targeting")

APP_ROOT = Path(__file__).resolve().parent


@dataclass
class YoloDetection:
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    confidence: float
    class_name: str


@dataclass
class YoloEngineConfig:
    weights_path: Path
    yolov5_root: Path | None = None
    inference_size: int = 416
    confidence_min: float = 0.5
    iou_thres: float = 0.25
    max_det: int = 12
    device: str = "auto"
    use_fp16: bool = False
    exclude_labels: frozenset[str] = frozenset({"teammate"})
    aim_fraction: float = 0.38
    target_pick: str = "nearest"
    ads_aim_scale: float = 0.7

    @classmethod
    def from_app_config(cls, cfg: dict[str, Any]) -> YoloEngineConfig:
        w = str(cfg.get("yolo_weights_path", "") or "").strip()
        if not w:
            raise ValueError("yolo_weights_path is required when detection_mode=yolo")
        wp = Path(w)
        if not wp.is_file():
            wp2 = APP_ROOT / w
            if wp2.is_file():
                wp = wp2
            else:
                raise FileNotFoundError(
                    f"YOLO weights not found: {cfg.get('yolo_weights_path')} "
                    f"(also tried {wp2})"
                )
        root_raw = str(cfg.get("yolo_yolov5_root", "") or "").strip()
        root = Path(root_raw) if root_raw else None
        if root is not None and not root.is_dir():
            r2 = APP_ROOT / root_raw
            root = r2 if r2.is_dir() else root
        excl = cfg.get("yolo_exclude_labels", ["teammate"])
        if isinstance(excl, str):
            excl_set = frozenset(x.strip().lower() for x in excl.split(",") if x.strip())
        else:
            excl_set = frozenset(str(x).strip().lower() for x in excl)
        pick = str(cfg.get("yolo_target_pick", cfg.get("target_selection_mode", "nearest"))).lower()
        return cls(
            weights_path=wp.resolve(),
            yolov5_root=root.resolve() if root and root.is_dir() else None,
            inference_size=int(cfg.get("yolo_inference_size", 416)),
            confidence_min=float(cfg.get("yolo_confidence_min", 0.5)),
            iou_thres=float(cfg.get("yolo_iou_thres", 0.25)),
            max_det=int(cfg.get("yolo_max_det", 12)),
            device=str(cfg.get("yolo_device", "auto")),
            use_fp16=bool(cfg.get("yolo_use_fp16", False)),
            exclude_labels=excl_set,
            aim_fraction=float(cfg.get("yolo_aim_fraction", cfg.get("torso_aim_fraction", 0.38))),
            target_pick=pick,
            ads_aim_scale=float(cfg.get("yolo_ads_lock_scale", 0.7)),
        )


_yolo_engine_cache: tuple[tuple[Any, ...], YoloEngine | None] | None = None


def reset_yolo_engine_cache() -> None:
    """Clear cached engine (tests / hot-reload)."""
    global _yolo_engine_cache
    _yolo_engine_cache = None


def _yolo_cache_key(cfg: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(cfg.get("detection_mode", "apex")).lower(),
        str(cfg.get("yolo_weights_path", "")),
        str(cfg.get("yolo_yolov5_root", "")),
        int(cfg.get("yolo_inference_size", 416)),
        float(cfg.get("yolo_confidence_min", 0.5)),
        str(cfg.get("yolo_device", "auto")),
    )


def get_yolo_engine(cfg: dict[str, Any]) -> YoloEngine | None:
    """Return a cached YoloEngine when detection_mode=yolo, else None."""
    global _yolo_engine_cache
    if str(cfg.get("detection_mode", "apex")).strip().lower() != "yolo":
        return None
    key = _yolo_cache_key(cfg)
    if _yolo_engine_cache is None or _yolo_engine_cache[0] != key:
        _yolo_engine_cache = (key, try_create_yolo_engine(cfg))
    return _yolo_engine_cache[1]


def try_create_yolo_engine(cfg: dict[str, Any]) -> YoloEngine | None:
    mode = str(cfg.get("detection_mode", "apex")).strip().lower()
    if mode != "yolo":
        return None
    try:
        ycfg = YoloEngineConfig.from_app_config(cfg)
        return YoloEngine(ycfg)
    except Exception as exc:
        logger.error("YOLO engine failed to load: %s", exc)
        return None


class YoloEngine:
    """Loads YOLOv5 once; runs inference each frame."""

    def __init__(self, config: YoloEngineConfig) -> None:
        self.config = config
        self._backend = "hub"
        self._model = None
        self._device = None
        self._names: dict[int, str] = {}
        self._stride = 32
        self._load()

    def _resolve_device(self, torch: Any) -> Any:
        d = self.config.device
        if d == "auto":
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return torch.device(d)

    def _load(self) -> None:
        import torch

        self._device = self._resolve_device(torch)
        wp = self.config.weights_path
        suffix = wp.suffix.lower()

        if suffix in (".engine", ".onnx", ".xml") or self.config.yolov5_root:
            self._load_detect_multi_backend(torch)
            return
        self._load_torch_hub(torch)

    def _load_torch_hub(self, torch: Any) -> None:
        root = self.config.yolov5_root
        if root and (root / "hubconf.py").is_file():
            logger.info("YOLO loading via local yolov5 %s", root)
            self._model = torch.hub.load(
                str(root),
                "custom",
                path=str(self.config.weights_path),
                source="local",
                force_reload=False,
            )
        else:
            logger.info("YOLO loading via ultralytics/yolov5 hub: %s", self.config.weights_path)
            self._model = torch.hub.load(
                "ultralytics/yolov5",
                "custom",
                path=str(self.config.weights_path),
                force_reload=False,
            )
        self._model.to(self._device)
        self._model.eval()
        if self.config.use_fp16 and self._device.type == "cuda":
            self._model.half()
        names = getattr(self._model, "names", None)
        if isinstance(names, dict):
            self._names = {int(k): str(v) for k, v in names.items()}
        self._backend = "hub"
        logger.info("YOLO ready (hub) classes=%s device=%s", self._names, self._device)

    def _load_detect_multi_backend(self, torch: Any) -> None:
        root = self.config.yolov5_root
        if root is None or not (root / "models" / "common.py").is_file():
            raise FileNotFoundError(
                "TensorRT/ONNX weights need yolo_yolov5_root pointing at a yolov5 "
                "clone (e.g. from ApexAimBot). For .pt only, leave yolo_yolov5_root empty."
            )
        r = str(root.resolve())
        if r not in sys.path:
            sys.path.insert(0, r)
        from models.common import DetectMultiBackend  # type: ignore[import-not-found]

        self._model = DetectMultiBackend(
            str(self.config.weights_path),
            device=self._device,
            dnn=False,
            data=None,
            fp16=self.config.use_fp16,
        )
        self._model.eval()
        self._stride = int(self._model.stride)
        self._names = self._model.names if isinstance(self._model.names, dict) else {}
        self._backend = "dmb"
        imgsz = self.config.inference_size
        self._model.warmup(imgsz=(1, 3, imgsz, imgsz))
        logger.info("YOLO ready (DetectMultiBackend) classes=%s", self._names)

    def infer(self, frame_bgr: np.ndarray) -> list[YoloDetection]:
        if self._backend == "hub":
            return self._infer_hub(frame_bgr)
        return self._infer_dmb(frame_bgr)

    def _infer_hub(self, frame_bgr: np.ndarray) -> list[YoloDetection]:
        assert self._model is not None
        results = self._model(frame_bgr, size=self.config.inference_size)
        df = results.pandas().xyxy[0]
        if df is None or df.empty:
            return []
        out: list[YoloDetection] = []
        for row in df.itertuples(index=False):
            conf = float(row[4]) if len(row) > 4 else 1.0
            if conf < self.config.confidence_min:
                continue
            name = str(row[5]).lower() if len(row) > 5 else ""
            if name in self.config.exclude_labels:
                continue
            out.append(
                YoloDetection(
                    float(row[0]),
                    float(row[1]),
                    float(row[2]),
                    float(row[3]),
                    conf,
                    name,
                )
            )
        return out

    def _infer_dmb(self, frame_bgr: np.ndarray) -> list[YoloDetection]:
        import torch
        from utils.augmentations import letterbox  # type: ignore[import-not-found]
        from utils.general import non_max_suppression, scale_boxes  # type: ignore[import-not-found]

        assert self._model is not None
        stride = self._stride
        imgsz = self.config.inference_size
        im = letterbox(frame_bgr, imgsz, stride=stride, auto=True)[0]
        im = im.transpose((2, 0, 1))[::-1]
        im = np.ascontiguousarray(im)
        im_t = torch.from_numpy(im).to(self._device)
        im_t = im_t.half() if self.config.use_fp16 else im_t.float()
        im_t /= 255.0
        if im_t.ndimension() == 3:
            im_t = im_t.unsqueeze(0)
        pred = self._model(im_t, augment=False, visualize=False)
        pred = non_max_suppression(
            pred,
            self.config.confidence_min,
            self.config.iou_thres,
            max_det=self.config.max_det,
        )
        out: list[YoloDetection] = []
        for det in pred:
            if det is None or not len(det):
                continue
            det = det.clone()
            det[:, :4] = scale_boxes(im_t.shape[2:], det[:, :4], frame_bgr.shape).round()
            for *xyxy, conf, cls in det.tolist():
                name = str(self._names.get(int(cls), int(cls))).lower()
                if name in self.config.exclude_labels:
                    continue
                out.append(
                    YoloDetection(
                        float(xyxy[0]),
                        float(xyxy[1]),
                        float(xyxy[2]),
                        float(xyxy[3]),
                        float(conf),
                        name,
                    )
                )
        return out

    def find_best_target(
        self,
        frame_bgr: np.ndarray,
        fov_radius: float,
        min_area: float,
        fov_center_x: float,
        fov_center_y: float,
        *,
        sticky_target: Target | None = None,
        stickiness_pixels: float = 90.0,
        min_height_px: float = 0.0,
        min_confidence: float = 0.30,
        body_shape_min_score: float = 0.40,
        currently_locked: bool = False,
        ads_active: bool = False,
        debug: bool = False,
    ) -> DetectionResult:
        dbg: list[str] = ["mode=yolo"]
        h, w = frame_bgr.shape[:2]
        cx = float(fov_center_x)
        cy = float(fov_center_y)
        fov_r = float(fov_radius)

        try:
            dets = self.infer(frame_bgr)
        except Exception as exc:
            dbg.append(f"yolo_infer_fail={exc}")
            return DetectionResult(None, 0, 0.0, debug_lines=dbg, active=False)

        dbg.append(f"yolo_raw={len(dets)}")

        candidates: list[Target] = []
        for d in dets:
            t = _yolo_det_to_target(d, cx, cy, self.config.aim_fraction)
            if t.area < min_area:
                continue
            if min_height_px > 0 and t.bbox_h < min_height_px:
                continue
            if t.distance_to_center > fov_r * 1.02:
                continue
            if t.body_shape_score < body_shape_min_score:
                continue
            candidates.append(t)

        if not candidates:
            dbg.append("yolo_no_candidates")
            return DetectionResult(None, 0, 0.0, debug_lines=dbg, active=False)

        def rank_fn(t: Target) -> float:
            return t.confidence * 1000.0 - t.distance_to_center * 0.5 + t.bbox_h * 0.2

        pick_mode = self.config.target_pick
        if sticky_target is not None and stickiness_pixels > 0:
            pool: list[Target] = []
            for t in candidates:
                iou = _bbox_iou(
                    sticky_target.bbox_x,
                    sticky_target.bbox_y,
                    sticky_target.bbox_w,
                    sticky_target.bbox_h,
                    t.bbox_x,
                    t.bbox_y,
                    t.bbox_w,
                    t.bbox_h,
                )
                dist = math.hypot(
                    t.centroid_x - sticky_target.centroid_x,
                    t.centroid_y - sticky_target.centroid_y,
                )
                lock_scale = self.config.ads_aim_scale if ads_active else 1.0
                max_dx = max(sticky_target.bbox_w, t.bbox_w) * lock_scale
                max_dy = max(sticky_target.bbox_h, t.bbox_h) * 0.5
                if iou >= 0.2 or dist <= stickiness_pixels:
                    if abs(t.centroid_x - cx) <= max_dx and abs(t.centroid_y - cy) <= max_dy:
                        pool.append(t)
            if pool:
                best = select_ranked_target(
                    pool,
                    rank_fn,
                    selection_mode=pick_mode,
                    fov_radius=fov_r,
                    body_shape_min=body_shape_min_score,
                )
                dbg.append(
                    f"yolo_sticky dist={best.distance_to_center:.0f} conf={best.confidence:.2f}"
                )
                return DetectionResult(
                    best,
                    len(candidates),
                    best.confidence,
                    debug_lines=dbg,
                    active=best.confidence >= min_confidence,
                )

        best = select_ranked_target(
            candidates,
            rank_fn,
            selection_mode=pick_mode,
            fov_radius=fov_r,
            body_shape_min=body_shape_min_score,
        )
        dbg.append(
            f"yolo_pick dist={best.distance_to_center:.0f} conf={best.confidence:.2f} "
            f"h={best.bbox_h} label_pool={len(candidates)}"
        )
        if debug:
            for i, t in enumerate(candidates[:5]):
                dbg.append(
                    f"  cand[{i}] conf={t.confidence:.2f} dist={t.distance_to_center:.0f} "
                    f"bbox={t.bbox_w}x{t.bbox_h}"
                )
        return DetectionResult(
            best,
            len(candidates),
            best.confidence,
            debug_lines=dbg,
            active=best.confidence >= min_confidence,
        )


def _yolo_det_to_target(d: YoloDetection, cx: float, cy: float, aim_fraction: float) -> Target:
    bw = max(1.0, d.xmax - d.xmin)
    bh = max(1.0, d.ymax - d.ymin)
    aim_x = (d.xmin + d.xmax) * 0.5
    aim_y = d.ymin + bh * aim_fraction
    area = bw * bh
    dist = math.hypot(aim_x - cx, aim_y - cy)
    aspect = bh / bw
    body = min(1.0, 0.40 + min(aspect, 3.0) * 0.12 + d.confidence * 0.35)
    return Target(
        centroid_x=aim_x,
        centroid_y=aim_y,
        area=area,
        distance_to_center=dist,
        confidence=d.confidence,
        bbox_x=int(d.xmin),
        bbox_y=int(d.ymin),
        bbox_w=int(bw),
        bbox_h=int(bh),
        solidity=0.75,
        humanoid_score=body,
        part_count=3,
        body_shape_score=body,
        head_score=min(1.0, body + 0.1),
        torso_score=min(1.0, body + 0.05),
        limb_stack_score=0.5,
        red_coverage=0.12,
        fill_ratio=0.65,
        max_circularity=0.45,
        has_classified_torso=True,
    )


def find_best_yolo_target(
    frame_bgr: np.ndarray,
    fov_radius: int,
    min_area: float,
    fov_center_x: float,
    fov_center_y: float,
    *,
    engine: YoloEngine,
    **kwargs: Any,
) -> DetectionResult:
    return engine.find_best_target(
        frame_bgr,
        float(fov_radius),
        float(min_area),
        fov_center_x,
        fov_center_y,
        **kwargs,
    )
