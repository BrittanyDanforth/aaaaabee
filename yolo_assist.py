"""Optional YOLOv5 assist (VALOAI-style) — second opinion for CV detection.

Requires PyTorch + weights. VALOAI ships ``best640.pt`` trained for Valorant
(purple highlight); it will NOT work on Apex red-outline enemies without your
own weights. When disabled or unavailable, runtime uses CV-only detection.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from tracking_fusion import ExternalBox

logger = logging.getLogger("targeting")

_torch = None
_hub_loaded = False


def _torch_available() -> bool:
    global _torch
    if _torch is not None:
        return True
    try:
        import torch as t

        _torch = t
        return True
    except ImportError:
        return False


class YoloAssist:
    """Thin wrapper around YOLOv5 custom weights (valoai layout)."""

    def __init__(
        self,
        *,
        weights_path: str | Path,
        yolov5_root: str | Path | None = None,
        inference_size: int = 320,
        confidence_min: float = 0.35,
        device: str = "auto",
    ) -> None:
        if not _torch_available():
            raise RuntimeError("PyTorch not installed — pip install torch or disable yolo_assist")
        self.weights_path = Path(weights_path)
        if not self.weights_path.is_file():
            raise FileNotFoundError(f"yolo weights not found: {self.weights_path}")
        self.inference_size = max(160, min(1280, int(inference_size)))
        self.confidence_min = max(0.05, min(0.99, float(confidence_min)))
        self._model = None
        self._yolov5_root = Path(yolov5_root) if yolov5_root else None
        self._device = device

    def _load(self) -> None:
        if self._model is not None:
            return
        torch = _torch
        assert torch is not None
        root = self._yolov5_root
        if root is None:
            for candidate in (
                Path("HWIDTool") / ".." / "v" / "scripts" / "yolov5-master",
                Path("v") / "scripts" / "yolov5-master",
            ):
                if (candidate / "hubconf.py").is_file():
                    root = candidate.resolve()
                    break
        if root is None or not (root / "hubconf.py").is_file():
            raise FileNotFoundError(
                "yolov5-master not found — clone valoai or set yolo_yolov5_root in config"
            )
        dev = self._device
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("YoloAssist loading %s on %s", self.weights_path, dev)
        self._model = torch.hub.load(
            str(root),
            "custom",
            path=str(self.weights_path),
            source="local",
            force_reload=False,
        )
        if dev == "cuda":
            self._model = self._model.cuda()
        self._model.eval()

    def detect(self, frame_bgr: Any) -> list[ExternalBox]:
        """Run YOLO on a BGR crop; return boxes in frame coordinates."""
        self._load()
        assert self._model is not None
        results = self._model(frame_bgr, size=self.inference_size)
        df = results.pandas().xyxy[0]
        if df is None or df.empty:
            return []
        out: list[ExternalBox] = []
        for row in df.itertuples(index=False):
            xmin, ymin, xmax, ymax = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            conf = float(row[4]) if len(row) > 4 else 1.0
            if conf < self.confidence_min:
                continue
            label = str(row[5]) if len(row) > 5 else ""
            out.append(
                ExternalBox(
                    xmin=xmin,
                    ymin=ymin,
                    xmax=xmax,
                    ymax=ymax,
                    confidence=conf,
                    label=label,
                )
            )
        out.sort(key=lambda b: b.confidence, reverse=True)
        return out


def try_create_yolo_assist(cfg: dict[str, Any]) -> YoloAssist | None:
    """Construct YoloAssist from config; return None if disabled or unavailable."""
    if not bool(cfg.get("yolo_assist_enabled", False)):
        return None
    weights = str(cfg.get("yolo_weights_path", "") or "").strip()
    if not weights:
        logger.warning("yolo_assist_enabled but yolo_weights_path is empty — disabled")
        return None
    try:
        root = cfg.get("yolo_yolov5_root")
        return YoloAssist(
            weights_path=weights,
            yolov5_root=str(root).strip() if root else None,
            inference_size=int(cfg.get("yolo_inference_size", 320)),
            confidence_min=float(cfg.get("yolo_confidence_min", 0.35)),
            device=str(cfg.get("yolo_device", "auto")),
        )
    except Exception as exc:
        logger.warning("YoloAssist unavailable: %s", exc)
        return None
