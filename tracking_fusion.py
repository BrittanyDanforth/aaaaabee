"""VALOAI / YOLO-style fusion helpers for ABA target ranking.

VALOAI (https://github.com/amiliadreams/valoai) uses YOLOv5 on a center FOV
crop and aims at the first bbox center. ABA keeps Apex shape+red fusion as the
primary detector; this module adds optional *second opinion* boosts and a
nearest-crosshair selection mode without replacing ``detector.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class ExternalBox:
    """External detector output (YOLO / ONNX / custom)."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float
    confidence: float = 1.0
    label: str = ""

    @property
    def width(self) -> float:
        return max(0.0, self.xmax - self.xmin)

    @property
    def height(self) -> float:
        return max(0.0, self.ymax - self.ymin)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.xmin + self.xmax) * 0.5, (self.ymin + self.ymax) * 0.5


def bbox_iou(
    ax: float,
    ay: float,
    aw: float,
    ah: float,
    bx: float,
    by: float,
    bw: float,
    bh: float,
) -> float:
    """IoU for axis-aligned boxes (x, y, w, h)."""
    if aw <= 0.0 or ah <= 0.0 or bw <= 0.0 or bh <= 0.0:
        return 0.0
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0.0 else 0.0


def associate_boxes_iou(
    prev: Sequence[ExternalBox],
    curr: Sequence[ExternalBox],
    *,
    iou_threshold: float = 0.30,
) -> list[tuple[int, int, float]]:
    """Greedy IoU association (SORT-style baseline). Returns (prev_i, curr_i, iou)."""
    if not prev or not curr:
        return []
    pairs: list[tuple[int, int, float]] = []
    used_curr: set[int] = set()
    for pi, p in enumerate(prev):
        best_j = -1
        best_iou = 0.0
        for ci, c in enumerate(curr):
            if ci in used_curr:
                continue
            iou = bbox_iou(
                p.xmin,
                p.ymin,
                p.width,
                p.height,
                c.xmin,
                c.ymin,
                c.width,
                c.height,
            )
            if iou > best_iou:
                best_iou = iou
                best_j = ci
        if best_j >= 0 and best_iou >= iou_threshold:
            used_curr.add(best_j)
            pairs.append((pi, best_j, best_iou))
    return pairs


def fusion_boost_for_target(
    target: object,
    boxes: Sequence[ExternalBox],
    *,
    fov_radius: float,
    boost_scale: float = 0.30,
    min_iou: float = 0.28,
    min_box_conf: float = 0.25,
) -> float:
    """Score boost when an external box overlaps the CV target bbox."""
    if not boxes or fov_radius <= 0.0:
        return 0.0
    tx = float(getattr(target, "bbox_x", 0.0))
    ty = float(getattr(target, "bbox_y", 0.0))
    tw = float(getattr(target, "bbox_w", 0.0))
    th = float(getattr(target, "bbox_h", 0.0))
    best = 0.0
    for box in boxes:
        if box.confidence < min_box_conf:
            continue
        iou = bbox_iou(tx, ty, tw, th, box.xmin, box.ymin, box.width, box.height)
        if iou < min_iou:
            continue
        agree = iou * box.confidence
        best = max(best, agree)
    if best <= 0.0:
        return 0.0
    return float(fov_radius) * float(boost_scale) * best


def apply_external_box_fusion(
    candidates: Sequence[object],
    boxes: Sequence[ExternalBox],
    *,
    fov_radius: float,
    boost_scale: float = 0.30,
    min_iou: float = 0.28,
    min_box_conf: float = 0.25,
) -> dict[int, float]:
    """Return per-candidate fusion boosts keyed by ``id(target)``."""
    boosts: dict[int, float] = {}
    for t in candidates:
        b = fusion_boost_for_target(
            t,
            boxes,
            fov_radius=fov_radius,
            boost_scale=boost_scale,
            min_iou=min_iou,
            min_box_conf=min_box_conf,
        )
        if b > 0.0:
            boosts[id(t)] = b
    return boosts


def pick_nearest_crosshair(
    candidates: Sequence[T],
    *,
    min_body_shape: float = 0.40,
    distance_attr: str = "distance_to_center",
    body_attr: str = "body_shape_score",
) -> T | None:
    """VALOAI-style pick: closest to crosshair among body-qualified targets."""
    pool = [
        t
        for t in candidates
        if float(getattr(t, body_attr, 0.0)) >= min_body_shape
    ]
    if not pool:
        return None
    return min(pool, key=lambda t: float(getattr(t, distance_attr, 1e9)))


def select_ranked_target(
    candidates: Sequence[T],
    rank_fn: Callable[[T], float],
    *,
    selection_mode: str = "apex",
    fov_radius: float = 140.0,
    body_shape_min: float = 0.40,
) -> T:
    """Choose best candidate: default Apex rank or nearest-center (valoai-like)."""
    if not candidates:
        raise ValueError("select_ranked_target: empty candidates")
    mode = (selection_mode or "apex").strip().lower()
    if mode in ("nearest", "nearest_center", "valoai"):
        near = pick_nearest_crosshair(candidates, min_body_shape=body_shape_min)
        if near is not None:
            return near
    return max(candidates, key=rank_fn)


def valoai_style_aim_point(box: ExternalBox) -> tuple[float, float]:
    """Geometric bbox center (what valoai.py uses). ABA prefers chest band in detector."""
    return box.center


def summarize_boxes(boxes: Sequence[ExternalBox]) -> str:
    if not boxes:
        return "yolo:0"
    conf = max(b.confidence for b in boxes)
    return f"yolo:{len(boxes)} max_conf={conf:.2f}"
