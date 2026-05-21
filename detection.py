"""HSV mask + Apex-style humanoid aim point inside FOV."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger("targeting")

_MAX_AREA_RATIO = 0.28
_VIEWMODEL_EXCLUDE_FRAC = 0.30
_MAX_ABOVE_CENTER_FRAC = 0.38
_MIN_CONFIDENCE = 0.32
_STICKY_SWITCH_RATIO = 2.35

# Standing humanoid: clearly taller than wide, never a screen-wide flat band.
_MIN_HUMANOID_ASPECT = 1.48
_MAX_HUMANOID_ASPECT = 6.0
_MAX_BBOX_WIDTH_FRAC = 0.30
_MIN_BBOX_HEIGHT_FRAC = 0.06
_MAX_BBOX_HEIGHT_FRAC = 0.62
_MAX_CONVEXITY_PANEL = 0.94
_MAX_STRIPE_SOLIDITY = 0.86


@dataclass
class Target:
    """Aim point in capture-frame coordinates (upper-torso biased)."""

    centroid_x: float
    centroid_y: float
    area: float
    distance_to_center: float = 0.0
    confidence: float = 0.0
    bbox_w: int = 0
    bbox_h: int = 0
    solidity: float = 1.0
    humanoid_score: float = 0.0


@dataclass
class DetectionResult:
    target: Target | None
    candidates: int
    best_score: float = 0.0


@dataclass(frozen=True)
class _ShapeMetrics:
    aspect: float
    solidity: float
    convexity: float
    width_frac: float
    height_frac: float
    humanoid_score: float


def _build_fov_mask(height: int, width: int, center_x: float, center_y: float, radius: int) -> np.ndarray:
    y, x = np.ogrid[:height, :width]
    cx = round(center_x)
    cy = round(center_y)
    dist_sq = (x - cx) ** 2 + (y - cy) ** 2
    return (dist_sq <= radius * radius).astype(np.uint8)


def _build_viewmodel_exclude_mask(height: int, width: int, exclude_bottom_frac: float) -> np.ndarray:
    frac = max(0.0, min(0.5, exclude_bottom_frac))
    if frac <= 0.0:
        return np.ones((height, width), dtype=np.uint8)
    cutoff = int(height * (1.0 - frac))
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[:cutoff, :] = 1
    return mask


def build_hsv_mask(frame_bgr: np.ndarray, hsv_ranges: list[dict[str, Any]]) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    combined = None
    for entry in hsv_ranges:
        lower = np.array(entry["lower"], dtype=np.uint8)
        upper = np.array(entry["upper"], dtype=np.uint8)
        part = cv2.inRange(hsv, lower, upper)
        combined = part if combined is None else cv2.bitwise_or(combined, part)
    if combined is None:
        return np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    return combined


def _shape_metrics(
    contour: np.ndarray,
    frame_w: int,
    frame_h: int,
) -> _ShapeMetrics | None:
    area = float(cv2.contourArea(contour))
    if area < 1.0:
        return None
    bx, by, bw, bh = cv2.boundingRect(contour)
    if bw <= 0 or bh <= 0:
        return None

    aspect = bh / float(bw)
    solidity = area / float(bw * bh)
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    convexity = area / hull_area if hull_area > 0 else 0.0
    width_frac = bw / float(frame_w)
    height_frac = bh / float(frame_h)

    score = 0.0
    score += min(1.0, max(0.0, (aspect - 1.2) / 2.2)) * 0.42
    score += min(1.0, max(0.0, (convexity - 0.55) / 0.35)) * 0.18
    score += min(1.0, max(0.0, 1.0 - width_frac / _MAX_BBOX_WIDTH_FRAC)) * 0.22
    score += min(1.0, solidity * 1.1) * 0.08
    if 0.08 <= height_frac <= 0.45:
        score += 0.10

    return _ShapeMetrics(
        aspect=aspect,
        solidity=solidity,
        convexity=convexity,
        width_frac=width_frac,
        height_frac=height_frac,
        humanoid_score=score,
    )


def _is_humanoid_contour(
    contour: np.ndarray,
    frame_w: int,
    frame_h: int,
) -> tuple[bool, int, int, float, float]:
    """
    Reject flat architecture: horizontal stripes, screen-wide bands, convex panels.
    Accept only blobs that look like a standing character (taller-than-wide, limited width).
    """
    metrics = _shape_metrics(contour, frame_w, frame_h)
    if metrics is None:
        return False, 0, 0, 0.0, 0.0

    bx, by, bw, bh = cv2.boundingRect(contour)
    aspect = metrics.aspect

    if aspect < _MIN_HUMANOID_ASPECT:
        return False, bw, bh, metrics.solidity, metrics.humanoid_score
    if aspect > _MAX_HUMANOID_ASPECT:
        return False, bw, bh, metrics.solidity, metrics.humanoid_score

    if bw >= bh:
        return False, bw, bh, metrics.solidity, metrics.humanoid_score

    if metrics.width_frac > _MAX_BBOX_WIDTH_FRAC:
        return False, bw, bh, metrics.solidity, metrics.humanoid_score

    if metrics.height_frac < _MIN_BBOX_HEIGHT_FRAC:
        return False, bw, bh, metrics.solidity, metrics.humanoid_score
    if metrics.height_frac > _MAX_BBOX_HEIGHT_FRAC:
        return False, bw, bh, metrics.solidity, metrics.humanoid_score

    if metrics.solidity > _MAX_STRIPE_SOLIDITY and aspect < 2.0:
        return False, bw, bh, metrics.solidity, metrics.humanoid_score

    if metrics.convexity > _MAX_CONVEXITY_PANEL and aspect < 2.4:
        return False, bw, bh, metrics.solidity, metrics.humanoid_score

    if len(contour) >= 5:
        try:
            (_ex, _ey), (axis_major, axis_minor), angle = cv2.fitEllipse(contour)
            if axis_major > 1.0 and axis_minor > 1.0:
                norm_angle = abs(angle) % 180.0
                horizontal = norm_angle < 32.0 or norm_angle > 148.0
                if horizontal and axis_major > axis_minor * 1.35 and bw > bh * 1.1:
                    return False, bw, bh, metrics.solidity, metrics.humanoid_score
        except cv2.error:
            pass

    perimeter = cv2.arcLength(contour, True)
    if perimeter > 0:
        circularity = 4.0 * math.pi * float(cv2.contourArea(contour)) / (perimeter * perimeter)
        if circularity < 0.08 and metrics.solidity > 0.8:
            return False, bw, bh, metrics.solidity, metrics.humanoid_score

    if metrics.humanoid_score < 0.38:
        return False, bw, bh, metrics.solidity, metrics.humanoid_score

    return True, bw, bh, metrics.solidity, metrics.humanoid_score


def score_target(
    target: Target,
    fov_radius: float,
    distance_weight: float,
    area_weight: float,
    *,
    center_y: float | None = None,
    vertical_penalty_weight: float = 1.35,
) -> float:
    dist_term = max(0.0, fov_radius - target.distance_to_center) * distance_weight
    area_cap = min(math.sqrt(target.area), math.sqrt(fov_radius * fov_radius * 0.22))
    area_term = area_cap * area_weight
    solidity_bonus = target.solidity * fov_radius * 0.03
    aspect = target.bbox_h / max(target.bbox_w, 1) if target.bbox_w > 0 else 1.0
    shape_quality = min(1.0, max(0.0, (aspect - _MIN_HUMANOID_ASPECT) / 1.8))
    shape_bonus = shape_quality * fov_radius * 0.28
    humanoid_bonus = target.humanoid_score * fov_radius * 0.35

    penalty = 0.0
    if center_y is not None and target.centroid_y < center_y:
        above = center_y - target.centroid_y
        penalty += above * vertical_penalty_weight
        if above > fov_radius * _MAX_ABOVE_CENTER_FRAC:
            penalty += fov_radius * 2.5

    if target.bbox_w > 0 and target.bbox_w >= target.bbox_h:
        penalty += fov_radius * 1.2

    return dist_term + area_term + solidity_bonus + shape_bonus + humanoid_bonus - penalty


def _normalize_confidence(raw_score: float, fov_radius: float, distance_weight: float) -> float:
    denom = max(1.0, fov_radius * (distance_weight + 0.65))
    return max(0.0, min(1.0, raw_score / denom))


def _aim_point_from_bbox(x: int, y: int, w: int, h: int, torso_fraction: float) -> tuple[float, float]:
    frac = max(0.2, min(0.55, torso_fraction))
    return x + w * 0.5, y + h * frac


def _merge_nearby_contours(
    contours: list[np.ndarray],
    merge_gap: int = 18,
) -> list[np.ndarray]:
    if len(contours) <= 1:
        return contours
    bboxes = [cv2.boundingRect(c) for c in contours]
    n = len(contours)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        x1, y1, w1, h1 = bboxes[i]
        a1 = h1 / max(w1, 1)
        for j in range(i + 1, n):
            x2, y2, w2, h2 = bboxes[j]
            gap_x = max(0, max(x1, x2) - min(x1 + w1, x2 + w2))
            gap_y = max(0, max(y1, y2) - min(y1 + h1, y2 + h2))
            if gap_x > merge_gap or gap_y > merge_gap:
                continue
            a2 = h2 / max(w2, 1)
            if (a1 < 1.1 and a2 > 1.5) or (a2 < 1.1 and a1 > 1.5):
                continue
            union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        r = find(i)
        groups.setdefault(r, []).append(i)

    merged: list[np.ndarray] = []
    for idxs in groups.values():
        if len(idxs) == 1:
            merged.append(contours[idxs[0]])
        else:
            pts = np.vstack([contours[i] for i in idxs])
            merged.append(pts)
    return merged


def _collect_targets(
    frame_bgr: np.ndarray,
    hsv_ranges: list[dict[str, Any]],
    fov_radius: int,
    min_area: float,
    cx: float,
    cy: float,
    *,
    min_height_px: float = 0.0,
    torso_aim_fraction: float = 0.38,
    distance_weight: float = 2.0,
    area_weight: float = 0.02,
    exclude_bottom_frac: float = _VIEWMODEL_EXCLUDE_FRAC,
) -> list[Target]:
    h, w = frame_bgr.shape[:2]
    mask = build_hsv_mask(frame_bgr, hsv_ranges)
    fov = _build_fov_mask(h, w, cx, cy, fov_radius)
    viewmodel = _build_viewmodel_exclude_mask(h, w, exclude_bottom_frac)
    mask = cv2.bitwise_and(mask, mask, mask=fov)
    mask = cv2.bitwise_and(mask, mask, mask=viewmodel)

    raw_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = _merge_nearby_contours(list(raw_contours), merge_gap=18)
    frame_area = float(h * w)
    max_blob_area = frame_area * _MAX_AREA_RATIO
    min_h = min_height_px if min_height_px > 0 else max(18.0, h * _MIN_BBOX_HEIGHT_FRAC * 0.85)
    targets: list[Target] = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        if area > max_blob_area:
            continue

        ok, bw, bh, solidity, humanoid_score = _is_humanoid_contour(contour, w, h)
        if not ok:
            continue

        bx, by, _bw, _bh = cv2.boundingRect(contour)
        tx, ty = _aim_point_from_bbox(bx, by, bw, bh, torso_aim_fraction)
        if ty < cy - fov_radius * _MAX_ABOVE_CENTER_FRAC:
            continue

        dist = float(np.hypot(tx - cx, ty - cy))
        if dist > fov_radius:
            continue

        raw = score_target(
            Target(tx, ty, area, dist, 0.0, bw, bh, solidity, humanoid_score),
            float(fov_radius),
            distance_weight,
            area_weight,
            center_y=cy,
        )
        if raw <= 0.0:
            continue

        targets.append(
            Target(
                centroid_x=tx,
                centroid_y=ty,
                area=area,
                distance_to_center=dist,
                confidence=_normalize_confidence(raw, float(fov_radius), distance_weight),
                bbox_w=bw,
                bbox_h=bh,
                solidity=solidity,
                humanoid_score=humanoid_score,
            )
        )
    return targets


def find_best_target(
    frame_bgr: np.ndarray,
    hsv_ranges: list[dict[str, Any]],
    fov_radius: int,
    min_area: float,
    fov_center_x: float | None = None,
    fov_center_y: float | None = None,
    *,
    sticky_target: Target | None = None,
    stickiness_pixels: float = 55.0,
    distance_weight: float = 2.0,
    area_weight: float = 0.02,
    min_height_px: float = 0.0,
    min_aspect: float | None = None,
    max_aspect: float | None = None,
    min_solidity: float | None = None,
    torso_aim_fraction: float = 0.38,
    min_confidence: float = _MIN_CONFIDENCE,
    exclude_bottom_frac: float = _VIEWMODEL_EXCLUDE_FRAC,
) -> DetectionResult:
    _ = (min_aspect, max_aspect, min_solidity)
    h, w = frame_bgr.shape[:2]
    cx = w / 2.0 if fov_center_x is None else fov_center_x
    cy = h / 2.0 if fov_center_y is None else fov_center_y

    candidates = _collect_targets(
        frame_bgr,
        hsv_ranges,
        fov_radius,
        min_area,
        cx,
        cy,
        min_height_px=min_height_px,
        torso_aim_fraction=torso_aim_fraction,
        distance_weight=distance_weight,
        area_weight=area_weight,
        exclude_bottom_frac=exclude_bottom_frac,
    )
    if not candidates:
        return DetectionResult(None, 0, 0.0)

    def pick_best(pool: list[Target]) -> Target:
        best = max(
            pool,
            key=lambda t: score_target(
                t,
                float(fov_radius),
                distance_weight,
                area_weight,
                center_y=cy,
            ),
        )
        raw = score_target(
            best,
            float(fov_radius),
            distance_weight,
            area_weight,
            center_y=cy,
        )
        best.confidence = _normalize_confidence(raw, float(fov_radius), distance_weight)
        return best

    def _score(t: Target) -> float:
        return score_target(
            t,
            float(fov_radius),
            distance_weight,
            area_weight,
            center_y=cy,
        )

    if sticky_target is not None and stickiness_pixels > 0:
        sticky_pool = [
            t
            for t in candidates
            if math.hypot(
                t.centroid_x - sticky_target.centroid_x,
                t.centroid_y - sticky_target.centroid_y,
            )
            <= stickiness_pixels
        ]
        if sticky_pool:
            sticky_best = min(
                sticky_pool,
                key=lambda t: math.hypot(
                    t.centroid_x - sticky_target.centroid_x,
                    t.centroid_y - sticky_target.centroid_y,
                ),
            )
            global_best = pick_best(candidates)
            sticky_score = _score(sticky_best)
            global_score = _score(global_best)
            global_closer = global_best.distance_to_center < sticky_best.distance_to_center - 8.0
            if global_score > sticky_score * _STICKY_SWITCH_RATIO and global_closer:
                chosen = global_best
            else:
                chosen = sticky_best
            if chosen.confidence < min_confidence:
                return DetectionResult(None, len(candidates), chosen.confidence)
            return DetectionResult(chosen, len(candidates), chosen.confidence)

    best = pick_best(candidates)
    if best.confidence < min_confidence:
        return DetectionResult(None, len(candidates), best.confidence)
    return DetectionResult(best, len(candidates), best.confidence)


def draw_debug(
    frame_bgr: np.ndarray,
    target: Target | None,
    fov_radius: int,
    fov_center_x: float | None = None,
    fov_center_y: float | None = None,
    magnetism_radius: int | None = None,
    hsv_ranges: list[dict[str, Any]] | None = None,
    stats_lines: list[str] | None = None,
) -> np.ndarray:
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    cx = int(round(w / 2 if fov_center_x is None else fov_center_x))
    cy = int(round(h / 2 if fov_center_y is None else fov_center_y))

    if hsv_ranges:
        cx_f = w / 2.0 if fov_center_x is None else float(fov_center_x)
        cy_f = h / 2.0 if fov_center_y is None else float(fov_center_y)
        mask = build_hsv_mask(frame_bgr, hsv_ranges)
        fov = _build_fov_mask(h, w, cx_f, cy_f, fov_radius)
        viewmodel = _build_viewmodel_exclude_mask(h, w, _VIEWMODEL_EXCLUDE_FRAC)
        mask = cv2.bitwise_and(mask, mask, mask=fov)
        mask = cv2.bitwise_and(mask, mask, mask=viewmodel)
        tint = np.zeros_like(out)
        tint[:, :] = (0, 255, 0)
        out = np.where(mask[:, :, None] > 0, cv2.addWeighted(out, 0.55, tint, 0.45, 0), out)

    cv2.circle(out, (cx, cy), fov_radius, (0, 255, 0), 2)
    if magnetism_radius and magnetism_radius > 0:
        cv2.circle(out, (cx, cy), magnetism_radius, (0, 180, 255), 1)
    cv2.drawMarker(out, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 12, 2)
    if target is not None:
        tx, ty = int(target.centroid_x), int(target.centroid_y)
        cv2.circle(out, (tx, ty), 8, (0, 0, 255), 2)
        cv2.line(out, (cx, cy), (tx, ty), (255, 0, 255), 1)
        cv2.putText(
            out,
            f"conf={target.confidence:.2f} hum={target.humanoid_score:.2f}",
            (tx + 8, ty - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 220, 255),
            1,
        )
    if stats_lines:
        y0 = 22
        for i, line in enumerate(stats_lines[:6]):
            cv2.putText(
                out,
                line,
                (8, y0 + i * 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
    return out
