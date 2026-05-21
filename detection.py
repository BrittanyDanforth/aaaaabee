"""HSV mask + structure-aware humanoid detection (head/torso/limbs, not just red)."""

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
_MIN_CONFIDENCE = 0.34
_STICKY_SWITCH_RATIO = 2.35
_MIN_FIGURE_SCORE = 0.52

_MIN_HUMANOID_ASPECT = 1.35
_MAX_BBOX_WIDTH_FRAC = 0.28
_MIN_BBOX_HEIGHT_FRAC = 0.05
_MAX_BBOX_HEIGHT_FRAC = 0.65


@dataclass
class Target:
    """Aim point in capture-frame coordinates (upper-torso / head biased)."""

    centroid_x: float
    centroid_y: float
    area: float
    distance_to_center: float = 0.0
    confidence: float = 0.0
    bbox_w: int = 0
    bbox_h: int = 0
    solidity: float = 1.0
    humanoid_score: float = 0.0
    part_count: int = 0


@dataclass
class DetectionResult:
    target: Target | None
    candidates: int
    best_score: float = 0.0


@dataclass
class _RedPart:
    contour: np.ndarray
    x: int
    y: int
    w: int
    h: int
    area: float
    cx: float
    cy: float
    aspect: float
    solidity: float
    extent: float
    circularity: float


def _scale(frame_w: int, frame_h: int) -> float:
    return min(frame_w, frame_h) / 1080.0


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
    return combined


def _is_health_bar_part(part: _RedPart, scale: float) -> bool:
    return (
        part.aspect >= 4.0
        and part.h <= 14 * scale
        and part.w >= 36 * scale
        and part.area <= 900 * scale * scale
    )


def _is_sight_part(part: _RedPart, scale: float) -> bool:
    return (
        part.area <= 280 * scale * scale
        and part.aspect <= 1.9
        and part.solidity >= 0.94
    )


def _extract_red_parts(mask: np.ndarray, frame_w: int, frame_h: int) -> list[_RedPart]:
    scale = _scale(frame_w, frame_h)
    area_min = 90 * scale * scale
    area_max = 14000 * scale * scale
    parts: list[_RedPart] = []

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < area_min or area > area_max:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue
        if w > frame_w * 0.24 or h > frame_h * 0.38:
            continue

        aspect = w / float(h)
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / hull_area if hull_area > 0 else 0.0
        extent = area / float(w * h)
        perim = cv2.arcLength(contour, True)
        circularity = (4.0 * math.pi * area / (perim * perim)) if perim > 0 else 0.0

        part = _RedPart(
            contour=contour,
            x=x,
            y=y,
            w=w,
            h=h,
            area=area,
            cx=x + w * 0.5,
            cy=y + h * 0.5,
            aspect=aspect,
            solidity=solidity,
            extent=extent,
            circularity=circularity,
        )

        if _is_health_bar_part(part, scale):
            continue
        if _is_sight_part(part, scale):
            continue
        if (
            area >= 6500 * scale * scale
            and w > frame_w * 0.12
            and solidity >= 0.91
            and extent >= 0.76
        ):
            continue

        parts.append(part)

    return parts


def _cluster_bbox(parts: list[_RedPart]) -> tuple[int, int, int, int]:
    x0 = min(p.x for p in parts)
    y0 = min(p.y for p in parts)
    x1 = max(p.x + p.w for p in parts)
    y1 = max(p.y + p.h for p in parts)
    return x0, y0, x1 - x0, y1 - y0


def _x_overlap_ratio(a: _RedPart, b: _RedPart) -> float:
    left = max(a.x, b.x)
    right = min(a.x + a.w, b.x + b.w)
    overlap = max(0, right - left)
    return overlap / max(1, min(a.w, b.w))


def _cluster_parts(parts: list[_RedPart], frame_w: int) -> list[list[_RedPart]]:
    if not parts:
        return []

    scale = _scale(frame_w, max(480, frame_w * 9 // 16))
    x_tol = max(20.0, 0.14 * float(np.median([p.w for p in parts])))
    sorted_parts = sorted(parts, key=lambda p: p.cx)
    clusters: list[list[_RedPart]] = []

    for part in sorted_parts:
        placed = False
        for cluster in clusters:
            cl_cx = float(np.mean([p.cx for p in cluster]))
            cl_x0 = min(p.x for p in cluster)
            cl_x1 = max(p.x + p.w for p in cluster)
            overlap = max(0, min(part.x + part.w, cl_x1) - max(part.x, cl_x0))
            min_w = max(1, min(part.w, cl_x1 - cl_x0))
            if abs(part.cx - cl_cx) <= x_tol or overlap / min_w >= 0.28:
                cluster.append(part)
                placed = True
                break
        if not placed:
            clusters.append([part])

    return clusters


def _vertical_red_segmentation_score(mask: np.ndarray, x0: int, y0: int, w: int, h: int) -> float:
    """Apex dummies: red/white stripes vertically. Walls: solid red columns."""
    if w < 4 or h < 8:
        return 0.0
    roi = mask[y0 : y0 + h, x0 : x0 + w]
    if roi.size == 0:
        return 0.0
    col_x = w // 2
    column = roi[:, col_x]
    red = (column > 0).astype(np.int8)
    if red.sum() == 0:
        return 0.0
    fill_ratio = float(red.mean())
    transitions = int(np.sum(np.abs(np.diff(red))))
    if fill_ratio > 0.88 and transitions <= 2:
        return 0.0
    if transitions >= 3 and 0.22 <= fill_ratio <= 0.78:
        return 1.0
    if transitions >= 2 and fill_ratio <= 0.72:
        return 0.65
    return 0.2 if fill_ratio < 0.85 else 0.0


def _is_head_like_part(part: _RedPart, scale: float) -> bool:
    if _is_health_bar_part(part, scale):
        return False
    return (
        120 * scale * scale <= part.area <= 4200 * scale * scale
        and 0.42 <= part.aspect <= 2.6
        and 6 * scale <= part.w <= 95 * scale
        and 6 * scale <= part.h <= 70 * scale
        and (part.circularity >= 0.22 or 0.55 <= part.aspect <= 1.8)
    )


def _is_segmented_body_stack(parts: list[_RedPart], scale: float) -> bool:
    if len(parts) < 2:
        return False

    parts_sorted = sorted(parts, key=lambda p: p.y)
    cx_std = float(np.std([p.cx for p in parts_sorted]))
    if cx_std > 24 * scale:
        return False

    widths = [p.w for p in parts_sorted]
    mean_w = float(np.mean(widths))
    if mean_w > 0 and float(np.std(widths)) / mean_w > 0.72:
        return False

    gaps: list[float] = []
    for i in range(len(parts_sorted) - 1):
        bottom = parts_sorted[i].y + parts_sorted[i].h
        top = parts_sorted[i + 1].y
        gaps.append(float(top - bottom))

    if any(g < -6 * scale for g in gaps):
        return False
    if any(g > 150 * scale for g in gaps):
        return False

    span = (parts_sorted[-1].y + parts_sorted[-1].h) - parts_sorted[0].y
    if span < 28 * scale or span > 240 * scale:
        return False

    plate_like = 0
    for p in parts_sorted:
        ar = p.h / max(p.w, 1)
        if 0.4 <= ar <= 3.2 and p.area >= 100 * scale * scale:
            plate_like += 1
    if plate_like < 2:
        return False

    return len(parts_sorted) >= 3 or (len(parts_sorted) == 2 and plate_like == 2)


def _figure_aim_point(
    parts: list[_RedPart], bx: int, by: int, bw: int, bh: int, frame_w: int, frame_h: int
) -> tuple[float, float]:
    parts_sorted = sorted(parts, key=lambda p: p.y)
    scale = _scale(frame_w, frame_h)
    if _is_head_like_part(parts_sorted[0], scale):
        head = parts_sorted[0]
        return head.cx, head.cy + head.h * 0.15

    torso_idx = 0
    if len(parts_sorted) >= 2:
        areas = [p.area for p in parts_sorted]
        mid = int(np.argmax(areas))
        torso_idx = mid
    torso = parts_sorted[torso_idx]
    return torso.cx, torso.y + torso.h * 0.38


def _validate_figure(
    parts: list[_RedPart],
    mask: np.ndarray,
    frame_w: int,
    frame_h: int,
) -> tuple[bool, float, int, int, int, int, float, int]:
    scale = _scale(frame_w, frame_h)
    bx, by, bw, bh = _cluster_bbox(parts)
    if bw <= 0 or bh <= 0:
        return False, 0.0, 0, 0, 0, 0, 0.0, 0

    aspect = bh / float(bw)
    if aspect < _MIN_HUMANOID_ASPECT:
        return False, 0.0, bx, by, bw, bh, 0.0, len(parts)
    if bw >= bh:
        return False, 0.0, bx, by, bw, bh, 0.0, len(parts)
    if bw > frame_w * _MAX_BBOX_WIDTH_FRAC:
        return False, 0.0, bx, by, bw, bh, 0.0, len(parts)
    if bh < frame_h * _MIN_BBOX_HEIGHT_FRAC or bh > frame_h * _MAX_BBOX_HEIGHT_FRAC:
        return False, 0.0, bx, by, bw, bh, 0.0, len(parts)
    if bw / max(bh, 1) < 0.18:
        return False, 0.0, bx, by, bw, bh, 0.0, len(parts)

    parts_sorted = sorted(parts, key=lambda p: p.y)
    has_head = _is_head_like_part(parts_sorted[0], scale)
    segmented = _is_segmented_body_stack(parts, scale)
    seg_score = _vertical_red_segmentation_score(mask, bx, by, bw, bh)

    if not has_head and not segmented:
        return False, 0.0, bx, by, bw, bh, 0.0, len(parts)
    if len(parts) == 1 and parts_sorted[0].extent >= 0.72 and parts_sorted[0].area >= 3500 * scale * scale:
        return False, 0.0, bx, by, bw, bh, 0.0, len(parts)

    score = 0.0
    if has_head:
        score += 0.36
    if segmented:
        score += 0.34
    score += seg_score * 0.22
    if len(parts) >= 3:
        score += 0.14
    elif len(parts) == 2:
        score += 0.07
    cx_std = float(np.std([p.cx for p in parts]))
    score += 0.08 * max(0.0, 1.0 - cx_std / (26 * scale))
    if aspect >= 1.6:
        score += 0.06

    total_area = sum(p.area for p in parts)
    solidity = total_area / float(bw * bh)
    return score >= _MIN_FIGURE_SCORE, score, bx, by, bw, bh, solidity, len(parts)


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

    parts = _extract_red_parts(mask, w, h)
    clusters = _cluster_parts(parts, w)
    frame_area = float(h * w)
    max_blob_area = frame_area * _MAX_AREA_RATIO
    targets: list[Target] = []

    for cluster in clusters:
        total_area = sum(p.area for p in cluster)
        if total_area < min_area:
            continue
        if total_area > max_blob_area:
            continue

        ok, fig_score, bx, by, bw, bh, solidity, part_count = _validate_figure(
            cluster, mask, w, h
        )
        if not ok:
            continue

        tx, ty = _figure_aim_point(cluster, bx, by, bw, bh, w, h)
        if ty < cy - fov_radius * _MAX_ABOVE_CENTER_FRAC:
            continue

        dist = float(np.hypot(tx - cx, ty - cy))
        if dist > fov_radius:
            continue

        target = Target(
            centroid_x=tx,
            centroid_y=ty,
            area=total_area,
            distance_to_center=dist,
            bbox_w=bw,
            bbox_h=bh,
            solidity=solidity,
            humanoid_score=fig_score,
            part_count=part_count,
        )
        raw = score_target(
            target,
            float(fov_radius),
            distance_weight,
            area_weight,
            center_y=cy,
        )
        if raw <= 0.0:
            continue

        target.confidence = _normalize_confidence(raw, float(fov_radius), distance_weight)
        targets.append(target)

    return targets


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
    shape_bonus = target.humanoid_score * fov_radius * 0.42
    part_bonus = min(target.part_count, 4) * fov_radius * 0.04

    penalty = 0.0
    if center_y is not None and target.centroid_y < center_y:
        above = center_y - target.centroid_y
        penalty += above * vertical_penalty_weight
        if above > fov_radius * _MAX_ABOVE_CENTER_FRAC:
            penalty += fov_radius * 2.5
    if target.bbox_w > 0 and target.bbox_w >= target.bbox_h:
        penalty += fov_radius * 1.5

    return dist_term + area_term + shape_bonus + part_bonus - penalty


def _normalize_confidence(raw_score: float, fov_radius: float, distance_weight: float) -> float:
    denom = max(1.0, fov_radius * (distance_weight + 0.7))
    return max(0.0, min(1.0, raw_score / denom))


def find_best_target(
    frame_bgr: np.ndarray,
    hsv_ranges: list[dict[str, Any]],
    fov_radius: int,
    min_area: float,
    fov_center_x: float | None = None,
    fov_center_y: float | None = None,
    *,
    sticky_target: Target | None = None,
    stickiness_pixels: float = 60.0,
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
    _ = (min_aspect, max_aspect, min_solidity, min_height_px, torso_aim_fraction)
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
                t, float(fov_radius), distance_weight, area_weight, center_y=cy
            ),
        )
        raw = score_target(
            best, float(fov_radius), distance_weight, area_weight, center_y=cy
        )
        best.confidence = _normalize_confidence(raw, float(fov_radius), distance_weight)
        return best

    def _score(t: Target) -> float:
        return score_target(t, float(fov_radius), distance_weight, area_weight, center_y=cy)

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
            global_closer = global_best.distance_to_center < sticky_best.distance_to_center - 10.0
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


# Legacy hook for tests
def _is_humanoid_contour(contour: np.ndarray, frame_w: int, frame_h: int) -> tuple[bool, int, int, float, float]:
    x, y, w, h = cv2.boundingRect(contour)
    part = _RedPart(
        contour=contour,
        x=x,
        y=y,
        w=w,
        h=h,
        area=float(cv2.contourArea(contour)),
        cx=x + w * 0.5,
        cy=y + h * 0.5,
        aspect=w / max(h, 1),
        solidity=0.5,
        extent=0.5,
        circularity=0.3,
    )
    scale = _scale(frame_w, frame_h)
    ok = (
        _is_head_like_part(part, scale)
        or (h / max(w, 1) >= _MIN_HUMANOID_ASPECT and w < h)
    )
    return ok, w, h, 0.5, 0.5 if ok else 0.0


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
            f"conf={target.confidence:.2f} parts={target.part_count}",
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
