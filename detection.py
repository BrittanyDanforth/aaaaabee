"""HSV mask + Apex-style humanoid aim point inside FOV."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger("targeting")

_MAX_AREA_RATIO = 0.35
# Bottom of screen: weapon viewmodel / red-dot sights (common false positives).
_VIEWMODEL_EXCLUDE_FRAC = 0.30
# Reject aim points too high above crosshair (sky, overhead UI, wall targets).
_MAX_ABOVE_CENTER_FRAC = 0.42
_MIN_CONFIDENCE = 0.22
_STICKY_SWITCH_RATIO = 2.25


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


@dataclass
class DetectionResult:
    target: Target | None
    candidates: int
    best_score: float = 0.0


def _build_fov_mask(height: int, width: int, center_x: float, center_y: float, radius: int) -> np.ndarray:
    y, x = np.ogrid[:height, :width]
    cx = round(center_x)
    cy = round(center_y)
    dist_sq = (x - cx) ** 2 + (y - cy) ** 2
    return (dist_sq <= radius * radius).astype(np.uint8)


def _build_viewmodel_exclude_mask(height: int, width: int, exclude_bottom_frac: float) -> np.ndarray:
    """Zero out the bottom band where the local weapon model lives."""
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
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    combined = cv2.dilate(combined, kernel, iterations=1)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    return combined


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
    area_cap = min(math.sqrt(target.area), math.sqrt(fov_radius * fov_radius * 0.25))
    area_term = area_cap * area_weight
    solidity_bonus = target.solidity * fov_radius * 0.04
    aspect = target.bbox_h / max(target.bbox_w, 1) if target.bbox_w > 0 else 1.0
    shape_quality = min(1.0, max(0.0, (aspect - 1.0) / 1.5))
    shape_bonus = shape_quality * fov_radius * 0.22

    penalty = 0.0
    if center_y is not None and target.centroid_y < center_y:
        above = center_y - target.centroid_y
        penalty += above * vertical_penalty_weight
        if above > fov_radius * _MAX_ABOVE_CENTER_FRAC:
            penalty += fov_radius * 2.0

    if aspect < 1.05:
        penalty += fov_radius * 0.35
    if target.bbox_h > 0 and target.bbox_w > target.bbox_h * 1.2:
        penalty += fov_radius * 0.25

    return dist_term + area_term + solidity_bonus + shape_bonus - penalty


def _normalize_confidence(raw_score: float, fov_radius: float, distance_weight: float) -> float:
    denom = max(1.0, fov_radius * (distance_weight + 0.5))
    return max(0.0, min(1.0, raw_score / denom))


def _aim_point_from_bbox(x: int, y: int, w: int, h: int, torso_fraction: float) -> tuple[float, float]:
    """Upper-torso aim (Apex-style), not geometric centroid of full blob."""
    frac = max(0.2, min(0.55, torso_fraction))
    return x + w * 0.5, y + h * frac


def _merge_nearby_contours(
    contours: list[np.ndarray],
    merge_gap: int = 30,
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
        for j in range(i + 1, n):
            x2, y2, w2, h2 = bboxes[j]
            gap_x = max(0, max(x1, x2) - min(x1 + w1, x2 + w2))
            gap_y = max(0, max(y1, y2) - min(y1 + h1, y2 + h2))
            if gap_x <= merge_gap and gap_y <= merge_gap:
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
    min_aspect: float = 0.0,
    max_aspect: float = 0.0,
    min_solidity: float = 0.0,
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
    contours = _merge_nearby_contours(list(raw_contours), merge_gap=30)
    frame_area = float(h * w)
    max_blob_area = frame_area * _MAX_AREA_RATIO
    if min_height_px <= 0.0:
        min_height_px = max(14.0, h * 0.045)
    targets: list[Target] = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        if area > max_blob_area:
            continue
        bx, by, bw, bh = cv2.boundingRect(contour)
        if bw <= 0 or bh <= 0:
            continue
        aspect = bh / float(bw)
        if bh < min_height_px:
            continue
        if min_aspect > 0 and aspect < min_aspect:
            continue
        if max_aspect > 0 and aspect > max_aspect:
            continue
        solidity = area / float(bw * bh)
        if min_solidity > 0 and solidity < min_solidity:
            continue

        tx, ty = _aim_point_from_bbox(bx, by, bw, bh, torso_aim_fraction)
        if ty < cy - fov_radius * _MAX_ABOVE_CENTER_FRAC:
            continue

        dist = float(np.hypot(tx - cx, ty - cy))
        if dist > fov_radius:
            continue

        raw = score_target(
            Target(tx, ty, area, dist, 0.0, bw, bh, solidity),
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
    min_aspect: float = 1.12,
    max_aspect: float = 5.5,
    min_solidity: float = 0.28,
    torso_aim_fraction: float = 0.38,
    min_confidence: float = _MIN_CONFIDENCE,
    exclude_bottom_frac: float = _VIEWMODEL_EXCLUDE_FRAC,
) -> DetectionResult:
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
        min_aspect=min_aspect,
        max_aspect=max_aspect,
        min_solidity=min_solidity,
        torso_aim_fraction=torso_aim_fraction,
        distance_weight=distance_weight,
        area_weight=area_weight,
        exclude_bottom_frac=exclude_bottom_frac,
    )
    if not candidates:
        return DetectionResult(None, 0, 0.0)

    def pick_best(pool: list[Target]) -> Target:
        def key(t: Target) -> float:
            return score_target(
                t,
                float(fov_radius),
                distance_weight,
                area_weight,
                center_y=cy,
            )

        best = max(pool, key=key)
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
            f"conf={target.confidence:.2f} d={target.distance_to_center:.0f}",
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
