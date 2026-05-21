"""Synthetic 1280x720 Apex detection test frame generators.

Each generator returns (frame_bgr, SyntheticFrameSpec). Pixel coordinates use
origin top-left, x right, y down. FOV center defaults to (640, 360), radius 200.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

# --- Shared constants (match tests/test_detection.py) ---
FRAME_W = 1280
FRAME_H = 720
CX = FRAME_W // 2
CY = FRAME_H // 2
FOV_RADIUS = 200
MIN_AREA = 60.0

RED = (0, 0, 255)
ORANGE_BAR = (0, 165, 255)  # Apex shield tone; H≈16, outside HSV_RED mask
BG_DARK = 40
HSV_RED = [
    {"lower": [0, 100, 100], "upper": [12, 255, 255]},
    {"lower": [170, 100, 100], "upper": [180, 255, 255]},
]


@dataclass(frozen=True)
class SyntheticFrameSpec:
    name: str
    expect_active: bool
    description: str
    pixel_layout: str
    pass_fail_rationale: str
    optional_assertions: dict[str, float | int | str] | None = None


def _blank(dark: int = 0) -> np.ndarray:
    return np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8) + dark


def _apex_dummy(
    frame: np.ndarray,
    cx: int,
    foot_y: int,
    *,
    scale: float = 1.0,
    crouch: bool = False,
    partial: bool = False,
    head_chest_gap: int | None = None,
    chest_knee_gap: int | None = None,
) -> dict[str, tuple[int, int, int, int]]:
    """Draw segmented plates; return bbox dict for spec documentation."""
    s = scale
    if crouch:
        head_h, chest_h, knee_h = int(20 * s), int(36 * s), int(16 * s)
        gaps = (
            head_chest_gap if head_chest_gap is not None else 6,
            chest_knee_gap if chest_knee_gap is not None else 8,
        )
    else:
        head_h, chest_h, knee_h = int(26 * s), int(44 * s), int(22 * s)
        gaps = (
            head_chest_gap if head_chest_gap is not None else 10,
            chest_knee_gap if chest_knee_gap is not None else 12,
        )
    head_w, chest_w, knee_w = int(28 * s), int(42 * s), int(24 * s)
    total_h = head_h + gaps[0] + chest_h + gaps[1] + (0 if partial else knee_h)
    hy = foot_y - total_h
    hx, hx1 = cx - head_w // 2, cx - head_w // 2 + head_w
    cv2.rectangle(frame, (hx, hy), (hx1, hy + head_h), RED, -1)
    cx0, cx1 = cx - chest_w // 2, cx - chest_w // 2 + chest_w
    cy0, cy1 = hy + head_h + gaps[0], hy + head_h + gaps[0] + chest_h
    cv2.rectangle(frame, (cx0, cy0), (cx1, cy1), RED, -1)
    boxes = {"head": (hx, hy, head_w, head_h), "chest": (cx0, cy0, chest_w, chest_h)}
    if not partial:
        kx, kx1 = cx - knee_w // 2, cx - knee_w // 2 + knee_w
        ky0, ky1 = cy1 + gaps[1], cy1 + gaps[1] + knee_h
        cv2.rectangle(frame, (kx, ky0), (kx1, ky1), RED, -1)
        boxes["knees"] = (kx, ky0, knee_w, knee_h)
    return boxes


def gen_noisy_lighting_gradient() -> tuple[np.ndarray, SyntheticFrameSpec]:
    """(1) Noisy lighting gradient background + centered dummy."""
    frame = _blank(0)
    for y in range(FRAME_H):
        v = int(22 + 38 * (y / FRAME_H))
        frame[y, :] = (v, v, v)
    rng = np.random.default_rng(7)
    frame = np.clip(frame.astype(np.int16) + rng.integers(-10, 11, frame.shape), 0, 255).astype(
        np.uint8
    )
    boxes = _apex_dummy(frame, CX, CY + 90, scale=1.0)
    spec = SyntheticFrameSpec(
        name="noisy_lighting_gradient",
        expect_active=True,
        description="Vertical gray gradient (22→60) with ±10 luma noise; dummy unchanged.",
        pixel_layout=(
            f"Background: per-row BGR ({22}..{60}). "
            f"Dummy foot_y={CY + 90}, scale=1.0 → "
            f"head {boxes['head']}, chest {boxes['chest']}, knees {boxes['knees']}."
        ),
        pass_fail_rationale=(
            "PASS: saturated RED plates stay inside HSV_RED; body structure (3 parts, "
            "fill≈0.83) survives lighting variation."
        ),
        optional_assertions={"min_body_shape_score": 0.40, "max_centroid_x_delta": 90},
    )
    return frame, spec


def gen_red_building_dummy_overlap() -> tuple[np.ndarray, SyntheticFrameSpec]:
    """(2) Red architecture panel behind dummy; 1px separation so contours stay distinct."""
    frame = _blank(BG_DARK)
    building = (480, 300, 138, 180)  # x, y, w, h — right edge x=618, dummy chest starts ~619
    bx, by, bw, bh = building
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), RED, -1)
    boxes = _apex_dummy(frame, CX, CY + 100, scale=1.1)
    spec = SyntheticFrameSpec(
        name="red_building_dummy_overlap",
        expect_active=True,
        description="Large red wall panel left-of-center; dummy overlaps visually in FOV.",
        pixel_layout=(
            f"Building filled rect ({bx}, {by})–({bx + bw}, {by + bh}). "
            f"Dummy scale=1.1 foot_y={CY + 100}: head {boxes['head']}, chest {boxes['chest']}. "
            "1px horizontal gap between building and chest plate."
        ),
        pass_fail_rationale=(
            "PASS: building contour filtered/excluded as architecture; dummy cluster "
            "keeps head+chest (2–3 parts), body_shape≥0.40. Building-only would FAIL (solid_wall)."
        ),
        optional_assertions={"min_body_shape_score": 0.40, "max_centroid_x_delta": 90},
    )
    return frame, spec


def gen_orange_health_bar_above_dummy() -> tuple[np.ndarray, SyntheticFrameSpec]:
    """(3) Orange shield bar above dummy; bar is out-of-mask, dummy is sole red target."""
    frame = _blank(0)
    boxes = _apex_dummy(frame, CX, CY + 90, scale=1.0)
    head_x, head_y, _, head_h = boxes["head"]
    bar_y0 = head_y - 18
    bar = (CX - 58, bar_y0, 116, 8)  # thin wide bar above head
    bx, by, bw, bh = bar
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), ORANGE_BAR, -1)
    spec = SyntheticFrameSpec(
        name="orange_health_bar_above_dummy",
        expect_active=True,
        description="Orange shield strip above segmented dummy (firing-range layout).",
        pixel_layout=(
            f"Orange bar ({bx}, {by})–({bx + bw}, {by + bh}) BGR{ORANGE_BAR}. "
            f"Dummy head {boxes['head']}, chest {boxes['chest']}, knees {boxes['knees']}."
        ),
        pass_fail_rationale=(
            "PASS: ORANGE_BAR H≈16 is outside HSV_RED; mask contains dummy only. "
            "If drawn with RED at same geometry, _is_health_bar would strip it and dummy still PASS."
        ),
        optional_assertions={"min_body_shape_score": 0.40, "min_part_count": 2},
    )
    return frame, spec


def gen_two_dummies_different_sizes() -> tuple[np.ndarray, SyntheticFrameSpec]:
    """(4) Near full-size dummy vs smaller far dummy; closer wins."""
    frame = _blank(0)
    near = _apex_dummy(frame, CX - 20, CY + 90, scale=1.0)
    far = _apex_dummy(frame, CX + 200, CY + 90, scale=0.65)
    spec = SyntheticFrameSpec(
        name="two_dummies_different_sizes",
        expect_active=True,
        description="Two valid humanoids; rank by score_target (body term + distance).",
        pixel_layout=(
            f"Near dummy cx={CX - 20} foot_y={CY + 90} scale=1.0 head {near['head']}. "
            f"Far dummy cx={CX + 200} scale=0.65 head {far['head']} (~65% plate sizes)."
        ),
        pass_fail_rationale=(
            "PASS: both clusters accepted; nearer (lower distance_to_center, higher body score) "
            "selected. expect distance_to_center < 120."
        ),
        optional_assertions={"max_distance_to_center": 120, "min_body_shape_score": 0.40},
    )
    return frame, spec


def gen_single_merged_blob_from_blur() -> tuple[np.ndarray, SyntheticFrameSpec]:
    """(5) Zero-gap plates + Gaussian blur → one solid contour (structure destroyed)."""
    frame = _blank(0)
    s = 1.0
    head_h, chest_h, knee_h = 26, 44, 22
    head_w, chest_w, knee_w = 28, 42, 24
    gaps = (0, 0)
    foot_y = CY + 90
    total_h = head_h + chest_h + knee_h
    hy = foot_y - total_h
    cx = CX
    cv2.rectangle(frame, (cx - head_w // 2, hy), (cx + head_w // 2, hy + head_h), RED, -1)
    cy0 = hy + head_h
    cv2.rectangle(frame, (cx - chest_w // 2, cy0), (cx + chest_w // 2, cy0 + chest_h), RED, -1)
    ky0 = cy0 + chest_h
    cv2.rectangle(frame, (cx - knee_w // 2, ky0), (cx + knee_w // 2, ky0 + knee_h), RED, -1)
    merged = cv2.GaussianBlur(frame, (21, 21), 0)
    blob_bbox = (cx - chest_w // 2, hy, chest_w, foot_y - hy)
    spec = SyntheticFrameSpec(
        name="single_merged_blob_from_blur",
        expect_active=False,
        description="Touching plates blurred into one filled silhouette.",
        pixel_layout=(
            f"Pre-blur plates stacked with 0px gaps, foot_y={foot_y}. "
            f"cv2.GaussianBlur(ksize=(21,21)). Expected merged bbox ≈ {blob_bbox}, "
            "fill≈1.0, aspect≈2.1, 1 contour part."
        ),
        pass_fail_rationale=(
            "FAIL: analyze_figure → solid_wall (fill>0.92, aspect<2.2, single part). "
            "Exercises shape-first gate; red color alone must not activate."
        ),
        optional_assertions={"reject_reason": "solid_wall"},
    )
    return merged, spec


def gen_red_stripe_dummy_in_fov() -> tuple[np.ndarray, SyntheticFrameSpec]:
    """(6) Horizontal wall stripe well above dummy (no contour merge)."""
    frame = _blank(0)
    stripe_y, stripe_h = 180, 28
    frame[stripe_y : stripe_y + stripe_h, :] = RED
    boxes = _apex_dummy(frame, CX, CY + 100, scale=1.0)
    spec = SyntheticFrameSpec(
        name="red_stripe_dummy_in_fov",
        expect_active=True,
        description="Full-width horizontal stripe + centered dummy, both inside FOV circle.",
        pixel_layout=(
            f"Stripe rows y∈[{stripe_y}, {stripe_y + stripe_h}) all x. "
            f"Dummy foot_y={CY + 100} head {boxes['head']}, chest {boxes['chest']}, "
            f"knees {boxes['knees']}. Vertical separation ≈128px (no shared contour)."
        ),
        pass_fail_rationale=(
            "PASS: stripe cluster rejected (horizontal_stripe); dummy cluster accepted. "
            "FAIL if stripe at cy±0 overlapping dummy (merged cluster → horizontal_stripe)."
        ),
        optional_assertions={"min_body_shape_score": 0.40, "min_part_count": 2},
    )
    return frame, spec


def gen_small_far_dummy() -> tuple[np.ndarray, SyntheticFrameSpec]:
    """(7) Distant dummy at scale 0.55 near upper FOV edge."""
    frame = _blank(0)
    foot_y = CY + 40
    boxes = _apex_dummy(frame, CX, foot_y, scale=0.55)
    spec = SyntheticFrameSpec(
        name="small_far_dummy",
        expect_active=True,
        description="Miniature segmented dummy (~55% plate sizes) higher in frame.",
        pixel_layout=(
            f"Dummy cx={CX} foot_y={foot_y} scale=0.55 → "
            f"head {boxes['head']} (~15×14px), chest {boxes['chest']} (~23×24px), "
            f"knees {boxes['knees']} (~13×12px). Total height ≈63px."
        ),
        pass_fail_rationale=(
            "PASS: total area > min_area (60); 3-part structure still clears "
            "_MIN_BODY_SHAPE_PARTIAL. scale<0.45 or foot_y<CY-120 would risk FAIL (too_small)."
        ),
        optional_assertions={"min_body_shape_score": 0.34, "max_bbox_h": 80},
    )
    return frame, spec


def gen_crouched_partial() -> tuple[np.ndarray, SyntheticFrameSpec]:
    """(8) Crouched dummy with head+chest only; widened gap avoids solid_wall."""
    frame = _blank(0)
    foot_y = CY + 55
    boxes = _apex_dummy(
        frame,
        CX,
        foot_y,
        scale=1.0,
        crouch=True,
        partial=True,
        head_chest_gap=14,
    )
    spec = SyntheticFrameSpec(
        name="crouched_partial",
        expect_active=True,
        description="Crouch plate sizes, no knee segment, 14px head–chest gap.",
        pixel_layout=(
            f"Crouch head 28×20, chest 42×36, gap=14px (no knees). "
            f"foot_y={foot_y} → head {boxes['head']}, chest {boxes['chest']}. "
            "Cluster bbox ≈43×71, fill≈0.82."
        ),
        pass_fail_rationale=(
            "PASS: 2-part partial body clears structure gate (head+torso scores). "
            "FAIL with default crouch gap=6 (fill≈0.92 → solid_wall). "
            "FAIL if only a single plate or health-bar aspect."
        ),
        optional_assertions={"min_body_shape_score": 0.34, "max_part_count": 2},
    )
    return frame, spec


# Registry for parametrized tests / debug scripts
SYNTHETIC_FRAME_GENERATORS: dict[str, Callable[[], tuple[np.ndarray, SyntheticFrameSpec]]] = {
    "noisy_lighting_gradient": gen_noisy_lighting_gradient,
    "red_building_dummy_overlap": gen_red_building_dummy_overlap,
    "orange_health_bar_above_dummy": gen_orange_health_bar_above_dummy,
    "two_dummies_different_sizes": gen_two_dummies_different_sizes,
    "single_merged_blob_from_blur": gen_single_merged_blob_from_blur,
    "red_stripe_dummy_in_fov": gen_red_stripe_dummy_in_fov,
    "small_far_dummy": gen_small_far_dummy,
    "crouched_partial": gen_crouched_partial,
}
