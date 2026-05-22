"""Humanoid targeting: shape/structure only (no HSV/skin color). Color mask optional legacy."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger("targeting")

# --- Tunables (1080p baseline, scales with frame size) ---
_MAX_AREA_RATIO = 0.30
_VIEWMODEL_EXCLUDE_FRAC = 0.22
_BODY_Y_LO_FRAC = 0.28
_BODY_Y_HI_FRAC = 0.52
_MAX_ABOVE_CENTER_FRAC = 0.40
_MIN_CONFIDENCE = 0.30
_STICKY_SWITCH_RATIO = 2.2
_MIN_BODY_SHAPE_ACCEPT = 0.40
_MIN_BODY_SHAPE_PARTIAL = 0.34

DETECTION_MODE_SHAPE = "shape"
DETECTION_MODE_HSV = "hsv"
DETECTION_MODE_HYBRID = "hybrid"
# Default "best-of-all-signals" mode tuned for Apex Legends. Fuses shape edges,
# saturation, motion difference, and the Apex red-enemy-outline cue.
DETECTION_MODE_APEX = "apex"
_VALID_DETECTION_MODES = frozenset(
    {DETECTION_MODE_SHAPE, DETECTION_MODE_HSV, DETECTION_MODE_HYBRID, DETECTION_MODE_APEX}
)
DETECTION_MODE_DEFAULT = DETECTION_MODE_APEX

_LAST_DEBUG_LINES: list[str] = []


class RejectReason(str, Enum):
    OK = "ok"
    TOO_SMALL = "too_small"
    TOO_LARGE = "too_large"
    HORIZONTAL_STRIPE = "horizontal_stripe"
    ARCHITECTURE_PANEL = "architecture_panel"
    DIAMOND_SIGN = "diamond_sign"
    HEALTH_BAR_ONLY = "health_bar_only"
    SIGHT_PIP = "sight_pip"
    SCOPE_RETICLE = "scope_reticle"
    NO_BODY_STRUCTURE = "no_body_structure"
    SOLID_WALL = "solid_wall"
    OUTSIDE_FOV = "outside_fov"
    FLOATING_CLUSTER = "floating_cluster"
    ROUND_NON_BODY = "round_non_body"
    NO_TORSO = "no_torso"
    NO_BODY_STACK = "no_body_stack"
    SKY_BLOB = "sky_blob"
    LOW_SCORE = "low_body_shape_score"


class PartRole(str, Enum):
    HEAD = "head"
    TORSO = "torso"
    LIMB = "limb"
    UNKNOWN = "unknown"


@dataclass
class Target:
    centroid_x: float
    centroid_y: float
    area: float
    distance_to_center: float = 0.0
    confidence: float = 0.0
    bbox_x: int = 0
    bbox_y: int = 0
    bbox_w: int = 0
    bbox_h: int = 0
    solidity: float = 1.0
    humanoid_score: float = 0.0
    part_count: int = 0
    body_shape_score: float = 0.0
    head_score: float = 0.0
    torso_score: float = 0.0
    limb_stack_score: float = 0.0
    # Fraction of bbox covered by the Apex red-enemy HSV mask. Used by
    # score_target to soften the single-part penalty when the silhouette is
    # clearly a red enemy whose body just happens to be one connected blob
    # (real Apex characters often are — armour + helmet merge into one mass
    # against dim terrain). 0.0 when no red coverage / red mask not built.
    red_coverage: float = 0.0
    # D3 (audit): fill_ratio (mask fill within bbox) is plumbed onto the
    # target so the red-coverage softener path can reject solid red blobs
    # (fill ~1.0 — synthetic blurred test blobs / red panels) while still
    # accepting real characters (fill ~0.40-0.85 with head/torso/limb gaps).
    fill_ratio: float = 0.0
    # D3 (audit) cont: max part-circularity for the cluster. Real humanoid
    # silhouettes are jagged (max_circ < 0.55 — head/torso/limb transitions
    # spike the perimeter). Solid synthetic blurred blobs are smooth and
    # have max_circ > 0.60; the softener path uses this to gate-out blobs.
    max_circularity: float = 0.0
    reject_reason: str = RejectReason.OK.value


@dataclass
class DetectionResult:
    target: Target | None
    candidates: int
    best_score: float = 0.0
    debug_lines: list[str] = field(default_factory=list)
    active: bool = False


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
    aspect_wh: float
    aspect_hw: float
    solidity: float
    extent: float
    circularity: float
    role: PartRole = PartRole.UNKNOWN


@dataclass
class _FigureAnalysis:
    accepted: bool
    reject_reason: RejectReason
    body_shape_score: float
    head_score: float
    torso_score: float
    limb_stack_score: float
    vertical_profile_score: float
    geometry_score: float
    fill_ratio: float
    aspect: float
    bx: int
    by: int
    bw: int
    bh: int
    part_count: int
    aim_x: float
    aim_y: float
    total_area: float
    solidity: float
    debug_detail: str


def get_last_debug_lines() -> list[str]:
    return list(_LAST_DEBUG_LINES)


def _scale(frame_w: int, frame_h: int) -> float:
    return min(frame_w, frame_h) / 1080.0


def _build_fov_mask(height: int, width: int, center_x: float, center_y: float, radius: int) -> np.ndarray:
    y, x = np.ogrid[:height, :width]
    cx = round(center_x)
    cy = round(center_y)
    return ((x - cx) ** 2 + (y - cy) ** 2 <= radius * radius).astype(np.uint8)


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
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, close_k, iterations=1)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    return combined



def _normalize_detection_mode(mode: str | None) -> str:
    m = (mode or DETECTION_MODE_DEFAULT).strip().lower()
    if m not in _VALID_DETECTION_MODES:
        return DETECTION_MODE_DEFAULT
    return m


def build_shape_mask(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Color-free foreground mask: local contrast + edges, morphology to join body plates.
    Works across Apex skin colors; shape scoring rejects UI/HUD blobs.
    """
    h, w = frame_bgr.shape[:2]
    scale = _scale(w, h)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    k_small = max(3, int(3 * scale) | 1)
    k_large = max(5, int(7 * scale) | 1)
    kernel_s = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_small, k_small))
    kernel_l = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_large, k_large))

    bg_k = max(15, int(21 * scale)) | 1
    bg_blur = cv2.GaussianBlur(blur, (bg_k, bg_k), 0)
    local = cv2.absdiff(blur, bg_blur)
    contrast_thr = max(8, int(12 * scale))
    _, contrast = cv2.threshold(local, contrast_thr, 255, cv2.THRESH_BINARY)
    canny_lo = max(18, int(25 * scale))
    canny_hi = max(50, int(85 * scale))
    edges = cv2.Canny(blur, canny_lo, canny_hi)
    edges = cv2.dilate(edges, kernel_s, iterations=2)

    mask = cv2.bitwise_or(contrast, edges)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_l, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_s, iterations=1)
    return mask



def build_chroma_spread_mask(frame_bgr: np.ndarray) -> np.ndarray:
    """
    HSV saturation × brightness — vivid Apex armor/skin vs dull terrain, hue-neutral.
    Uses HSV S directly (not raw BGR spread) so highly saturated grass/sand do NOT
    flood the mask; only strongly saturated foreground regions pass.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    # D4 (audit): sat_thr lowered 120 -> 90 so desaturated Apex skins
    # (Horizon white, Crypto dark, Wraith greys) still reach the cluster
    # stage. Highly saturated grass/sand stays out because val_thr keeps
    # dim terrain rejected; foreground sat>=90 + val>=70 is rare in
    # Apex world tiles but typical of armour/skin highlights.
    sat_thr = 90
    val_thr = 70
    sat_ok = cv2.threshold(s, sat_thr, 255, cv2.THRESH_BINARY)[1]
    val_ok = cv2.threshold(v, val_thr, 255, cv2.THRESH_BINARY)[1]
    mask = cv2.bitwise_and(sat_ok, val_ok)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    return mask


# Apex red enemy highlight (BGR red plus auto-color saturated edges).
# Apex draws a reddish silhouette / glow around enemies; the hue wraps around
# 0/180 in OpenCV HSV (H is 0..179). We split the lower/upper red band and
# loosen saturation/value floors so dim-lit red armour and red base skins
# (Crypto red, Bloodhound rust, Revenant maroon, Lifeline red shield etc.)
# still pass — the morphology pass below cleans pepper noise and stitches
# small gaps without stripping the filled interior.
# D2 (audit): saturation floor lowered 80 -> 55 and value floor lowered
# 70 -> 50 so blended red glow over skin/terrain (S ~50-100, V ~50-90 — the
# Horizon/Crypto/Wraith case) still passes the gate. Hue bands unchanged.
_APEX_RED_HSV_LO_A = np.array([0, 55, 50], dtype=np.uint8)
_APEX_RED_HSV_HI_A = np.array([12, 255, 255], dtype=np.uint8)
_APEX_RED_HSV_LO_B = np.array([168, 55, 50], dtype=np.uint8)
_APEX_RED_HSV_HI_B = np.array([180, 255, 255], dtype=np.uint8)


def build_red_enemy_mask(frame_bgr: np.ndarray) -> np.ndarray:
    """Apex red-enemy mask — filled red regions, not just an outline ribbon.

    Live-game testing showed the previous outline-only implementation
    (MORPH_GRADIENT after CLOSE) collapsed solid red characters — red armour,
    red helmet, red gloves — down to a thin perimeter band whose pixel count
    was too small to clear ``_extract_parts`` and whose fill ratio looked
    like a hollow ring to ``analyze_figure``. The detector then fell back to
    shape + chroma + motion alone and missed the dead-centre red enemy.

    The new implementation keeps the interior filled. Two HSV bands cover
    the hue wrap at 0/180; we OR the bands, run a small MORPH_OPEN to drop
    pepper noise, and a slightly larger MORPH_CLOSE to bridge tiny gaps
    (red gap between helmet/chest plates etc.). Downstream contour
    extraction then sees a coherent humanoid silhouette.
    """
    h, w = frame_bgr.shape[:2]
    scale = _scale(w, h)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    a = cv2.inRange(hsv, _APEX_RED_HSV_LO_A, _APEX_RED_HSV_HI_A)
    b = cv2.inRange(hsv, _APEX_RED_HSV_LO_B, _APEX_RED_HSV_HI_B)
    raw = cv2.bitwise_or(a, b)
    # D1 (audit): MORPH_OPEN(3) erased a Horizon-style 2-px-wide red glow
    # ribbon entirely. When the raw red mask is sparse (thin halo only) we
    # SKIP open to preserve the ribbon. When the raw mask is high-density
    # (saturated red character or noisy BG with low-S pickups) we still
    # need the 3x3 ellipse open to remove pepper noise — otherwise textured
    # BGs flood the cluster pass with scattered red micro-blobs. The
    # sparse-glow case is what the audit's D1 was targeting.
    raw_count = int((raw > 0).sum())
    frame_px = max(1, h * w)
    raw_ratio = raw_count / frame_px
    if raw_ratio > 0.003:
        # High-density: keep the original 3x3 ellipse open to drop pepper.
        k_open = max(3, int(3 * scale) | 1)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
        cleaned = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel_open, iterations=1)
    else:
        # Sparse: keep the ribbon intact. Use a 2x2 cross to drop single
        # isolated pixels only when we still have a non-trivial mass.
        if raw_ratio > 0.0005:
            kernel_open = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
            cleaned = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel_open, iterations=1)
        else:
            cleaned = raw
    k_close = max(5, int(5 * scale) | 1)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
    filled = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    return filled


# Back-compat shim: existing call sites still import build_red_outline_mask.
# The function no longer produces an outline-only ribbon; it returns the
# filled red-enemy mask. Kept as an alias so external scripts continue to
# work; new code should use build_red_enemy_mask().
def build_red_outline_mask(frame_bgr: np.ndarray) -> np.ndarray:
    """Deprecated alias for :func:`build_red_enemy_mask` (filled mask)."""
    return build_red_enemy_mask(frame_bgr)


def build_motion_diff_mask(
    gray_now: np.ndarray,
    prev_gray: np.ndarray,
    *,
    threshold: int = 14,
) -> np.ndarray:
    """
    Inter-frame absolute difference — picks up *any* moving silhouette regardless
    of color/contrast. This is the main reason Apex enemies remain detectable when
    their armor blends into terrain: they move, the background does not.

    Output is a thin edge-of-motion ribbon (no dilation) so it reinforces the
    current silhouette boundary rather than creating a "ghost" of the prior frame.
    """
    if gray_now.shape != prev_gray.shape:
        return np.zeros_like(gray_now)
    diff = cv2.absdiff(gray_now, prev_gray)
    _, mask = cv2.threshold(diff, max(6, int(threshold)), 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    return mask


@dataclass
class DetectionContext:
    """
    Per-runtime state for the detection pipeline.

    Holds the previous gray frame so build_detection_mask can fuse a motion-difference
    channel. The most recent motion-diff mask is cached so the scoring stage can
    boost candidates whose bbox overlaps real movement (suppresses false positives
    on static walls/UI panels). Reset() clears state when assist is paused/idle.

    A short-lived "motion-validated" memory keeps a body that *was* moving from
    losing its motion bonus the instant it stops strafing — without this the
    confidence dips below threshold for one frame, the target is dropped, and
    the dot flickers as the runtime re-locks. The credit decays linearly to zero
    over ``motion_memory_frames`` detection frames after the last live motion
    coverage above ``motion_validate_threshold``.
    """

    prev_gray: np.ndarray | None = None
    prev_size: tuple[int, int] = (0, 0)
    motion_assist: bool = True
    motion_threshold: int = 10
    motion_decay: float = 0.55
    last_motion_mask: np.ndarray | None = None
    # Raised from 12 → 24 so a still-locked target keeps its motion-validated
    # bonus across a longer dry spell (≈400 ms at 60 FPS) — prevents the
    # single-part penalty from snapping the dot off a stationary enemy.
    motion_memory_frames: int = 24
    motion_validate_threshold: float = 0.12
    _validated_bbox: tuple[int, int, int, int] | None = None
    _validated_credit: int = 0

    def reset(self) -> None:
        self.prev_gray = None
        self.prev_size = (0, 0)
        self.last_motion_mask = None
        self._validated_bbox = None
        self._validated_credit = 0

    def update_prev(self, gray: np.ndarray) -> None:
        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = gray.copy()
        else:
            decay = max(0.0, min(0.95, self.motion_decay))
            cv2.addWeighted(self.prev_gray, decay, gray, 1.0 - decay, 0, dst=self.prev_gray)
        self.prev_size = (gray.shape[1], gray.shape[0])
        if self._validated_credit > 0:
            self._validated_credit -= 1

    def motion_coverage_ratio(self, bx: int, by: int, bw: int, bh: int) -> float:
        """
        Fraction of the bbox covered by recent motion-diff pixels.

        Returns 0.0 when no motion mask is cached (e.g. first frame after reset).
        Used by the scoring stage to differentiate a real moving body from a
        static wall whose silhouette happens to look humanoid.
        """
        mm = self.last_motion_mask
        if mm is None or bw <= 0 or bh <= 0:
            return 0.0
        x0 = max(0, int(bx))
        y0 = max(0, int(by))
        x1 = min(mm.shape[1], int(bx + bw))
        y1 = min(mm.shape[0], int(by + bh))
        if x1 <= x0 or y1 <= y0:
            return 0.0
        roi = mm[y0:y1, x0:x1]
        if roi.size == 0:
            return 0.0
        return float((roi > 0).sum()) / float(roi.size)

    def note_motion_validated(self, bx: int, by: int, bw: int, bh: int) -> None:
        """Record the bbox of a target whose LIVE motion coverage cleared the
        ``motion_validate_threshold`` gate. Refreshes the decay credit so the
        next few frames inherit the validation even if the body stops moving."""
        if bw <= 0 or bh <= 0:
            return
        self._validated_bbox = (int(bx), int(by), int(bw), int(bh))
        self._validated_credit = max(self._validated_credit, int(self.motion_memory_frames))

    def motion_coverage_with_memory(self, bx: int, by: int, bw: int, bh: int) -> float:
        """Live coverage, falling back to a decaying remembered coverage if the
        bbox overlaps the last motion-validated target. This prevents a freshly
        stationary enemy from losing the motion bonus, dropping below the
        confidence floor, and triggering a relock flicker.
        """
        live = self.motion_coverage_ratio(bx, by, bw, bh)
        if live >= 0.04:
            return live
        vb = self._validated_bbox
        if vb is None or self._validated_credit <= 0 or bw <= 0 or bh <= 0:
            return live
        vbx, vby, vbw, vbh = vb
        cx_q = bx + bw * 0.5
        cy_q = by + bh * 0.5
        cx_v = vbx + vbw * 0.5
        cy_v = vby + vbh * 0.5
        tol = max(float(vbw), float(vbh), float(bw), float(bh)) * 0.6
        if math.hypot(cx_q - cx_v, cy_q - cy_v) > tol:
            return live
        ratio = self._validated_credit / max(1, int(self.motion_memory_frames))
        virtual = 0.20 * max(0.0, min(1.0, ratio))
        return max(live, virtual)


def build_detection_mask(
    frame_bgr: np.ndarray,
    hsv_ranges: list[dict[str, Any]] | None,
    *,
    detection_mode: str | None = None,
    context: DetectionContext | None = None,
) -> np.ndarray:
    """Build binary mask for contour extraction (default: shape-only).

    If a DetectionContext is provided, an inter-frame motion-difference channel is
    fused into the result — boosting recall on Apex characters whose color blends
    into the background but who move dynamically.
    """
    mode = _normalize_detection_mode(detection_mode)
    shape_m = build_shape_mask(frame_bgr)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    bg_mean = float(np.mean(gray))
    shape_px = int((shape_m > 0).sum())

    # Chroma spread fuses for shape/hybrid/apex modes — Apex armor/skin is more
    # saturated than dull terrain, so this catches the low-contrast cases the
    # plain edge+contrast mask misses (yellow on grass, olive on grass).
    # Gate stays tight to avoid merging plates of high-contrast targets.
    if mode != DETECTION_MODE_HSV:
        if bg_mean > 35.0 and shape_px < 15000:
            shape_m = cv2.bitwise_or(shape_m, build_chroma_spread_mask(frame_bgr))

    # Temporal motion channel — strongest signal for moving enemies regardless of color.
    if mode != DETECTION_MODE_HSV and context is not None and context.motion_assist:
        if context.prev_gray is not None and context.prev_gray.shape == gray.shape:
            motion = build_motion_diff_mask(gray, context.prev_gray, threshold=context.motion_threshold)
            shape_m = cv2.bitwise_or(shape_m, motion)
            context.last_motion_mask = motion
        else:
            context.last_motion_mask = None
        context.update_prev(gray)

    if mode == DETECTION_MODE_APEX:
        # D1 (audit): only fuse the OUTLINE (MORPH_GRADIENT) of the filled
        # red mask into clustering. Fusing the filled INTERIOR collapses
        # the inner v_score signal for uniformly-red small characters
        # (firing-range crouched dummy) which then look identical to
        # synthetic blurred blobs. The glow-ring case the audit targeted
        # is handled by the MORPH_OPEN fix inside build_red_enemy_mask:
        # the 2-px ring now survives, and MORPH_GRADIENT of a thin ring
        # is still a thin ring — sufficient to seed a cluster. The FILLED
        # mask is still used downstream by red_coverage scoring.
        filled = build_red_enemy_mask(frame_bgr)
        fh, fw = filled.shape[:2]
        k = max(3, int(3 * _scale(fw, fh)) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        outline = cv2.morphologyEx(filled, cv2.MORPH_GRADIENT, kernel, iterations=1)
        return cv2.bitwise_or(shape_m, outline)
    if mode == DETECTION_MODE_SHAPE:
        return shape_m
    if mode == DETECTION_MODE_HSV:
        return build_hsv_mask(frame_bgr, hsv_ranges or [])
    # hybrid: union shape + optional HSV (legacy tuning)
    if hsv_ranges:
        return cv2.bitwise_or(shape_m, build_hsv_mask(frame_bgr, hsv_ranges))
    return shape_m


def clamp_point_to_fov(
    x: float,
    y: float,
    center_x: float,
    center_y: float,
    radius: float,
    *,
    margin_frac: float = 0.92,
) -> tuple[float, float]:
    """Keep aim point inside detection FOV — prevents overlay dot at screen edge."""
    if radius <= 0.0 or not (math.isfinite(x) and math.isfinite(y)):
        return x, y
    dx = x - center_x
    dy = y - center_y
    dist = math.hypot(dx, dy)
    cap = max(1.0, radius * max(0.5, min(1.0, margin_frac)))
    if dist <= cap or dist <= 0.0:
        return x, y
    s = cap / dist
    return center_x + dx * s, center_y + dy * s


def _is_health_bar(part: _RedPart, scale: float) -> bool:
    return (
        part.aspect_wh >= 3.6
        and part.h <= 16 * scale
        and part.w >= 32 * scale
        and part.area <= 1100 * scale * scale
    )


def _is_sight_pip(part: _RedPart, scale: float) -> bool:
    """Tiny UI pips only — not armor plates at any resolution."""
    max_dim = max(part.w, part.h)
    min_dim = min(part.w, part.h)
    return (
        max_dim <= max(12.0, 14.0 * scale)
        and min_dim <= max(8.0, 10.0 * scale)
        and part.solidity >= 0.9
    )


def _is_scope_reticle(
    part: _RedPart,
    scale: float,
    fov_cx: float | None,
    fov_cy: float | None,
) -> bool:
    """Reject scope/sight reticle artifacts that sit on the crosshair during ADS.

    Scope reticles (1x/2x/3x/4x) overlay the viewmodel near the FOV centre, where
    the standard ``_VIEWMODEL_EXCLUDE_FRAC`` bottom-mask cannot help. A real body
    silhouette is much wider/taller than a reticle line and is rarely centred
    exactly on the crosshair (the player would already be aiming AT the body).
    This filter is FOV-centre-aware: large body-sized blobs near the crosshair
    are still allowed; only extreme-aspect lines, pip-sized dots, and small
    triangular chevrons inside the inner crosshair zone are rejected.
    """
    if fov_cx is None or fov_cy is None:
        return False
    dx = part.cx - fov_cx
    dy = part.cy - fov_cy
    d = math.hypot(dx, dy)

    # Thin horizontal reticle line (e.g. HCOG horizontal bar) inside crosshair zone.
    if d <= 32.0 * scale and part.h <= max(5.0, 6.0 * scale) and part.aspect_wh >= 6.0:
        return True
    # Thin vertical reticle line / chevron stem.
    if d <= 32.0 * scale and part.w <= max(5.0, 6.0 * scale) and part.aspect_hw >= 6.0:
        return True
    # Tiny solid pip (red dot, holo dot) just larger than the _is_sight_pip cap.
    if (
        d <= 20.0 * scale
        and part.area <= max(40.0, 50.0 * scale * scale)
        and part.solidity >= 0.85
        and max(part.w, part.h) <= max(16.0, 18.0 * scale)
    ):
        return True
    # Small triangular / chevron / range-marker shape: low extent, small area,
    # dead-centre. Real bodies have extent >= 0.55 or are far larger.
    if (
        d <= 26.0 * scale
        and part.area <= 220.0 * scale * scale
        and part.extent < 0.55
        and max(part.w, part.h) <= max(22.0, 26.0 * scale)
    ):
        return True
    return False


def _is_diamond_sign(part: _RedPart, scale: float) -> bool:
    """Practice-board diamond (~45 deg). Axis-aligned rects are armor plates, not signs."""
    # D8 (audit): upper-area cap raised 7000 -> 25000 * scale^2 so large
    # firing-range diamond boards (180-px-side hazard signs) are still
    # rejected at this single-part filter. Real human characters never
    # have part-circularity / aspect / rect-angle profile of a diamond,
    # so the gate stays specific.
    if part.area < 500 * scale * scale or part.area > 25000 * scale * scale:
        return False
    if part.aspect_hw >= 1.18:
        return False
    if not (0.82 <= part.aspect_wh <= 1.18):
        return False
    rect = cv2.minAreaRect(part.contour)
    (_, _), (rw, rh), angle = rect
    if min(rw, rh) <= 0:
        return False
    if max(rw, rh) / min(rw, rh) > 1.45:
        return False
    ang = abs(angle)
    if ang > 45.0:
        ang = 90.0 - ang
    return 28.0 <= ang <= 62.0




def _extract_parts(
    mask: np.ndarray,
    frame_w: int,
    frame_h: int,
    *,
    fov_cx: float | None = None,
    fov_cy: float | None = None,
) -> list[_RedPart]:
    scale = _scale(frame_w, frame_h)
    area_lo = 60 * scale * scale
    area_hi = 95000 * scale * scale
    parts: list[_RedPart] = []

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < area_lo or area > area_hi:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue
        if w > frame_w * 0.26:
            continue
        if h > frame_h * 0.42 and w < frame_w * 0.22:
            continue

        hull = cv2.convexHull(contour)
        hull_a = float(cv2.contourArea(hull))
        solidity = area / hull_a if hull_a > 0 else 0.0
        extent = area / float(w * h)
        peri = cv2.arcLength(contour, True)
        circ = (4.0 * math.pi * area / (peri * peri)) if peri > 0 else 0.0

        part = _RedPart(
            contour=contour,
            x=x,
            y=y,
            w=w,
            h=h,
            area=area,
            cx=x + w * 0.5,
            cy=y + h * 0.5,
            aspect_wh=w / float(h),
            aspect_hw=h / float(w),
            solidity=solidity,
            extent=extent,
            circularity=circ,
        )
        if _is_health_bar(part, scale):
            continue
        if _is_sight_pip(part, scale):
            continue
        if _is_scope_reticle(part, scale, fov_cx, fov_cy):
            continue
        if _is_diamond_sign(part, scale):
            continue
        if (
            area >= 9000 * scale * scale
            and w > frame_w * 0.14
            and extent >= 0.74
            and w >= h * 1.12
        ):
            continue
        if (
            circ >= 0.72
            and 0.75 <= (w / float(h)) <= 1.35
            and area >= 600 * scale * scale
            and (y + h) < frame_h * 0.45
        ):
            continue
        parts.append(part)

    return parts


def _classify_parts(parts: list[_RedPart], cluster_h: int, scale: float) -> None:
    if not parts:
        return
    sorted_p = sorted(parts, key=lambda p: p.y)
    areas = [p.area for p in sorted_p]
    max_area = max(areas)
    top = sorted_p[0]

    for i, p in enumerate(sorted_p):
        rel_y = (p.cy - sorted_p[0].y) / max(cluster_h, 1)
        ar = p.aspect_hw
        if i == 0 and rel_y < 0.32:
            if (
                p.area <= max_area * 0.85
                and 0.35 <= ar <= 3.5
                and p.h <= 85 * scale
                and not _is_health_bar(p, scale)
            ):
                p.role = PartRole.HEAD
                continue
        if 0.12 <= rel_y <= 0.72 and p.area >= max_area * 0.45:
            p.role = PartRole.TORSO
            continue
        if rel_y > 0.45 and 0.35 <= ar <= 4.0:
            p.role = PartRole.LIMB
            continue
        p.role = PartRole.UNKNOWN


def _cluster_parts(parts: list[_RedPart], frame_w: int, frame_h: int | None = None) -> list[list[_RedPart]]:
    if not parts:
        return []
    fh = frame_h if frame_h is not None else int(max(p.y + p.h for p in parts))
    x_tol = max(22.0, 0.16 * float(np.median([p.w for p in parts])))
    clusters: list[list[_RedPart]] = []
    for part in sorted(parts, key=lambda p: p.cx):
        placed = False
        for cl in clusters:
            cl_cx = float(np.mean([p.cx for p in cl]))
            x0 = min(p.x for p in cl)
            x1 = max(p.x + p.w for p in cl)
            cl_y1 = max(p.y + p.h for p in cl)
            cl_y0 = min(p.y for p in cl)
            overlap = max(0, min(part.x + part.w, x1) - max(part.x, x0))
            py0, py1 = part.y, part.y + part.h
            if py1 < cl_y0:
                v_sep = float(cl_y0 - py1)
            elif py0 > cl_y1:
                v_sep = float(py0 - cl_y1)
            else:
                v_sep = -1.0
            if v_sep > max(85.0, 95.0 * _scale(frame_w, fh)):
                continue
            if abs(part.cx - cl_cx) <= x_tol or overlap / max(1, min(part.w, x1 - x0)) >= 0.25:
                cl.append(part)
                placed = True
                break
        if not placed:
            clusters.append([part])
    return _merge_nearby_clusters(clusters, frame_w, fh)




def _clusters_should_merge(
    parts_a: list[_RedPart],
    parts_b: list[_RedPart],
    scale: float,
) -> bool:
    """Merge left/right arm plates into one humanoid column."""
    bx1, by1, bw1, bh1 = _cluster_bbox(parts_a)
    bx2, by2, bw2, bh2 = _cluster_bbox(parts_b)
    cx1 = bx1 + bw1 * 0.5
    cx2 = bx2 + bw2 * 0.5
    x_tol = max(48.0 * scale, 0.55 * max(bw1, bw2))
    if abs(cx1 - cx2) > x_tol:
        return False
    y_overlap = min(by1 + bh1, by2 + bh2) - max(by1, by2)
    if y_overlap >= -8 * scale:
        return True
    if by2 + bh2 < by1:
        gap = float(by1 - (by2 + bh2))
    elif by1 + bh1 < by2:
        gap = float(by2 - (by1 + bh1))
    else:
        gap = 0.0
    return gap <= 95 * scale


def _merge_nearby_clusters(clusters: list[list[_RedPart]], frame_w: int, frame_h: int) -> list[list[_RedPart]]:
    if len(clusters) <= 1:
        return clusters
    scale = _scale(frame_w, frame_h)
    merged: list[list[_RedPart]] = []
    used = [False] * len(clusters)
    for i, cl in enumerate(clusters):
        if used[i]:
            continue
        acc = list(cl)
        used[i] = True
        changed = True
        while changed:
            changed = False
            for j, other in enumerate(clusters):
                if used[j]:
                    continue
                if _clusters_should_merge(acc, other, scale):
                    acc.extend(other)
                    used[j] = True
                    changed = True
        merged.append(acc)
    return merged




def _max_part_circularity(parts: list[_RedPart]) -> float:
    if not parts:
        return 0.0
    return max(p.circularity for p in parts)


def _part_alignment_score(parts: list[_RedPart], scale: float) -> float:
    """How well parts share a vertical body column (dummy joint layout)."""
    if len(parts) < 2:
        return 0.0
    cx_std = float(np.std([p.cx for p in parts]))
    align = max(0.0, 1.0 - cx_std / (24.0 * scale))
    sorted_p = sorted(parts, key=lambda p: p.y)
    gaps = [
        sorted_p[i + 1].y - (sorted_p[i].y + sorted_p[i].h)
        for i in range(len(sorted_p) - 1)
    ]
    good = sum(1 for g in gaps if -6 * scale <= g <= 100 * scale)
    return min(1.0, align * 0.55 + 0.45 * (good / max(len(gaps), 1)))


def _body_structure_reject(
    parts: list[_RedPart],
    bx: int,
    by: int,
    bw: int,
    bh: int,
    frame_h: int,
    frame_w: int,
    scale: float,
    head_s: float,
    torso_s: float,
    limb_s: float,
    v_score: float,
    fill: float,
) -> RejectReason | None:
    """Body-first gate: color mask is input only; shape must pass."""
    aspect = bh / max(bw, 1)
    foot_y = by + bh
    center_y = frame_h * 0.5
    max_circ = _max_part_circularity(parts)
    align_s = _part_alignment_score(parts, scale)

    # --- sky / balloon family ---
    if foot_y < frame_h * 0.48 and aspect < 1.75 and len(parts) <= 2:
        if max_circ >= 0.48 or (aspect < 1.35 and bh < frame_h * 0.14):
            return RejectReason.SKY_BLOB

    if len(parts) == 1:
        p = parts[0]
        if (
            p.circularity >= 0.68
            and aspect < 1.45
            and not (v_score >= 0.45 and bh >= 60 * scale and aspect >= 0.90)
        ):
            return RejectReason.ROUND_NON_BODY
        if p.circularity >= 0.52 and aspect < 1.2 and bh < 70 * scale:
            return RejectReason.ROUND_NON_BODY
        if p.circularity >= 0.45 and aspect < 1.15 and fill > 0.82:
            return RejectReason.ROUND_NON_BODY
        # D3 (audit) cont: tall-oval blobs (Gaussian-blurred mass) form a
        # single high-circularity part. A real Apex crouched dummy reaches
        # circ ~0.66 with 3 distinct plates yielding clear transitions; a
        # blurred-blob hits ~0.72 with no segmentation cue. Threshold sits
        # just above the highest observed real-body circ (0.66) and just
        # below the lowest observed blob circ (0.72).
        if p.circularity >= 0.70:
            return RejectReason.ROUND_NON_BODY

    if aspect < 1.22 and len(parts) <= 2 and max_circ >= 0.42 and bh < frame_h * 0.16:
        return RejectReason.ROUND_NON_BODY

    if mid_y := by + bh * 0.5:
        if mid_y < center_y - frame_h * 0.12 and aspect < 1.5 and limb_s < 0.35:
            return RejectReason.SKY_BLOB

    # --- torso / stack required (firing-range dummy layout) ---
    has_head = any(p.role == PartRole.HEAD for p in parts) or head_s >= 0.38
    has_torso = any(p.role == PartRole.TORSO for p in parts) or torso_s >= 0.30
    has_limbs = limb_s >= 0.40 or len(parts) >= 3
    # Single tall narrow blob (Apex character whose silhouette fills uniformly).
    silhouette_ok = (
        len(parts) == 1
        and aspect >= 1.55
        and bw < frame_w * 0.12
        and bh >= 70 * scale
    )

    if not has_torso and torso_s < 0.24:
        if not (len(parts) >= 3 and limb_s >= 0.50 and v_score >= 0.35) and not silhouette_ok:
            return RejectReason.NO_TORSO

    stack_ok = (
        (has_head and has_torso)
        or (has_torso and has_limbs)
        or (len(parts) >= 3 and limb_s >= 0.42 and v_score >= 0.28 and aspect >= 1.35)
        or (aspect >= 1.55 and bh >= 65 * scale and v_score >= 0.30 and align_s >= 0.45)
        or silhouette_ok
    )
    if not stack_ok:
        return RejectReason.NO_BODY_STACK

    if aspect < 1.30 and len(parts) <= 2 and torso_s < 0.28 and limb_s < 0.40:
        return RejectReason.NO_BODY_STACK

    # D8 (audit): cluster-level rectangular-sign reject. Large multi-part
    # boards (dark frame + colored centre + decal) form a square axis-aligned
    # cluster with very low vertical-profile score because the interior rows
    # look identical (no head/torso/limb transitions). Real bodies always
    # have v_score >= 0.30 due to head-narrow/torso-wide/limb-narrow stack.
    near_square = 0.92 <= aspect <= 1.12
    if (
        near_square
        and v_score < 0.25
        and bh > 60 * scale
        and bh < frame_h * 0.50
    ):
        # Axis-aligned parts only — rotated body parts shouldn't trip this.
        axis_aligned = True
        for p in parts:
            rect = cv2.minAreaRect(p.contour)
            (_, _), (rw, rh), angle = rect
            ang = abs(angle)
            if ang > 45.0:
                ang = 90.0 - ang
            if ang > 18.0:
                axis_aligned = False
                break
        if axis_aligned:
            return RejectReason.ARCHITECTURE_PANEL

    return None


def _is_floating_cluster(
    bx: int,
    by: int,
    bw: int,
    bh: int,
    frame_h: int,
    center_y: float,
    parts: list[_RedPart],
    scale: float,
) -> bool:
    """Reject sky pips / UI blobs above the play space (not grounded humanoids)."""
    foot_y = by + bh
    mid_y = by + bh * 0.5
    if foot_y < frame_h * 0.40 and bh < frame_h * 0.28:
        return True
    if mid_y < center_y - frame_h * 0.18 and len(parts) <= 3 and bh < 95 * scale:
        return True
    if by < frame_h * 0.12 and bh < frame_h * 0.22:
        return True
    if len(parts) == 1:
        p = parts[0]
        if p.circularity >= 0.55 and foot_y < frame_h * 0.48 and p.h < 55 * scale:
            return True
    return False


def _cluster_bbox(parts: list[_RedPart]) -> tuple[int, int, int, int]:
    x0 = min(p.x for p in parts)
    y0 = min(p.y for p in parts)
    x1 = max(p.x + p.w for p in parts)
    y1 = max(p.y + p.h for p in parts)
    return x0, y0, x1 - x0, y1 - y0


def _vertical_profile_score(mask: np.ndarray, bx: int, by: int, bw: int, bh: int) -> tuple[float, float]:
    if bw < 5 or bh < 10:
        return 0.0, 0.0
    roi = mask[by : by + bh, bx : bx + bw]
    if roi.size == 0:
        return 0.0, 0.0
    cols = [max(0, min(bw - 1, int(bw * f))) for f in (0.25, 0.5, 0.75)]
    best_v, fill = 0.0, 0.0
    for cx in cols:
        column = roi[:, cx]
        red = (column > 0).astype(np.float32)
        f = float(red.mean())
        trans = int(np.sum(np.abs(np.diff(red.astype(np.int8)))))
        fill = max(fill, f)
        v = 0.0
        if f > 0.9 and trans <= 2:
            v = 0.0
        elif trans >= 3 and 0.18 <= f <= 0.82:
            v = 1.0
        elif trans >= 2 and f <= 0.75:
            v = 0.72
        elif trans >= 1 and f <= 0.88:
            v = 0.45
        elif f < 0.9:
            v = 0.15
        best_v = max(best_v, v)
    return best_v, fill


def _zone_density(mask: np.ndarray, bx: int, by: int, bw: int, bh: int, y0f: float, y1f: float) -> float:
    y0 = by + int(bh * y0f)
    y1 = by + int(bh * y1f)
    y0 = max(by, min(y0, by + bh - 1))
    y1 = max(y0 + 1, min(y1, by + bh))
    roi = mask[y0:y1, bx : bx + bw]
    if roi.size == 0:
        return 0.0
    return float((roi > 0).mean())

def _mask_chest_anchor(
    mask: np.ndarray,
    bx: int,
    by: int,
    bw: int,
    bh: int,
    *,
    y0f: float = 0.30,
    y1f: float = 0.50,
) -> tuple[float, float] | None:
    """Centroid of red pixels in upper-chest band — stable body mass, not sky above head."""
    if bw < 4 or bh < 8:
        return None
    y0 = by + int(bh * y0f)
    y1 = by + int(bh * y1f)
    y0 = max(by, min(y0, by + bh - 2))
    y1 = max(y0 + 2, min(y1, by + bh))
    roi = mask[y0:y1, bx : bx + bw]
    if roi.size == 0:
        return None
    ys, xs = np.where(roi > 0)
    if len(xs) < 8:
        return None
    # D5 (audit): use a column-density peak instead of mean/median for the
    # X anchor. The mean is mass-pulled toward an extended gun arm on a
    # sideways-viewed character; the median is only weakly robust against
    # a heavy single-side outlier. The densest column in the chest band
    # is the torso column — that's where the live game expects the dot
    # to land. We smooth column densities with a 5-px window so a noisy
    # peak doesn't latch onto a single column.
    col_density = (roi > 0).sum(axis=0).astype(np.float32)
    if col_density.size >= 5:
        kernel = np.ones(5, dtype=np.float32) / 5.0
        col_density = np.convolve(col_density, kernel, mode="same")
    densest_col = int(np.argmax(col_density))
    ax = float(bx + densest_col)
    ay = float(y0 + np.mean(ys))
    return ax, ay


def _clamp_aim_to_body_bbox(
    ax: float,
    ay: float,
    bx: int,
    by: int,
    bw: int,
    bh: int,
    parts: list[_RedPart],
    torso_fraction: float,
    aim_y_lo_frac: float = _BODY_Y_LO_FRAC,
    aim_y_hi_frac: float = _BODY_Y_HI_FRAC,
) -> tuple[float, float]:
    """Keep aim inside upper-chest band; never above head plate or into sky above bbox."""
    frac = max(0.32, min(0.48, torso_fraction))
    y_lo = by + bh * aim_y_lo_frac
    y_hi = by + bh * min(aim_y_hi_frac, frac + 0.10)
    heads = [p for p in parts if p.role == PartRole.HEAD]
    if heads:
        head_bottom = heads[0].y + heads[0].h
        y_lo = max(y_lo, head_bottom - bh * 0.02)
    ay = max(y_lo, min(y_hi, ay))
    x_lo = bx + bw * 0.30
    x_hi = bx + bw * 0.70
    ax = max(x_lo, min(x_hi, ax))
    return ax, ay


def _aim_inside_body_bbox(
    ax: float,
    ay: float,
    bx: int,
    by: int,
    bw: int,
    bh: int,
    *,
    margin_x: float = 0.12,
    margin_y_top: float = 0.08,
    margin_y_bot: float = 0.12,
) -> bool:
    if bw <= 0 or bh <= 0:
        return False
    mx = bw * margin_x
    myt = bh * margin_y_top
    myb = bh * margin_y_bot
    return (
        bx + mx <= ax <= bx + bw - mx
        and by + myt <= ay <= by + bh - myb
    )



def _score_head(parts: list[_RedPart], mask: np.ndarray, bx: int, by: int, bw: int, bh: int, scale: float) -> float:
    heads = [p for p in parts if p.role == PartRole.HEAD]
    if heads:
        h = heads[0]
        return min(1.0, 0.55 + h.circularity * 0.8 + (0.15 if h.aspect_hw < 2.2 else 0.0))

    has_body_part = any(p.role in (PartRole.HEAD, PartRole.TORSO, PartRole.LIMB) for p in parts)
    top_zone = _zone_density(mask, bx, by, bw, bh, 0.0, 0.28)
    if has_body_part and len(parts) >= 2 and top_zone >= 0.18:
        return min(0.45, top_zone * 0.9)
    if len(parts) == 1 and parts[0].aspect_hw >= 1.35:
        if _zone_density(mask, bx, by, bw, bh, 0.0, 0.22) >= 0.2 and parts[0].h >= 18 * scale:
            return 0.42
    return 0.0


def _score_torso(parts: list[_RedPart], mask: np.ndarray, bx: int, by: int, bw: int, bh: int) -> float:
    torsos = [p for p in parts if p.role == PartRole.TORSO]
    if torsos:
        t = max(torsos, key=lambda p: p.area)
        return min(1.0, 0.5 + min(t.area / max(sum(p.area for p in parts), 1), 1.0) * 0.45)

    if len(parts) >= 2:
        mid = _zone_density(mask, bx, by, bw, bh, 0.22, 0.58)
        if mid >= 0.18:
            return min(0.4, mid * 0.85)
    return 0.0


def _score_limb_stack(parts: list[_RedPart], scale: float) -> float:
    if len(parts) >= 3:
        sorted_p = sorted(parts, key=lambda p: p.y)
        gaps = []
        for i in range(len(sorted_p) - 1):
            gaps.append(sorted_p[i + 1].y - (sorted_p[i].y + sorted_p[i].h))
        good_gaps = sum(1 for g in gaps if -4 * scale <= g <= 130 * scale)
        cx_std = float(np.std([p.cx for p in parts]))
        align = max(0.0, 1.0 - cx_std / (28 * scale))
        return min(1.0, 0.35 + 0.25 * len(parts) + 0.2 * good_gaps + 0.2 * align)

    if len(parts) == 2:
        sorted_p = sorted(parts, key=lambda p: p.y)
        gap = sorted_p[1].y - (sorted_p[0].y + sorted_p[0].h)
        if -6 * scale <= gap <= 120 * scale:
            return 0.62
        return 0.35

    limbs = [p for p in parts if p.role == PartRole.LIMB]
    if limbs:
        return 0.48
    return 0.2 if len(parts) == 1 else 0.0


def _score_geometry(bw: int, bh: int, frame_w: int, frame_h: int, fill: float) -> float:
    if bw <= 0 or bh <= 0:
        return 0.0
    aspect = bh / float(bw)
    wf = bw / float(frame_w)
    hf = bh / float(frame_h)
    score = 0.0
    if 1.15 <= aspect <= 6.5:
        score += 0.45 * min(1.0, (aspect - 1.0) / 2.5)
    if wf <= 0.26:
        score += 0.25 * max(0.0, 1.0 - wf / 0.26)
    if 0.045 <= hf <= 0.58:
        score += 0.2
    if 0.15 <= fill <= 0.85:
        score += 0.1
    if bw >= bh:
        score *= 0.25
    return min(1.0, score)


def _hard_reject(
    parts: list[_RedPart],
    bx: int,
    by: int,
    bw: int,
    bh: int,
    frame_w: int,
    frame_h: int,
    fill: float,
) -> RejectReason | None:
    if bw <= 0 or bh <= 0:
        return RejectReason.TOO_SMALL
    aspect = bh / float(bw)
    if bw > frame_w * 0.32:
        return RejectReason.ARCHITECTURE_PANEL
    if aspect < 0.95:
        return RejectReason.HORIZONTAL_STRIPE
    if bw >= bh * 1.08 and fill < 0.35 and len(parts) >= 2:
        return RejectReason.HORIZONTAL_STRIPE
    if bw >= bh * 1.05 and bh < frame_h * 0.08:
        return RejectReason.HORIZONTAL_STRIPE
    if bw >= bh * 1.02 and bh < frame_h * 0.12:
        return RejectReason.HORIZONTAL_STRIPE
    # Tall narrow filled silhouettes (e.g. Apex character whose armor is one color
    # against a similar-toned terrain — motion-diff fills the body interior) are
    # legitimate humanoid signals; only reject if proportions match a wall/panel.
    humanoid_silhouette = (
        aspect >= 1.55
        and bw < frame_w * 0.12
        and bh > frame_h * 0.11
    )
    if fill > 0.94 and len(parts) <= 1 and not humanoid_silhouette:
        return RejectReason.SOLID_WALL
    if fill > 0.97 and not humanoid_silhouette:
        return RejectReason.SOLID_WALL
    if len(parts) == 1 and parts[0].aspect_hw > 5.0:
        return RejectReason.ARCHITECTURE_PANEL
    if aspect > 8.0:
        return RejectReason.ARCHITECTURE_PANEL
    if len(parts) == 1:
        p = parts[0]
        if _is_health_bar(p, _scale(frame_w, frame_h)):
            return RejectReason.HEALTH_BAR_ONLY
        if _is_diamond_sign(p, _scale(frame_w, frame_h)):
            return RejectReason.DIAMOND_SIGN
        if (
            p.extent >= 0.82
            and p.area >= 4500 * _scale(frame_w, frame_h) ** 2
            and aspect < 1.08
        ):
            return RejectReason.SOLID_WALL

    if aspect < 1.65 and bh <= 22 * _scale(frame_w, frame_h) and len(parts) <= 2:
        return RejectReason.HEALTH_BAR_ONLY
    if len(parts) == 1 and all(_is_health_bar(p, _scale(frame_w, frame_h)) for p in parts):
        return RejectReason.HEALTH_BAR_ONLY
    return None


def _figure_aim_point(
    parts: list[_RedPart],
    mask: np.ndarray,
    bx: int,
    by: int,
    bw: int,
    bh: int,
    torso_fraction: float = 0.38,
    aim_y_lo_frac: float = _BODY_Y_LO_FRAC,
    aim_y_hi_frac: float = _BODY_Y_HI_FRAC,
) -> tuple[float, float]:
    """
    Upper-chest anchor from mask mass in torso band; clamped below head, inside bbox.
    """
    frac = max(0.34, min(0.46, torso_fraction))
    ax = bx + bw * 0.5
    ay = by + bh * frac

    chest = _mask_chest_anchor(mask, bx, by, bw, bh, y0f=0.32, y1f=0.50)
    if chest is not None:
        ax = 0.40 * ax + 0.60 * chest[0]
        ay = 0.50 * ay + 0.50 * chest[1]

    y_lo = by + bh * aim_y_lo_frac
    y_hi = by + bh * aim_y_hi_frac

    if len(parts) >= 2:
        torsos = [p for p in parts if p.role == PartRole.TORSO]
        if torsos:
            t = max(torsos, key=lambda p: p.area)
            tcy = max(y_lo, min(y_hi, t.cy))
            # D5 (audit): increased torso-role X blend 0.12 -> 0.45 and
            # clamp X to the torso part's x-extent. The previous 0.12 blend
            # was too weak to overcome a sideways character's gun-arm pull
            # on the merged-bbox centre. With X clamped to [t.x, t.x+t.w]
            # the dot stays inside the actual torso column even if the
            # outer bbox includes an extended arm.
            ax = 0.55 * ax + 0.45 * t.cx
            ay = 0.82 * ay + 0.18 * tcy
            ax = max(float(t.x), min(float(t.x + t.w), ax))
        else:
            cx_parts = float(np.mean([p.cx for p in parts]))
            cy_parts = float(np.mean([max(y_lo, min(y_hi, p.cy)) for p in parts]))
            ax = 0.90 * ax + 0.10 * cx_parts
            ay = 0.80 * ay + 0.20 * cy_parts

    return _clamp_aim_to_body_bbox(ax, ay, bx, by, bw, bh, parts, frac, aim_y_lo_frac, aim_y_hi_frac)


def analyze_figure(
    parts: list[_RedPart],
    mask: np.ndarray,
    frame_w: int,
    frame_h: int,
    *,
    torso_aim_fraction: float = 0.38,
    body_shape_min_score: float | None = None,
    head_score_weight: float = 0.26,
    torso_score_weight: float = 0.26,
    limb_stack_score_weight: float = 0.22,
    aim_y_min_fraction: float = 0.28,
    aim_y_max_fraction: float = 0.52,
) -> _FigureAnalysis:
    scale = _scale(frame_w, frame_h)
    bx, by, bw, bh = _cluster_bbox(parts)
    total_area = sum(p.area for p in parts)
    _classify_parts(parts, bh, scale)

    v_score, fill = _vertical_profile_score(mask, bx, by, bw, bh)
    hard = _hard_reject(parts, bx, by, bw, bh, frame_w, frame_h, fill)
    center_y = frame_h * 0.5
    if hard is None and _is_floating_cluster(bx, by, bw, bh, frame_h, center_y, parts, scale):
        hard = RejectReason.FLOATING_CLUSTER
    if hard is not None:
        return _FigureAnalysis(
            accepted=False,
            reject_reason=hard,
            body_shape_score=0.0,
            head_score=0.0,
            torso_score=0.0,
            limb_stack_score=0.0,
            vertical_profile_score=v_score,
            geometry_score=0.0,
            fill_ratio=fill,
            aspect=bh / max(bw, 1),
            bx=bx,
            by=by,
            bw=bw,
            bh=bh,
            part_count=len(parts),
            aim_x=bx + bw * 0.5,
            aim_y=by + bh * 0.36,
            total_area=total_area,
            solidity=total_area / max(bw * bh, 1),
            debug_detail=hard.value,
        )

    head_s = _score_head(parts, mask, bx, by, bw, bh, scale)
    torso_s = _score_torso(parts, mask, bx, by, bw, bh)
    limb_s = _score_limb_stack(parts, scale)
    geom_s = _score_geometry(bw, bh, frame_w, frame_h, fill)
    align_s = _part_alignment_score(parts, scale)
    max_circ = _max_part_circularity(parts)

    struct_reject = _body_structure_reject(
        parts, bx, by, bw, bh, frame_h, frame_w, scale,
        head_s, torso_s, limb_s, v_score, fill,
    )
    if struct_reject is not None:
        return _FigureAnalysis(
            accepted=False,
            reject_reason=struct_reject,
            body_shape_score=0.0,
            head_score=head_s,
            torso_score=torso_s,
            limb_stack_score=limb_s,
            vertical_profile_score=v_score,
            geometry_score=geom_s,
            fill_ratio=fill,
            aspect=bh / max(bw, 1),
            bx=bx, by=by, bw=bw, bh=bh,
            part_count=len(parts),
            aim_x=bx + bw * 0.5,
            aim_y=by + bh * 0.36,
            total_area=total_area,
            solidity=total_area / max(bw * bh, 1),
            debug_detail=(
                f"{struct_reject.value} circ={max_circ:.2f} align={align_s:.2f} "
                f"head={head_s:.2f} torso={torso_s:.2f} limb={limb_s:.2f} vert={v_score:.2f}"
            ),
        )

    w_sum = max(0.01, head_score_weight + torso_score_weight + limb_stack_score_weight + 0.26)
    body_shape = (
        head_s * head_score_weight
        + torso_s * torso_score_weight
        + limb_s * limb_stack_score_weight
        + v_score * (0.14 / w_sum)
        + geom_s * (0.12 / w_sum)
    )
    if len(parts) >= 2:
        body_shape += 0.06
    if head_s >= 0.45 and torso_s >= 0.35:
        body_shape += 0.08
    if limb_s >= 0.55 and v_score >= 0.4:
        body_shape += 0.05

    body_shape = min(1.0, body_shape)
    aspect = bh / max(bw, 1)
    if body_shape_min_score is not None:
        full_min = float(body_shape_min_score)
        partial_min = min(full_min, _MIN_BODY_SHAPE_PARTIAL)
    else:
        full_min = _MIN_BODY_SHAPE_ACCEPT
        partial_min = _MIN_BODY_SHAPE_PARTIAL
    min_accept = partial_min if len(parts) <= 2 else full_min
    has_head_part = any(p.role == PartRole.HEAD for p in parts)
    structure_ok = (
        (has_head_part and torso_s >= 0.28 and (limb_s >= 0.32 or len(parts) >= 3))
        or (torso_s >= 0.30 and limb_s >= 0.40 and len(parts) >= 2)
        or (len(parts) >= 3 and limb_s >= 0.42 and v_score >= 0.30 and aspect >= 1.35)
        or (aspect >= 1.55 and bh >= 65 * scale and align_s >= 0.45 and v_score >= 0.28)
    )
    if len(parts) == 1:
        if (
            aspect >= 1.40
            and torso_s >= 0.28
            and v_score >= 0.30
            and fill <= 0.92
            and bh >= 50 * scale
        ):
            structure_ok = True
        elif (
            aspect >= 1.55
            and bw < frame_w * 0.12
            and bh >= 70 * scale
            and torso_s >= 0.25
        ):
            # Tall narrow uniform-color humanoid silhouette: shape matches a body
            # column even though the mask is fully filled. Accept on geometry.
            structure_ok = True
        elif v_score < 0.35 or fill > 0.88:
            structure_ok = False
    foot_y = by + bh
    if foot_y < frame_h * 0.42 and len(parts) < 3:
        structure_ok = False

    accepted = body_shape >= min_accept and structure_ok
    reason = RejectReason.OK if accepted else RejectReason.NO_BODY_STRUCTURE
    if accepted and body_shape < min_accept:
        reason = RejectReason.LOW_SCORE

    ax, ay = _figure_aim_point(parts, mask, bx, by, bw, bh, torso_aim_fraction, aim_y_min_fraction, aim_y_max_fraction)
    if not _aim_inside_body_bbox(ax, ay, bx, by, bw, bh):
        return _FigureAnalysis(
            accepted=False,
            reject_reason=RejectReason.SKY_BLOB,
            body_shape_score=body_shape,
            head_score=head_s,
            torso_score=torso_s,
            limb_stack_score=limb_s,
            vertical_profile_score=v_score,
            geometry_score=geom_s,
            fill_ratio=fill,
            aspect=aspect,
            bx=bx, by=by, bw=bw, bh=bh,
            part_count=len(parts),
            aim_x=bx + bw * 0.5,
            aim_y=by + bh * 0.40,
            total_area=total_area,
            solidity=total_area / max(bw * bh, 1),
            debug_detail="aim_outside_body_bbox",
        )
    detail = (
        f"parts={len(parts)} head={head_s:.2f} torso={torso_s:.2f} "
        f"limb={limb_s:.2f} vert={v_score:.2f} geom={geom_s:.2f} align={align_s:.2f} "
        f"circ={max_circ:.2f} fill={fill:.2f}"
    )

    return _FigureAnalysis(
        accepted=accepted,
        reject_reason=reason if accepted else (
            RejectReason.LOW_SCORE if body_shape < min_accept else RejectReason.NO_BODY_STRUCTURE
        ),
        body_shape_score=body_shape,
        head_score=head_s,
        torso_score=torso_s,
        limb_stack_score=limb_s,
        vertical_profile_score=v_score,
        geometry_score=geom_s,
        fill_ratio=fill,
        aspect=bh / max(bw, 1),
        bx=bx,
        by=by,
        bw=bw,
        bh=bh,
        part_count=len(parts),
        aim_x=ax,
        aim_y=ay,
        total_area=total_area,
        solidity=total_area / max(bw * bh, 1),
        debug_detail=detail,
    )




def _strip_non_body_parts(parts: list[_RedPart], scale: float) -> list[_RedPart]:
    return [p for p in parts if not _is_health_bar(p, scale) and not _is_diamond_sign(p, scale)]


def _filter_scope_reticles(
    parts: list[_RedPart],
    scale: float,
    fov_cx: float | None,
    fov_cy: float | None,
) -> list[_RedPart]:
    """Belt-and-braces: drop reticle artefacts even if they survived contour extraction
    (e.g. when entering via mask paths that did not have a FOV centre to consult)."""
    if fov_cx is None or fov_cy is None:
        return parts
    return [p for p in parts if not _is_scope_reticle(p, scale, fov_cx, fov_cy)]


def _bbox_iou(
    ax: int, ay: int, aw: int, ah: int,
    bx: int, by: int, bw: int, bh: int,
) -> float:
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / max(union, 1)




@dataclass
class CandidateInfo:
    """Per-cluster body-shape analysis (accepted or rejected)."""

    idx: int
    accepted: bool
    reject_reason: str
    body_shape_score: float
    head_score: float
    torso_score: float
    limb_stack_score: float
    vertical_profile_score: float
    aspect: float
    fill_ratio: float
    max_circularity: float
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    aim_x: float
    aim_y: float
    part_count: int
    total_area: float
    distance_to_center: float
    debug_detail: str


def enumerate_candidates(
    frame_bgr: np.ndarray,
    hsv_ranges: list[dict[str, Any]] | None,
    fov_radius: int,
    min_area: float,
    fov_center_x: float | None = None,
    fov_center_y: float | None = None,
    *,
    exclude_bottom_frac: float = _VIEWMODEL_EXCLUDE_FRAC,
    torso_aim_fraction: float = 0.38,
    body_shape_min_score: float | None = None,
    head_score_weight: float = 0.26,
    torso_score_weight: float = 0.26,
    limb_stack_score_weight: float = 0.22,
    aim_y_min_fraction: float = 0.28,
    aim_y_max_fraction: float = 0.52,
    detection_mode: str | None = None,
    context: DetectionContext | None = None,
) -> tuple[list[CandidateInfo], np.ndarray, list[_RedPart]]:
    """All clusters with scores/reject reasons — for debug artifacts (not color-only)."""
    h, w = frame_bgr.shape[:2]
    cx = w / 2.0 if fov_center_x is None else fov_center_x
    cy = h / 2.0 if fov_center_y is None else fov_center_y
    mask = build_detection_mask(frame_bgr, hsv_ranges, detection_mode=detection_mode, context=context)
    fov = _build_fov_mask(h, w, cx, cy, fov_radius)
    vm = _build_viewmodel_exclude_mask(h, w, exclude_bottom_frac)
    mask = cv2.bitwise_and(mask, mask, mask=fov)
    mask = cv2.bitwise_and(mask, mask, mask=vm)
    parts = _extract_parts(mask, w, h, fov_cx=cx, fov_cy=cy)
    clusters = _cluster_parts(parts, w, h)
    max_area = float(h * w) * _MAX_AREA_RATIO
    out: list[CandidateInfo] = []
    scale = _scale(w, h)

    for idx, cluster in enumerate(clusters):
        total = sum(p.area for p in cluster)
        if total < min_area:
            out.append(
                CandidateInfo(
                    idx=idx,
                    accepted=False,
                    reject_reason=RejectReason.TOO_SMALL.value,
                    body_shape_score=0.0,
                    head_score=0.0,
                    torso_score=0.0,
                    limb_stack_score=0.0,
                    vertical_profile_score=0.0,
                    aspect=0.0,
                    fill_ratio=0.0,
                    max_circularity=0.0,
                    bbox_x=0,
                    bbox_y=0,
                    bbox_w=0,
                    bbox_h=0,
                    aim_x=cx,
                    aim_y=cy,
                    part_count=len(cluster),
                    total_area=total,
                    distance_to_center=0.0,
                    debug_detail=f"area={total:.0f} < min={min_area}",
                )
            )
            continue
        if total > max_area:
            out.append(
                CandidateInfo(
                    idx=idx,
                    accepted=False,
                    reject_reason=RejectReason.TOO_LARGE.value,
                    body_shape_score=0.0,
                    head_score=0.0,
                    torso_score=0.0,
                    limb_stack_score=0.0,
                    vertical_profile_score=0.0,
                    aspect=0.0,
                    fill_ratio=0.0,
                    max_circularity=0.0,
                    bbox_x=0,
                    bbox_y=0,
                    bbox_w=0,
                    bbox_h=0,
                    aim_x=cx,
                    aim_y=cy,
                    part_count=len(cluster),
                    total_area=total,
                    distance_to_center=0.0,
                    debug_detail=f"area={total:.0f} too_large",
                )
            )
            continue

        body_parts = _strip_non_body_parts(cluster, scale)
        if not body_parts:
            bx, by, bw, bh = _cluster_bbox(cluster)
            out.append(
                CandidateInfo(
                    idx=idx,
                    accepted=False,
                    reject_reason=RejectReason.NO_BODY_STRUCTURE.value,
                    body_shape_score=0.0,
                    head_score=0.0,
                    torso_score=0.0,
                    limb_stack_score=0.0,
                    vertical_profile_score=0.0,
                    aspect=bh / max(bw, 1),
                    fill_ratio=0.0,
                    max_circularity=_max_part_circularity(cluster),
                    bbox_x=bx,
                    bbox_y=by,
                    bbox_w=bw,
                    bbox_h=bh,
                    aim_x=bx + bw * 0.5,
                    aim_y=by + bh * 0.36,
                    part_count=len(cluster),
                    total_area=total,
                    distance_to_center=float(np.hypot(bx + bw * 0.5 - cx, by + bh * 0.36 - cy)),
                    debug_detail="junk only",
                )
            )
            continue

        fig = analyze_figure(
            body_parts,
            mask,
            w,
            h,
            torso_aim_fraction=torso_aim_fraction,
            body_shape_min_score=body_shape_min_score,
            head_score_weight=head_score_weight,
            torso_score_weight=torso_score_weight,
            limb_stack_score_weight=limb_stack_score_weight,
            aim_y_min_fraction=aim_y_min_fraction,
            aim_y_max_fraction=aim_y_max_fraction,
        )
        dist = float(np.hypot(fig.aim_x - cx, fig.aim_y - cy))
        accepted = fig.accepted and dist <= fov_radius
        reason = fig.reject_reason.value
        if fig.accepted and dist > fov_radius:
            accepted = False
            reason = RejectReason.OUTSIDE_FOV.value

        out.append(
            CandidateInfo(
                idx=idx,
                accepted=accepted,
                reject_reason=reason if not accepted else RejectReason.OK.value,
                body_shape_score=fig.body_shape_score,
                head_score=fig.head_score,
                torso_score=fig.torso_score,
                limb_stack_score=fig.limb_stack_score,
                vertical_profile_score=fig.vertical_profile_score,
                aspect=fig.aspect,
                fill_ratio=fig.fill_ratio,
                max_circularity=_max_part_circularity(body_parts),
                bbox_x=fig.bx,
                bbox_y=fig.by,
                bbox_w=fig.bw,
                bbox_h=fig.bh,
                aim_x=fig.aim_x,
                aim_y=fig.aim_y,
                part_count=fig.part_count,
                total_area=fig.total_area,
                distance_to_center=dist,
                debug_detail=fig.debug_detail,
            )
        )
    return out, mask, parts


def render_debug_artifacts(
    frame_bgr: np.ndarray,
    candidates: list[CandidateInfo],
    selected: Target | None,
    fov_radius: int,
    fov_center_x: float,
    fov_center_y: float,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Composite: original + mask tint + candidate bboxes + upper-chest aim point."""
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    cx, cy = int(round(fov_center_x)), int(round(fov_center_y))

    if mask is not None:
        tint = np.zeros_like(out)
        tint[:, :] = (0, 255, 0)
        out = np.where(mask[:, :, None] > 0, cv2.addWeighted(out, 0.55, tint, 0.45, 0), out)

    cv2.circle(out, (cx, cy), fov_radius, (0, 255, 0), 2)
    cv2.drawMarker(out, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 14, 2)

    for c in candidates:
        color = (0, 220, 0) if c.accepted else (0, 120, 255)
        x, y, bw, bh = c.bbox_x, c.bbox_y, c.bbox_w, c.bbox_h
        if bw <= 0 or bh <= 0:
            continue
        cv2.rectangle(out, (x, y), (x + bw, y + bh), color, 2)
        label = f"{c.idx}:{c.reject_reason[:12]} b={c.body_shape_score:.2f}"
        cv2.putText(
            out,
            label,
            (x, max(14, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )
        ax, ay = int(c.aim_x), int(c.aim_y)
        cv2.circle(out, (ax, ay), 4, color, -1)

    if selected is not None:
        x, y, bw, bh = selected.bbox_x, selected.bbox_y, selected.bbox_w, selected.bbox_h
        cv2.rectangle(out, (x, y), (x + bw, y + bh), (0, 0, 255), 3)
        ax, ay = int(selected.centroid_x), int(selected.centroid_y)
        cv2.drawMarker(out, (ax, ay), (0, 255, 255), cv2.MARKER_CROSS, 16, 2)
        chest_y = int(ay)
        chest_x = int(ax)
        cv2.circle(out, (chest_x, chest_y), 6, (255, 0, 255), 2)
        cv2.putText(
            out,
            f"SELECT body={selected.body_shape_score:.2f} h={selected.head_score:.2f} "
            f"t={selected.torso_score:.2f} l={selected.limb_stack_score:.2f}",
            (8, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(out, "SELECT: none", (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 120, 255), 2)

    return out


def inspect_frame(
    frame_bgr: np.ndarray,
    hsv_ranges: list[dict[str, Any]] | None,
    fov_radius: int,
    fov_center_x: float | None = None,
    fov_center_y: float | None = None,
    *,
    detection_mode: str | None = None,
) -> dict[str, Any]:
    """Dump mask/part/cluster stats for tuning on real screenshots."""
    h, w = frame_bgr.shape[:2]
    cx = w / 2.0 if fov_center_x is None else fov_center_x
    cy = h / 2.0 if fov_center_y is None else fov_center_y
    mask = build_detection_mask(frame_bgr, hsv_ranges, detection_mode=detection_mode)
    fov = _build_fov_mask(h, w, cx, cy, fov_radius)
    mask = cv2.bitwise_and(mask, mask, mask=fov)
    parts = _extract_parts(mask, w, h, fov_cx=cx, fov_cy=cy)
    clusters = _cluster_parts(parts, w, h)
    report: dict[str, Any] = {
        "frame": (w, h),
        "mask_pixels": int((mask > 0).sum()),
        "parts": len(parts),
        "clusters": len(clusters),
        "clusters_detail": [],
    }
    for i, cl in enumerate(clusters):
        clean = _strip_non_body_parts(cl, _scale(w, h))
        fig = analyze_figure(clean if clean else cl, mask, w, h)
        report["clusters_detail"].append({
            "idx": i,
            "parts": len(cl),
            "reject": fig.reject_reason.value,
            "body": round(fig.body_shape_score, 3),
            "head": round(fig.head_score, 3),
            "torso": round(fig.torso_score, 3),
            "limb": round(fig.limb_stack_score, 3),
            "fill": round(fig.fill_ratio, 3),
            "bbox": (fig.bx, fig.by, fig.bw, fig.bh),
        })
    return report

def _collect_candidates(
    frame_bgr: np.ndarray,
    hsv_ranges: list[dict[str, Any]] | None,
    fov_radius: int,
    min_area: float,
    cx: float,
    cy: float,
    *,
    exclude_bottom_frac: float = _VIEWMODEL_EXCLUDE_FRAC,
    torso_aim_fraction: float = 0.38,
    body_shape_min_score: float | None = None,
    head_score_weight: float = 0.26,
    torso_score_weight: float = 0.26,
    limb_stack_score_weight: float = 0.22,
    aim_y_min_fraction: float = 0.28,
    aim_y_max_fraction: float = 0.52,
    debug: bool = False,
    detection_mode: str | None = None,
    context: DetectionContext | None = None,
    # D7 (audit): aspect / solidity profile filters were previously dropped
    # silently in find_best_target via `_ = (...)`. Plumb them through so
    # the profile-supplied tuning actually filters candidates.
    min_aspect: float | None = None,
    max_aspect: float | None = None,
    min_solidity: float | None = None,
) -> tuple[list[Target], list[str]]:
    global _LAST_DEBUG_LINES
    h, w = frame_bgr.shape[:2]
    mask = build_detection_mask(frame_bgr, hsv_ranges, detection_mode=detection_mode, context=context)
    fov = _build_fov_mask(h, w, cx, cy, fov_radius)
    vm = _build_viewmodel_exclude_mask(h, w, exclude_bottom_frac)
    mask = cv2.bitwise_and(mask, mask, mask=fov)
    mask = cv2.bitwise_and(mask, mask, mask=vm)

    # FILLED red-enemy mask (separate from `mask` above which only fuses
    # the red outline for clustering). Used to compute Target.red_coverage:
    # a per-bbox fraction that lets score_target soften the single-part
    # penalty on a clearly red humanoid that the shape mask collapses to a
    # single connected blob. Only built in apex mode — keeps shape/hsv/
    # hybrid modes free of any color-bias side-effects.
    red_filled = (
        build_red_enemy_mask(frame_bgr)
        if _normalize_detection_mode(detection_mode) == DETECTION_MODE_APEX
        else None
    )

    parts = _extract_parts(mask, w, h, fov_cx=cx, fov_cy=cy)
    clusters = _cluster_parts(parts, w, h)
    max_area = float(h * w) * _MAX_AREA_RATIO
    targets: list[Target] = []
    lines: list[str] = []

    for idx, cluster in enumerate(clusters):
        total = sum(p.area for p in cluster)
        if total < min_area or total > max_area:
            if debug:
                lines.append(f"cand[{idx}] skip area={total:.0f}")
            continue

        body_parts = _strip_non_body_parts(cluster, _scale(w, h))
        if not body_parts:
            if debug:
                lines.append(f"cand[{idx}] {RejectReason.NO_BODY_STRUCTURE.value} (junk only)")
            continue
        fig = analyze_figure(
            body_parts, mask, w, h,
            torso_aim_fraction=torso_aim_fraction,
            body_shape_min_score=body_shape_min_score,
            head_score_weight=head_score_weight,
            torso_score_weight=torso_score_weight,
            limb_stack_score_weight=limb_stack_score_weight,
            aim_y_min_fraction=aim_y_min_fraction,
            aim_y_max_fraction=aim_y_max_fraction,
        )
        mc = _max_part_circularity(body_parts)
        dist_c = float(np.hypot(fig.aim_x - cx, fig.aim_y - cy))
        line = (
            f"cand[{idx}] reject={fig.reject_reason.value} body={fig.body_shape_score:.2f} "
            f"head={fig.head_score:.2f} torso={fig.torso_score:.2f} limb={fig.limb_stack_score:.2f} "
            f"aspect={fig.aspect:.2f} fill={fig.fill_ratio:.2f} circ={mc:.2f} "
            f"bbox=({fig.bx},{fig.by},{fig.bw}x{fig.bh}) area={fig.total_area:.0f} "
            f"dist={dist_c:.0f} parts={fig.part_count} | {fig.debug_detail}"
        )
        if debug:
            lines.append(line)

        if not fig.accepted:
            continue

        # D7 (audit): apply profile-supplied aspect / solidity filters.
        if fig.bw > 0 and fig.bh > 0:
            asp = fig.bh / float(max(1, fig.bw))
            if min_aspect is not None and min_aspect > 0 and asp < float(min_aspect):
                if debug:
                    lines.append(
                        f"cand[{idx}] drop aspect={asp:.2f} < min_aspect={min_aspect:.2f}"
                    )
                continue
            if max_aspect is not None and max_aspect > 0 and asp > float(max_aspect):
                if debug:
                    lines.append(
                        f"cand[{idx}] drop aspect={asp:.2f} > max_aspect={max_aspect:.2f}"
                    )
                continue
        if min_solidity is not None and min_solidity > 0 and fig.solidity < float(min_solidity):
            if debug:
                lines.append(
                    f"cand[{idx}] drop solidity={fig.solidity:.2f} < min={min_solidity:.2f}"
                )
            continue

        aim_x, aim_y = clamp_point_to_fov(fig.aim_x, fig.aim_y, cx, cy, float(fov_radius))
        dist = float(np.hypot(aim_x - cx, aim_y - cy))
        if dist > float(fov_radius) * 1.02:
            if debug:
                lines.append(f"cand[{idx}] {RejectReason.OUTSIDE_FOV.value}")
            continue
        # Reject screen-edge junk: bbox center far outside FOV
        bcx = fig.bx + fig.bw * 0.5
        bcy = fig.by + fig.bh * 0.5
        if float(np.hypot(bcx - cx, bcy - cy)) > float(fov_radius) * 1.08:
            if debug:
                lines.append(f"cand[{idx}] {RejectReason.OUTSIDE_FOV.value} bbox_center")
            continue

        red_cov = 0.0
        if red_filled is not None and fig.bw > 0 and fig.bh > 0:
            x0 = max(0, int(fig.bx))
            y0 = max(0, int(fig.by))
            x1 = min(red_filled.shape[1], int(fig.bx + fig.bw))
            y1 = min(red_filled.shape[0], int(fig.by + fig.bh))
            if x1 > x0 and y1 > y0:
                roi = red_filled[y0:y1, x0:x1]
                if roi.size > 0:
                    red_cov = float((roi > 0).sum()) / float(roi.size)

        targets.append(
            Target(
                centroid_x=aim_x,
                centroid_y=aim_y,
                area=fig.total_area,
                distance_to_center=dist,
                bbox_x=fig.bx,
                bbox_y=fig.by,
                bbox_w=fig.bw,
                bbox_h=fig.bh,
                solidity=fig.solidity,
                humanoid_score=fig.body_shape_score,
                part_count=fig.part_count,
                body_shape_score=fig.body_shape_score,
                head_score=fig.head_score,
                torso_score=fig.torso_score,
                limb_stack_score=fig.limb_stack_score,
                red_coverage=red_cov,
                fill_ratio=float(fig.fill_ratio),
                max_circularity=float(_max_part_circularity(body_parts)),
                reject_reason=fig.reject_reason.value,
            )
        )

    if debug:
        lines.append(f"accepted={len(targets)} from {len(clusters)} clusters / {len(parts)} parts")
        _LAST_DEBUG_LINES = lines
    return targets, lines


def _closeness_bonus(target: Target, fov_radius: float) -> float:
    """Closer-enemy preference: bigger bbox height = closer enemy.

    Capped at 1.5x FOV radius so giant nearby targets don't saturate the
    score. The weight is tuned so a 2x-taller body clearly outscores a
    similarly-positioned smaller body, but a marginally taller body (≤1.1x)
    does NOT override a center-distance advantage from a similarly-sized
    target — see ``tests/test_target_size_priority.py``.
    """
    bbox_h_eff = max(0.0, float(target.bbox_h))
    closeness_unit = min(bbox_h_eff, fov_radius * 1.5) / max(fov_radius, 1.0)
    return closeness_unit * fov_radius * 0.42


def score_target(
    target: Target,
    fov_radius: float,
    distance_weight: float,
    area_weight: float,
    *,
    center_y: float | None = None,
    motion_overlap: float = 0.0,
    include_closeness: bool = False,
) -> float:
    body_term = target.body_shape_score * fov_radius * 1.15
    dist_term = max(0.0, fov_radius - target.distance_to_center) * distance_weight * 0.35
    area_term = min(math.sqrt(target.area), 80.0) * area_weight
    closeness_term = _closeness_bonus(target, fov_radius) if include_closeness else 0.0
    # Motion-overlap bonus rewards candidates whose bbox covers an inter-frame
    # diff region — these are confirmed *moving* silhouettes (real Apex enemies)
    # vs static red walls/panels whose shape may look humanoid by accident.
    motion_bonus = min(1.0, max(0.0, motion_overlap)) * fov_radius * 0.55
    # Red-coverage bonus: Apex enemies whose body is materially covered by
    # the red-enemy HSV mask are almost always real targets (the only
    # in-game source of that hue at that saturation is the enemy highlight).
    # The bonus is gated on bbox height so red props (test panels, blurred
    # plate stubs) with accidental humanoid aspect cannot exploit it; only
    # candidates tall enough to be a real Apex character at engagement
    # distance receive the additive boost.
    red_cov = min(1.0, max(0.0, target.red_coverage))
    # D3 (audit): drop the 110 px hard pixel floor — it cut off any real
    # enemy beyond medium range on a 720p capture (a fully-visible character
    # at long range is < 110 px tall). Use 60 px or 10 % of bbox height as
    # the floor instead. The aspect-ratio requirement stays so red boards
    # (red panels, square hazard signs) still cannot grab the bonus.
    red_humanoid_height = target.bbox_h >= 60.0 and target.bbox_h >= target.bbox_w * 1.55
    red_bonus = red_cov * fov_radius * 0.15 if red_humanoid_height else 0.0
    penalty = 0.0
    if center_y is not None:
        if target.centroid_y < center_y:
            penalty += (center_y - target.centroid_y) * 2.8
        if target.bbox_h > 0:
            chest_hi = target.bbox_y + target.bbox_h * 0.48
            if target.centroid_y < chest_hi - target.bbox_h * 0.06:
                penalty += (chest_hi - target.centroid_y) * 4.5
        foot_y = target.bbox_y + target.bbox_h
        if foot_y < center_y + fov_radius * 0.15:
            penalty += fov_radius * 2.2
    if target.bbox_w >= target.bbox_h:
        penalty += fov_radius * 0.8
    # Single-part target: harsh penalty by default; soft penalty when EITHER
    # motion confirms a real moving body OR the bbox is clearly red-covered
    # (a real Apex enemy whose armour merged into one connected silhouette
    # against same-toned terrain — the live-game case the user reported).
    # Red-coverage softener is gated by a stricter height threshold (≥110 px)
    # so blurred-plate synthetic blobs (~98 px tall) cannot exploit it.
    is_humanoid_column = (
        target.bbox_h >= 80.0
        and target.bbox_h >= target.bbox_w * 1.55
        and target.body_shape_score >= 0.45
    )
    # D3 (audit): lower height floor 110 -> max(60, 10 % of bbox_h scaled
    # to frame) and lower red_cov threshold 0.35 -> 0.12 for the softener
    # path. The previous 0.35 was unreachable for a thin glow ring whose
    # theoretical max bbox-fill is ~0.10. Also accept a "red outline"
    # perimeter-ratio path: the red mask covers most of the bbox perimeter
    # even when interior red_cov is low (Horizon-style glow).
    # D3 (audit) cont: the softener also rejects solid round blobs whose
    # max_circularity is high (>0.55). Real Apex characters have jagged
    # outlines (head/torso/limb transitions) and never breach ~0.50; a
    # blurred test blob smooths to ~0.65-0.75 and would otherwise exploit
    # the lowered red_cov threshold.
    red_humanoid_column = (
        target.bbox_h >= 60.0
        and target.bbox_h >= target.bbox_w * 1.55
        and target.body_shape_score >= 0.45
        and target.max_circularity < 0.55
    )
    motion_confirms = (
        is_humanoid_column
        and motion_overlap >= 0.12
        and target.max_circularity < 0.60
    )
    # Approximate outline-coverage: a thin ring of width 2 around a bbox
    # of dimensions w*h occupies ~2*(w+h) pixels out of w*h area. If actual
    # red_coverage clears even a fraction of that, it's almost certainly a
    # real red outline rather than a stray red prop.
    bw_eff = max(1.0, float(target.bbox_w))
    bh_eff = max(1.0, float(target.bbox_h))
    outline_floor = max(0.04, min(0.25, 2.0 * (bw_eff + bh_eff) / (bw_eff * bh_eff)))
    red_confirms = red_humanoid_column and (red_cov >= 0.12 or red_cov >= outline_floor * 0.6)
    if target.part_count < 2:
        if motion_confirms or red_confirms:
            penalty += fov_radius * 0.35
        else:
            penalty += fov_radius * 1.4
    if target.bbox_h < target.bbox_w * 1.25:
        penalty += fov_radius * 1.8
    # The body<0.48 penalty is normally a strong rejection of marginal
    # silhouettes — but when motion or red coverage independently confirm
    # a tall humanoid bbox, we trust the geometric / chromatic evidence
    # over a small dip in the analyze_figure body score (which is fragile
    # against motion-fused masks that saturate the bbox fill).
    # D6 (audit): replace flat +0.9*fov_radius cliff at body_shape<0.48 with
    # a linear ramp scaled by how far below 0.48 we are. A score of 0.47
    # used to drop a full 0.9*fov penalty (~200 px @ FOV 220) — enough to
    # tip a marginal real character below the confidence floor. Now 0.47
    # only loses 0.01 * 2.0 * fov ≈ 4 px; 0.30 loses 0.18 * 2.0 * fov ≈ 80;
    # 0.00 still loses ~190 (close to the old cap). Motion/red softener
    # path still skips the penalty entirely as before.
    if target.body_shape_score < 0.48 and not (motion_confirms or red_confirms):
        penalty += max(0.0, 0.48 - target.body_shape_score) * fov_radius * 2.0
    return body_term + dist_term + area_term + closeness_term + motion_bonus + red_bonus - penalty


def _normalize_confidence(raw: float, fov_radius: float, body_shape: float = 0.0) -> float:
    from_dist = raw / max(fov_radius * 2.8, 1.0)
    base = 0.55 * body_shape + 0.45 * from_dist
    return max(0.0, min(1.0, base))


def find_best_target(
    frame_bgr: np.ndarray,
    hsv_ranges: list[dict[str, Any]] | None,
    fov_radius: int,
    min_area: float,
    fov_center_x: float | None = None,
    fov_center_y: float | None = None,
    *,
    sticky_target: Target | None = None,
    stickiness_pixels: float = 90.0,
    distance_weight: float = 2.0,
    area_weight: float = 0.015,
    min_height_px: float = 0.0,
    min_aspect: float | None = None,
    max_aspect: float | None = None,
    min_solidity: float | None = None,
    torso_aim_fraction: float = 0.38,
    body_shape_min_score: float | None = None,
    head_score_weight: float = 0.26,
    torso_score_weight: float = 0.26,
    limb_stack_score_weight: float = 0.22,
    aim_y_min_fraction: float = 0.28,
    aim_y_max_fraction: float = 0.52,
    min_confidence: float = _MIN_CONFIDENCE,
    exclude_bottom_frac: float = _VIEWMODEL_EXCLUDE_FRAC,
    debug: bool = False,
    detection_mode: str | None = None,
    context: DetectionContext | None = None,
    currently_locked: bool = False,
) -> DetectionResult:
    h, w = frame_bgr.shape[:2]
    cx = w / 2.0 if fov_center_x is None else fov_center_x
    cy = h / 2.0 if fov_center_y is None else fov_center_y

    resolved_mode = detection_mode
    if resolved_mode is None:
        resolved_mode = DETECTION_MODE_DEFAULT

    candidates, dbg = _collect_candidates(
        frame_bgr,
        hsv_ranges,
        fov_radius,
        min_area,
        cx,
        cy,
        exclude_bottom_frac=exclude_bottom_frac,
        torso_aim_fraction=torso_aim_fraction,
        body_shape_min_score=body_shape_min_score,
        head_score_weight=head_score_weight,
        torso_score_weight=torso_score_weight,
        limb_stack_score_weight=limb_stack_score_weight,
        aim_y_min_fraction=aim_y_min_fraction,
        aim_y_max_fraction=aim_y_max_fraction,
        debug=debug,
        detection_mode=resolved_mode,
        context=context,
        min_aspect=min_aspect,
        max_aspect=max_aspect,
        min_solidity=min_solidity,
    )
    if min_height_px is not None and min_height_px > 0:
        before = len(candidates)
        candidates = [t for t in candidates if t.bbox_h >= min_height_px]
        if len(candidates) < before:
            dbg.append(f"min_height filter: {before} -> {len(candidates)} (min_h={min_height_px})")
    if not candidates:
        return DetectionResult(None, 0, 0.0, debug_lines=dbg, active=False)

    def _motion_overlap(t: Target) -> float:
        if context is None:
            return 0.0
        return context.motion_coverage_with_memory(t.bbox_x, t.bbox_y, t.bbox_w, t.bbox_h)

    def rank(t: Target) -> float:
        # Ranking includes closeness so a clearly larger (closer) enemy beats a
        # similarly-positioned smaller one. The bonus is NOT applied to the
        # confidence calculation — confidence must measure detection quality,
        # not target proximity, otherwise small far enemies would never make
        # the threshold and large but low-quality blobs would always pass.
        return score_target(
            t,
            float(fov_radius),
            distance_weight,
            area_weight,
            center_y=cy,
            motion_overlap=_motion_overlap(t),
            include_closeness=True,
        )

    def finalize(t: Target) -> Target:
        raw = score_target(
            t,
            float(fov_radius),
            distance_weight,
            area_weight,
            center_y=cy,
            motion_overlap=_motion_overlap(t),
            include_closeness=False,
        )
        t.confidence = _normalize_confidence(raw, float(fov_radius), t.body_shape_score)
        return t

    def _refresh_validation(t: Target) -> None:
        if context is None:
            return
        live = context.motion_coverage_ratio(t.bbox_x, t.bbox_y, t.bbox_w, t.bbox_h)
        if live >= context.motion_validate_threshold:
            context.note_motion_validated(t.bbox_x, t.bbox_y, t.bbox_w, t.bbox_h)

    if sticky_target is not None and stickiness_pixels > 0:
        pool = []
        for t in candidates:
            iou = _bbox_iou(
                sticky_target.bbox_x, sticky_target.bbox_y, sticky_target.bbox_w, sticky_target.bbox_h,
                t.bbox_x, t.bbox_y, t.bbox_w, t.bbox_h,
            )
            dist = math.hypot(t.centroid_x - sticky_target.centroid_x, t.centroid_y - sticky_target.centroid_y)
            if iou >= 0.12 or dist <= stickiness_pixels:
                pool.append(t)
        if pool:
            sticky_best = max(pool, key=rank)
            global_best = max(candidates, key=rank)
            # Size-based switch guard: a clearly larger (closer) enemy can
            # win, but the existing _STICKY_SWITCH_RATIO must still bound how
            # easily two similarly-sized enemies flicker the lock.
            sticky_h = max(1.0, float(sticky_best.bbox_h))
            global_h = max(1.0, float(global_best.bbox_h))
            size_ratio = global_h / sticky_h
            switch_allowed = (
                rank(global_best) > rank(sticky_best) * _STICKY_SWITCH_RATIO
                and global_best.body_shape_score > sticky_best.body_shape_score + 0.15
                and global_best.part_count >= 2
                and global_best.bbox_h >= global_best.bbox_w * 1.25
                and _bbox_iou(
                    sticky_target.bbox_x, sticky_target.bbox_y, sticky_target.bbox_w, sticky_target.bbox_h,
                    global_best.bbox_x, global_best.bbox_y, global_best.bbox_w, global_best.bbox_h,
                ) < 0.08
                and size_ratio >= 1.8
            )
            if switch_allowed:
                chosen = finalize(global_best)
            else:
                chosen = finalize(sticky_best)
            # Locked-target confidence floor: when the runtime indicates the
            # current candidate is the same enemy we were already locked on
            # (sticky overlap proves this), do NOT drop below the confidence
            # threshold on a single weak frame. The motion-memory channel
            # already softens transient dips, but a hard floor here prevents
            # one bad detection from breaking the lock and causing a re-acquire
            # flicker visible to the user as a "glitching" dot.
            iou_lock = _bbox_iou(
                sticky_target.bbox_x, sticky_target.bbox_y,
                sticky_target.bbox_w, sticky_target.bbox_h,
                chosen.bbox_x, chosen.bbox_y, chosen.bbox_w, chosen.bbox_h,
            )
            lock_dist = math.hypot(
                chosen.centroid_x - sticky_target.centroid_x,
                chosen.centroid_y - sticky_target.centroid_y,
            )
            lock_overlap = currently_locked and (iou_lock >= 0.5 or lock_dist < 60.0)
            if chosen.confidence < min_confidence:
                if lock_overlap:
                    floor = max(min_confidence, _MIN_CONFIDENCE)
                    chosen.confidence = floor
                    dbg.append(
                        f"locked floor applied conf->{floor:.2f} iou={iou_lock:.2f} d={lock_dist:.0f}"
                    )
                else:
                    dbg.append(f"selected reject low_conf={chosen.confidence:.2f}")
                    return DetectionResult(None, len(candidates), chosen.confidence, debug_lines=dbg, active=False)
            _refresh_validation(chosen)
            dbg.append(
                f"SELECTED body={chosen.body_shape_score:.2f} head={chosen.head_score:.2f} "
                f"dist={chosen.distance_to_center:.0f} reason={chosen.reject_reason}"
            )
            return DetectionResult(chosen, len(candidates), chosen.confidence, debug_lines=dbg, active=True)
        dbg.append("sticky lost lock (no overlapping candidate)")

    best = finalize(max(candidates, key=rank))
    if best.confidence < min_confidence:
        dbg.append(f"selected reject low_conf={best.confidence:.2f}")
        return DetectionResult(None, len(candidates), best.confidence, debug_lines=dbg, active=False)
    _refresh_validation(best)
    dbg.append(
        f"SELECTED body={best.body_shape_score:.2f} head={best.head_score:.2f} "
        f"dist={best.distance_to_center:.0f} bbox={best.bbox_w}x{best.bbox_h}"
    )
    return DetectionResult(best, len(candidates), best.confidence, debug_lines=dbg, active=True)


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
        aspect_wh=w / max(h, 1),
        aspect_hw=h / max(w, 1),
        solidity=0.5,
        extent=0.5,
        circularity=0.3,
    )
    fig = analyze_figure([part], np.ones((frame_h, frame_w), dtype=np.uint8) * 255, frame_w, frame_h)
    return fig.accepted, w, h, fig.solidity, fig.body_shape_score


def draw_debug(
    frame_bgr: np.ndarray,
    target: Target | None,
    fov_radius: int,
    fov_center_x: float | None = None,
    fov_center_y: float | None = None,
    magnetism_radius: int | None = None,
    hsv_ranges: list[dict[str, Any]] | None = None,
    stats_lines: list[str] | None = None,
    *,
    detection_debug: list[str] | None = None,
    display_fov_radius: int | None = None,
    debug_show_detect_ring: bool = False,
) -> np.ndarray:
    """Render the OpenCV debug-window frame.

    O1 (audit): the debug ring is now drawn at ``display_fov_radius``
    (matching the on-screen overlay ring the user sees) rather than the
    larger ``fov_radius`` (the detection radius). When ``display_fov_radius``
    is None we fall back to the detection radius; the optional
    ``debug_show_detect_ring`` flag draws a SECOND faint ring at the
    detection radius (off by default) only when explicit comparison is
    needed. Default behaviour now shows a single ring matching the live
    overlay so the "two FOV rings" screenshot artefact cannot recur.
    """
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    cx = int(round(w / 2 if fov_center_x is None else fov_center_x))
    cy = int(round(h / 2 if fov_center_y is None else fov_center_y))

    cx_f = float(cx)
    cy_f = float(cy)
    mask = build_detection_mask(frame_bgr, hsv_ranges, detection_mode=DETECTION_MODE_SHAPE)
    fov = _build_fov_mask(h, w, cx_f, cy_f, fov_radius)
    vm = _build_viewmodel_exclude_mask(h, w, _VIEWMODEL_EXCLUDE_FRAC)
    mask = cv2.bitwise_and(mask, mask, mask=fov)
    mask = cv2.bitwise_and(mask, mask, mask=vm)
    tint = np.zeros_like(out)
    tint[:, :] = (0, 255, 0)
    out = np.where(mask[:, :, None] > 0, cv2.addWeighted(out, 0.5, tint, 0.5, 0), out)

    ring_r = int(display_fov_radius) if display_fov_radius is not None else int(fov_radius)
    cv2.circle(out, (cx, cy), ring_r, (0, 255, 0), 2)
    if debug_show_detect_ring and display_fov_radius is not None and int(display_fov_radius) != int(fov_radius):
        # Optional second faint ring at detection FOV. Off by default so
        # the user only ever sees ONE green ring matching the live overlay.
        cv2.circle(out, (cx, cy), int(fov_radius), (0, 128, 0), 1)
    cv2.drawMarker(out, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 12, 2)
    if target is not None:
        tx, ty = int(target.centroid_x), int(target.centroid_y)
        cv2.circle(out, (tx, ty), 8, (0, 0, 255), 2)
        cv2.line(out, (cx, cy), (tx, ty), (255, 0, 255), 1)

    lines = list(stats_lines or [])
    if target is not None:
        lines.insert(
            0,
            f"ACTIVE body={target.body_shape_score:.2f} h={target.head_score:.2f} "
            f"t={target.torso_score:.2f} l={target.limb_stack_score:.2f}",
        )
    else:
        lines.insert(0, "ACTIVE=no body target")
    if detection_debug:
        lines.extend(detection_debug[-8:])
    y0 = 20
    for i, line in enumerate(lines[:10]):
        cv2.putText(out, line[:72], (8, y0 + i * 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    return out
