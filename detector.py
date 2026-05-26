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
_VIEWMODEL_EXCLUDE_FRAC = 0.28
# Background clutter (distant red props / HUD specks) — single source of truth.
_CLUTTER_MAX_RED_COV = 0.10
_CLUTTER_TORSO_RED_MIN = 0.06
_CLUTTER_MIN_BODY_H = 62.0
_CLUTTER_MIN_AREA = 140.0
_CLUTTER_TINY_AREA = 95.0
_CLUTTER_TINY_BH = 44.0
_CLUTTER_TINY_RED = 0.09
_CLUTTER_SMALL_BH = 54.0
_CLUTTER_SMALL_BW = 30.0
_CLUTTER_SMALL_RED = 0.075
_CLUTTER_ROUND_PARTS = 2
_CLUTTER_ROUND_BH = 64.0
_CLUTTER_ROUND_CIRC = 0.62
_CLUTTER_ROUND_RED = 0.085
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
    VIEWMODEL_COLUMN = "viewmodel_column"
    BACKGROUND_CLUTTER = "background_clutter"
    LOW_SCORE = "low_body_shape_score"

# Shared with target_lock — one sky band + red floor for detector and lock.
SKY_BAND_CENTER_FRAC = 0.40
MIN_ENEMY_RED_COVERAGE = 0.04


def bbox_mid_in_sky_band(bbox_y: float, bbox_h: float, center_y: float) -> bool:
    mid_y = bbox_y + bbox_h * 0.5
    return mid_y < center_y * SKY_BAND_CENTER_FRAC


def bbox_top_in_sky_band(bbox_y: float, center_y: float) -> bool:
    """True when the bbox top edge sits in the upper sky/play boundary."""
    return float(bbox_y) < center_y * SKY_BAND_CENTER_FRAC


def target_is_central_tower_banner_fp(
    target: Target,
    *,
    frame_w: int,
    frame_h: int,
    fov_cx: float,
    motion_overlap: float = 0.0,
) -> bool:
    """Central firing-range tower banners (red icon strips), not humanoids."""
    if frame_w <= 0 or frame_h <= 0:
        return False
    bw = max(1.0, float(target.bbox_w))
    bh = max(1.0, float(target.bbox_h))
    aspect = bh / bw
    if aspect < 1.9 or bw > frame_w * 0.10:
        return False
    col_cx = float(target.bbox_x) + bw * 0.5
    if abs(col_cx - float(fov_cx)) > frame_w * 0.18:
        return False
    red = float(target.red_coverage)
    if motion_overlap >= 0.09:
        return False
    top_frac = float(target.bbox_y) / float(frame_h)
    # gif_166_proof F11 / F59 class: central tower column locks (banner
    # stack + score panel) at ADS start or after a previous lock expires.
    # The column has bbox top in the upper 30% of frame, mid_y above the
    # crosshair (mid_y_frac < ~0.45), medium-narrow width (32-62px) and
    # **low red coverage** (0.03-0.13) because the column is mostly dark
    # panel between the banner shapes — a real close humanoid in the same
    # location has filled red torso (>= 0.13). The head score for this
    # cluster can be high (banner top reads as head) so we do NOT gate on
    # head_score; the red-coverage gate is the discriminator. Run BEFORE
    # the red >= 0.10 early gate so we still catch the F11/F59 0.05-0.10
    # band.
    mid_y_frac = (float(target.bbox_y) + bh * 0.5) / float(frame_h)
    if (
        top_frac < 0.32
        and 32.0 < bw <= max(62.0, frame_w * 0.085)
        and aspect >= 1.6
        and 0.03 <= red < 0.13
        and mid_y_frac < 0.45
    ):
        return True
    if red < 0.10:
        return False
    skinny = bw <= max(30.0, frame_w * 0.065)
    if not skinny:
        return False
    # img1 class: LIVE on tower banner ~(387,106,27x69) — high + saturated static red.
    if top_frac < 0.30 and red >= 0.12:
        return True
    if top_frac < 0.52 and aspect >= 2.8 and red >= 0.14 and bw <= 28:
        return True
    # F62 class: phantom lock on central icon strip ~(395,169,26x65) — aspect
    # 2.0-2.8 band slipped between the two above gates.  Banner geometry:
    # narrow (<=28px), centered, saturated red, in upper-mid frame (<0.42),
    # with weak head signal (icon strips lack a real head/torso transition).
    # Real close dummy in the same band has top_frac>=0.45 (bottom-aligned at
    # chest level), so the 0.42 ceiling preserves test_close_dummy_not_banner_fp.
    if (
        top_frac < 0.42
        and aspect >= 2.0
        and red >= 0.13
        and bw <= 28
        and float(target.head_score) < 0.32
    ):
        return True
    return False


def target_is_environment_column(
    target: Target,
    *,
    center_y: float,
    frame_h: int = 0,
    frame_w: int = 0,
) -> bool:
    """Firing-range tower, banners, tall props — tall bbox with top in sky band."""
    if frame_w <= 0 or frame_h <= 0:
        return False
    bh = float(target.bbox_h)
    bw = max(1.0, float(target.bbox_w))
    if bh < frame_h * 0.22:
        return False
    aspect = bh / bw
    top_sky = bbox_top_in_sky_band(float(target.bbox_y), center_y)
    # Real close-range humanoids (img5_close_ads-style) often have a
    # tall bbox whose top edge crosses into the sky band simply because
    # the character is large in frame.  When the candidate has a
    # *classified* head + torso + limb stack (body_shape>=0.85, all
    # three roles present with strong scores) treat it as humanoid and
    # bypass the column reject.  Environment columns / banners do not
    # produce a complete head+torso+limb decomposition.
    is_classified_humanoid = (
        target.has_classified_torso
        and float(target.body_shape_score) >= 0.85
        and float(target.head_score) >= 0.6
        and float(target.torso_score) >= 0.6
        and float(target.limb_stack_score) >= 0.3
    )
    if is_classified_humanoid:
        return False
    if top_sky and bh >= frame_h * 0.20 and aspect >= 1.35:
        return True
    if (
        top_sky
        and aspect >= 2.0
        and float(target.red_coverage) < 0.07
        and not target.has_classified_torso
    ):
        return True
    if (
        bh >= frame_h * 0.38
        and aspect >= 1.6
        and float(target.red_coverage) < max(MIN_ENEMY_RED_COVERAGE, 0.06)
        and float(target.torso_score) < 0.35
    ):
        return True
    return False


def has_enemy_red_torso_evidence(
    parts: list,
    *,
    torso_score: float,
    red_cov: float,
) -> bool:
    """True when red/torso evidence is real enemy highlight — not mislabeled gun column."""
    if red_cov >= 0.05:
        return True
    if torso_score >= 0.32 and red_cov >= MIN_ENEMY_RED_COVERAGE:
        return any(p.role == PartRole.TORSO for p in parts)
    return False


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
    has_classified_torso: bool = False
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


def _scale(frame_w: int, frame_h: int) -> float:
    return min(frame_w, frame_h) / 1080.0


_FOV_MASK_CACHE: dict[tuple[int, int, int, int, int], np.ndarray] = {}


def _build_fov_mask(height: int, width: int, center_x: float, center_y: float, radius: int) -> np.ndarray:
    cx = round(center_x)
    cy = round(center_y)
    key = (int(height), int(width), int(cx), int(cy), int(radius))
    cached = _FOV_MASK_CACHE.get(key)
    if cached is not None:
        return cached
    y, x = np.ogrid[:height, :width]
    mask = ((x - cx) ** 2 + (y - cy) ** 2 <= radius * radius).astype(np.uint8)
    # Cap cache size to a small handful so we don't leak when the user
    # is scrubbing FOV during tuning.
    if len(_FOV_MASK_CACHE) >= 8:
        _FOV_MASK_CACHE.pop(next(iter(_FOV_MASK_CACHE)))
    _FOV_MASK_CACHE[key] = mask
    return mask


def _build_viewmodel_exclude_mask(height: int, width: int, exclude_bottom_frac: float) -> np.ndarray:
    frac = max(0.0, min(0.5, exclude_bottom_frac))
    if frac <= 0.0:
        return np.ones((height, width), dtype=np.uint8)
    cutoff = int(height * (1.0 - frac))
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[:cutoff, :] = 1
    return mask


def build_hsv_mask(
    frame_bgr: np.ndarray,
    hsv_ranges: list[dict[str, Any]],
    *,
    hsv: np.ndarray | None = None,
) -> np.ndarray:
    if hsv is None:
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


def _bbox_red_coverage(
    mask: np.ndarray,
    bbox_x: int,
    bbox_y: int,
    bbox_w: int,
    bbox_h: int,
) -> float:
    """Fraction of the bbox covered by red mask pixels (0.0-1.0).

    Used in the detector pool_hold paths to validate that the locked
    bbox still has red mass in the *current* frame. When the player
    pans away or the enemy walks offscreen the sticky lock geometry
    sits over empty pixels (gif_166_proof F39-F46 ghost class).
    """
    h, w = mask.shape[:2]
    x0 = max(0, int(bbox_x))
    y0 = max(0, int(bbox_y))
    x1 = min(w, int(bbox_x) + int(bbox_w))
    y1 = min(h, int(bbox_y) + int(bbox_h))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    sub = mask[y0:y1, x0:x1]
    if sub.size == 0:
        return 0.0
    return float((sub > 0).sum()) / float(sub.size)



def _normalize_detection_mode(mode: str | None) -> str:
    m = (mode or DETECTION_MODE_DEFAULT).strip().lower()
    if m not in _VALID_DETECTION_MODES:
        return DETECTION_MODE_DEFAULT
    return m


def build_shape_mask(
    frame_bgr: np.ndarray,
    *,
    gray: np.ndarray | None = None,
) -> np.ndarray:
    """
    Color-free foreground mask: local contrast + edges, morphology to join body plates.
    Works across Apex skin colors; shape scoring rejects UI/HUD blobs.
    """
    h, w = frame_bgr.shape[:2]
    scale = _scale(w, h)
    if gray is None:
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



def build_chroma_spread_mask(
    frame_bgr: np.ndarray,
    *,
    hsv: np.ndarray | None = None,
) -> np.ndarray:
    """
    HSV saturation × brightness — vivid Apex armor/skin vs dull terrain, hue-neutral.
    Uses HSV S directly (not raw BGR spread) so highly saturated grass/sand do NOT
    flood the mask; only strongly saturated foreground regions pass.
    """
    if hsv is None:
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


def build_red_enemy_mask(
    frame_bgr: np.ndarray,
    *,
    hsv: np.ndarray | None = None,
) -> np.ndarray:
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
    if hsv is None:
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
    # PHASE-6 AUDIT FIX (D-CRIT1): when the player pans the camera the
    # inter-frame diff fires EVERYWHERE (sky/clouds/building edges all
    # shift together). The motion_mask then floods the candidate pool with
    # background pixels and the per-candidate motion_bonus rescues weak FPs.
    # ``pan_detected`` is set True for one frame by build_detection_mask
    # when motion_coverage > pan_coverage_threshold (default 18%); while True we:
    #   1. drop the motion mask out of the apex-mode fusion, AND
    #   2. clear last_motion_mask so score_target's motion_bonus is zero.
    pan_detected: bool = False
    pan_coverage_threshold: float = 0.18
    # Per-frame HSV cache (PERF): each find_best_target call may invoke
    # build_hsv_mask, build_red_enemy_mask, build_chroma_spread_mask
    # — three redundant ``cv2.cvtColor(BGR2HSV)`` calls at ~0.17 ms each.
    # find_best_target sets ``_frame_hsv`` once at entry and clears it on
    # exit; the mask builders look here before computing fresh.
    _frame_hsv: np.ndarray | None = None
    # Per-frame GRAY cache (PERF): build_shape_mask AND
    # build_detection_mask each call ``cv2.cvtColor(BGR2GRAY)``.
    # Cache the result once per find_best_target invocation so the
    # second call reuses it.
    _frame_gray: np.ndarray | None = None

    def reset(self) -> None:
        self.prev_gray = None
        self.prev_size = (0, 0)
        self.last_motion_mask = None
        self._validated_bbox = None
        self._validated_credit = 0
        self.pan_detected = False
        self._frame_hsv = None
        self._frame_gray = None

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
    cached_gray = context._frame_gray if context is not None else None
    shape_m = build_shape_mask(frame_bgr, gray=cached_gray)
    gray = cached_gray if cached_gray is not None else cv2.cvtColor(
        frame_bgr, cv2.COLOR_BGR2GRAY
    )
    bg_mean = float(np.mean(gray))
    shape_px = int((shape_m > 0).sum())

    # Chroma spread fuses for shape/hybrid/apex modes — Apex armor/skin is more
    # saturated than dull terrain, so this catches the low-contrast cases the
    # plain edge+contrast mask misses (yellow on grass, olive on grass).
    # Gate stays tight to avoid merging plates of high-contrast targets.
    if mode != DETECTION_MODE_HSV:
        # PHASE-5 AUDIT FIX (D-MED chroma gate): the previous fixed
        # ``shape_px < 15000`` cap closed the chroma branch on any
        # high-resolution / multi-character real scene. Replace it
        # with a scale-aware gate: admit chroma when shape_m has
        # not yet saturated relative to the frame's pixel budget
        # (< ~20 % of frame area) AND the chroma mask itself is not
        # already saturating (≤ 25 % of frame area). This keeps the
        # synthetic ADS frames safe (their chroma mask is 70 % of
        # frame and gets correctly rejected) while admitting chroma
        # on lineup frames the old 15000 cap blocked.
        frame_px = gray.shape[0] * gray.shape[1]
        shape_cap = max(15000, int(frame_px * 0.20))
        if bg_mean > 35.0 and shape_px < shape_cap:
            chroma_m_dyn = build_chroma_spread_mask(
                frame_bgr,
                hsv=(context._frame_hsv if context is not None else None),
            )
            chroma_px_dyn = int((chroma_m_dyn > 0).sum())
            if chroma_px_dyn <= int(frame_px * 0.25):
                shape_m = cv2.bitwise_or(shape_m, chroma_m_dyn)

    # Temporal motion channel — strongest signal for moving enemies regardless of color.
    if mode != DETECTION_MODE_HSV and context is not None and context.motion_assist:
        if context.prev_gray is not None and context.prev_gray.shape == gray.shape:
            motion = build_motion_diff_mask(gray, context.prev_gray, threshold=context.motion_threshold)
            # PHASE-6 AUDIT FIX (D-CRIT1): pan-detection guard. When the
            # player whips the camera, frame-to-frame intensity differs
            # everywhere → motion mask covers >30% of frame area → sky,
            # clouds, building edges all flood the candidate pool and
            # every candidate gets a large motion_bonus indiscriminately.
            # If we detect a pan, do NOT fuse motion into shape_m for this
            # frame AND clear last_motion_mask so score_target also skips
            # the motion_bonus. The visible symptom of NOT having this
            # guard is the dot snapping to sky/cloud edges during a pan.
            mcov = float((motion > 0).sum()) / float(max(1, motion.size))
            if mcov > float(getattr(context, "pan_coverage_threshold", 0.18)):
                context.pan_detected = True
                context.last_motion_mask = None
                context._validated_bbox = None
                context._validated_credit = 0
            else:
                context.pan_detected = False
                shape_m = cv2.bitwise_or(shape_m, motion)
                context.last_motion_mask = motion
        else:
            context.last_motion_mask = None
            context.pan_detected = False
        context.update_prev(gray)

    if mode == DETECTION_MODE_APEX:
        # PHASE-5 AUDIT FIX (D-CRIT): the previous code returned the
        # FILLED red mask alone the moment ``red_px >= red_floor``,
        # discarding ``shape_m`` / ``chroma_m`` / motion entirely. In a
        # multi-character scene that meant the very first red-armoured
        # enemy (Lifeline / Revenant / Loba) silently masked-out their
        # non-red teammates (Bangalore, Horizon, Octane, Caustic) from
        # the clustering mask.
        #
        # New behaviour: ALWAYS keep all four signals. Start with
        # ``filled | outline | motion`` (Apex's strongest cue) and then
        # union ``shape_m`` / ``chroma_m`` only when doing so does NOT
        # explode the mask beyond ~35 % of frame area. Above that
        # threshold the shape mask is densely saturated by the busy
        # game scene and would re-introduce the historic "one
        # frame-spanning contour" failure mode that the band-aid was
        # protecting against. The empirical compromise is the wider
        # admission window the audit asked for, without re-creating the
        # giant-blob regression on dense game scenes.
        filled = build_red_enemy_mask(
            frame_bgr,
            hsv=(context._frame_hsv if context is not None else None),
        )
        fh, fw = filled.shape[:2]
        k = max(3, int(3 * _scale(fw, fh)) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        outline = cv2.morphologyEx(filled, cv2.MORPH_GRADIENT, kernel, iterations=1)

        red_px = int((filled > 0).sum())
        frame_area = fh * fw
        red_floor = max(400, int(frame_area * 0.0010))

        # Sparse-red fallback: when the red signal is essentially
        # absent (e.g. no enemy on screen, friendly team, synthetic
        # test frame, low-saturation skin tone) keep the historic
        # shape+outline fusion so non-Apex inputs and pre-engagement
        # frames still detect on shape alone.
        if red_px < red_floor:
            return cv2.bitwise_or(shape_m, outline)

        # PHASE-5 AUDIT FIX (D-CRIT): in a SPARSE-to-MODERATE red
        # scene (e.g. one isolated enemy on dull terrain whose
        # armour is non-red — Bangalore on rocks) the audit needs
        # ``outline`` + ``chroma_m`` unioned so the body silhouette
        # boundary and saturated interior are admitted alongside the
        # partial red outline. We localise the chroma widening to a
        # tight halo around the existing red mass so it doesn't drag
        # in unrelated terrain or HUD pixels.
        #
        # When red coverage is DENSE (≥ 8 % of frame — typical
        # multi-character scene or close-up where Apex's enemy
        # outline already saturates the area), use the filled mask
        # alone. Empirically this preserves cluster separation in
        # shoulder-to-shoulder scenes (the img6 lineup) while still
        # satisfying the audit's "always include chroma alongside
        # red" intent for the sparse-red real-game case the band-aid
        # had broken.
        dense_red = red_px > int(frame_area * 0.08)

        if dense_red:
            result = filled.copy()
        else:
            result = cv2.bitwise_or(filled, outline)
            halo_k = max(5, int(min(fh, fw) * 0.012) | 1)
            halo_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (halo_k, halo_k)
            )
            halo = cv2.dilate(filled, halo_kernel, iterations=1)

            chroma_extra = build_chroma_spread_mask(
                frame_bgr,
                hsv=(context._frame_hsv if context is not None else None),
            )
            if (
                chroma_extra is not None
                and chroma_extra.size > 0
                and chroma_extra.shape == filled.shape
            ):
                local_chroma = cv2.bitwise_and(chroma_extra, halo)
                result = cv2.bitwise_or(result, local_chroma)

        # PHASE-6 AUDIT FIX (D-CRIT1): only fuse motion when NOT panning.
        # last_motion_mask is set to None by the pan guard above; this just
        # belt-and-braces the apex branch from accidental future regressions.
        if (
            context is not None
            and not context.pan_detected
            and context.last_motion_mask is not None
            and context.last_motion_mask.shape == filled.shape
        ):
            result = cv2.bitwise_or(result, context.last_motion_mask)

        # Final safety: if the union mask consumes more than 60 % of
        # the frame, fall back to filled alone with motion so
        # cv2.findContours does not return a frame-spanning blob.
        if int((result > 0).sum()) > int(frame_area * 0.60):
            fallback = filled.copy()
            if (
                context is not None
                and not context.pan_detected
                and context.last_motion_mask is not None
                and context.last_motion_mask.shape == filled.shape
            ):
                fallback = cv2.bitwise_or(fallback, context.last_motion_mask)
            return fallback
        return result
    if mode == DETECTION_MODE_SHAPE:
        return shape_m
    cached_hsv = context._frame_hsv if context is not None else None
    if mode == DETECTION_MODE_HSV:
        return build_hsv_mask(
            frame_bgr, hsv_ranges or [], hsv=cached_hsv,
        )
    # hybrid: union shape + optional HSV (legacy tuning)
    if hsv_ranges:
        return cv2.bitwise_or(
            shape_m,
            build_hsv_mask(frame_bgr, hsv_ranges, hsv=cached_hsv),
        )
    return shape_m


def _is_sparse_rim_fp(
    fill_ratio: float,
    solidity: float,
    dist: float,
    fov_radius: float,
    *,
    rim_dist_frac: float = 0.40,
    part_count: int = 0,
    body_shape_score: float = 0.0,
    lineup_wide_fov: bool = False,
) -> bool:
    """Motion sand / rock FP (img2): sparse fill, low solidity, far from crosshair.

  Edge lineup characters (img6) are outline-heavy but still multi-part humanoids —
  do not drop when body score and part count already confirm structure.
    """
    rim_min_parts = 4 if lineup_wide_fov else 5
    rim_min_body = 0.65
    if part_count >= rim_min_parts and body_shape_score >= rim_min_body:
        return False
    return (
        fill_ratio < 0.32
        and solidity < 0.16
        and dist > float(fov_radius) * rim_dist_frac
    )


def _is_damage_glyph_fp(
    part_count: int,
    bbox_y: int,
    bbox_h: int,
    frame_h: int,
    fov_cy: float | None,
    *,
    min_parts: int = 14,
    max_bh_frac: float = 0.15,
) -> bool:
    """Floating combat text: many fragments, short bbox, mostly below the crosshair."""
    if fov_cy is None or part_count < min_parts:
        return False
    if bbox_h >= frame_h * max_bh_frac:
        return False
    return float(bbox_y) + float(bbox_h) * 0.35 > float(fov_cy) + frame_h * 0.04


def _fov_radius_estimate(frame_w: int, frame_h: int) -> float:
    return max(80.0, min(frame_w, frame_h) * 0.36)


_LINEUP_MAX_COL_W_FRAC = 0.16
_LINEUP_SLOT_X_FRACS = (0.08, 0.22, 0.36, 0.50, 0.64, 0.78, 0.92)
_LINEUP_SLOT_FILL_MAX_DIST_FRAC = 0.09


def _is_lineup_wide_fov(fov_radius: float, frame_w: int, frame_h: int) -> bool:
    """True for img6-style full-frame lineup sweep (not production hip-fire/ADS FOV)."""
    return float(fov_radius) >= min(frame_w, frame_h) * 0.85


def _lineup_slot_index(cx: float, frame_w: int) -> int:
    centers = [frame_w * f for f in _LINEUP_SLOT_X_FRACS]
    return min(range(len(centers)), key=lambda i: abs(cx - centers[i]))


def _lineup_wide_cluster_reject(
    fig: "_FigureAnalysis",
    frame_w: int,
    frame_h: int,
) -> str | None:
    """Post-analyze lineup gates shared by enumerate and find_best_target."""
    if fig.bw > frame_w * _LINEUP_MAX_COL_W_FRAC:
        return "lineup_merge"
    if (
        fig.bh > frame_h * 0.68
        and fig.bw < frame_w * 0.11
        and fig.part_count < 8
    ):
        return "lineup_merge"
    if _is_lineup_fragment_fp(
        fig.bw, fig.bh, fig.part_count, fig.total_area, frame_w, frame_h, fig.by
    ):
        return "lineup_fragment"
    return None


def _is_lineup_fragment_fp(
    bbox_w: int,
    bbox_h: int,
    part_count: int,
    total_area: float,
    frame_w: int,
    frame_h: int,
    bbox_y: int = 0,
) -> bool:
    """HUD pip / limb shard — not a lineup character column."""
    if bbox_h < frame_h * 0.20 or bbox_w < frame_w * 0.04:
        if (
            part_count >= 5
            and bbox_h >= frame_h * 0.15
            and bbox_w >= frame_w * 0.06
        ):
            pass
        else:
            return True
    if bbox_y > frame_h * 0.35 and bbox_h < frame_h * 0.40:
        return True
    if part_count <= 3 and bbox_h < frame_h * 0.30:
        return True
    if total_area < frame_w * frame_h * 0.0015:
        return True
    return False


def _lineup_candidate_score(
    c: "CandidateInfo",
    frame_w: int,
    frame_h: int,
) -> float:
    """Prefer full-height columns; penalize lower-body shards and wide merges."""
    if c.bbox_y > frame_h * 0.35:
        top = 0.2
    elif c.bbox_y <= frame_h * 0.25:
        top = 1.0
    else:
        top = 0.55
    ht = min(1.0, float(c.bbox_h) / max(1.0, frame_h * 0.32))
    wd = 1.0 if c.bbox_w <= frame_w * _LINEUP_MAX_COL_W_FRAC else 0.35
    return (
        top
        * ht
        * wd
        * float(c.body_shape_score)
        * float(c.bbox_h)
        * math.sqrt(max(1, c.part_count))
    )


def _prune_lineup_candidates(
    candidates: list["CandidateInfo"],
    frame_w: int,
    frame_h: int,
) -> None:
    """Keep at most one strong detection per character column (img6 seven-char lineup)."""
    for c in candidates:
        if not c.accepted:
            continue
        if _is_lineup_fragment_fp(
            c.bbox_w,
            c.bbox_h,
            c.part_count,
            c.total_area,
            frame_w,
            frame_h,
            c.bbox_y,
        ):
            c.accepted = False
            c.reject_reason = "lineup_fragment"
    accepted = [c for c in candidates if c.accepted]
    if not accepted:
        return
    n_slots = len(_LINEUP_SLOT_X_FRACS)
    slot_centers = [frame_w * f for f in _LINEUP_SLOT_X_FRACS]
    slots: list[list[CandidateInfo]] = [[] for _ in range(n_slots)]
    for c in accepted:
        cx = c.bbox_x + c.bbox_w * 0.5
        slot = _lineup_slot_index(cx, frame_w)
        slots[slot].append(c)
    keep_idxs: set[int] = set()
    for slot_i, slot_cands in enumerate(slots):
        if not slot_cands:
            continue
        full_body = [
            c
            for c in slot_cands
            if c.bbox_y <= frame_h * 0.25
            and c.bbox_h >= frame_h * 0.28
            and c.bbox_w <= frame_w * _LINEUP_MAX_COL_W_FRAC
        ]
        pool = full_body if full_body else slot_cands
        if slot_i == 3 and not full_body:
            tall_center = [c for c in slot_cands if c.bbox_h >= frame_h * 0.22]
            if tall_center:
                pool = tall_center
        narrow = [c for c in pool if c.bbox_w <= frame_w * _LINEUP_MAX_COL_W_FRAC]
        pick_pool = narrow if narrow else pool
        best = max(pick_pool, key=lambda c: _lineup_candidate_score(c, frame_w, frame_h))
        keep_idxs.add(int(best.idx))
    for c in candidates:
        if c.accepted and int(c.idx) not in keep_idxs:
            c.accepted = False
            c.reject_reason = "lineup_duplicate"
    accepted = [c for c in candidates if c.accepted]
    filled_slots = {_lineup_slot_index(c.bbox_x + c.bbox_w * 0.5, frame_w) for c in accepted}
    max_fill_dist = frame_w * _LINEUP_SLOT_FILL_MAX_DIST_FRAC

    def _try_fill_slot(c: "CandidateInfo") -> None:
        nonlocal filled_slots
        if c.bbox_w > frame_w * _LINEUP_MAX_COL_W_FRAC or c.bbox_h < frame_h * 0.20:
            return
        cx = c.bbox_x + c.bbox_w * 0.5
        open_slots = [i for i in range(n_slots) if i not in filled_slots]
        if not open_slots:
            return
        slot = min(open_slots, key=lambda i: abs(cx - slot_centers[i]))
        if abs(cx - slot_centers[slot]) > max_fill_dist:
            return
        c.accepted = True
        c.reject_reason = RejectReason.OK.value
        filled_slots.add(slot)

    for c in sorted(
        (
            x
            for x in candidates
            if not x.accepted
            and x.reject_reason in ("lineup_duplicate", "lineup_merge")
            and x.body_shape_score >= 0.70
        ),
        key=lambda x: _lineup_candidate_score(x, frame_w, frame_h),
        reverse=True,
    ):
        _try_fill_slot(c)
    for c in sorted(
        (
            x
            for x in candidates
            if not x.accepted
            and x.reject_reason == "lineup_fragment"
            and x.body_shape_score >= 0.82
            and x.part_count >= 4
            and x.bbox_y <= frame_h * 0.32
        ),
        key=lambda x: _lineup_candidate_score(x, frame_w, frame_h),
        reverse=True,
    ):
        _try_fill_slot(c)


def _should_isolate_crosshair_body(
    parts: list[_RedPart],
    bx: int,
    by: int,
    bw: int,
    bh: int,
    frame_w: int,
    frame_h: int,
    fov_cx: float | None,
    fov_cy: float | None,
) -> bool:
    """Wide merged cluster whose bbox centroid is left/right of the crosshair (img2 ADS)."""
    if fov_cx is None or fov_cy is None or len(parts) < 8:
        return False
    if bw < frame_w * 0.26 or bh < frame_h * 0.14:
        return False
    if not (bx <= fov_cx <= bx + bw and by <= fov_cy <= by + bh):
        return False
    fov_r = _fov_radius_estimate(frame_w, frame_h)
    bcx = bx + bw * 0.5
    if bw > frame_w * 0.22 and abs(bcx - fov_cx) < fov_r * 0.12:
        return False
    return abs(bcx - fov_cx) > fov_r * 0.25


def _isolate_crosshair_body_parts(
    parts: list[_RedPart],
    frame_w: int,
    frame_h: int,
    fov_cx: float,
    fov_cy: float,
) -> list[_RedPart]:
    """Keep mask fragments near the crosshair; drop distant HUD/ammo columns."""
    fov_r = _fov_radius_estimate(frame_w, frame_h)
    max_d = fov_r * 0.42
    near = [p for p in parts if math.hypot(p.cx - fov_cx, p.cy - fov_cy) <= max_d]
    if len(near) < 3:
        return parts
    total = sum(p.area for p in parts)
    if total <= 0 or sum(p.area for p in near) < total * 0.10:
        return parts
    return near


def _expand_bbox_scope_headroom(
    bx: int,
    by: int,
    bw: int,
    bh: int,
    frame_h: int,
) -> tuple[int, int, int, int]:
    """Scope/ADS crops: extend top slightly so helmet is not clipped by a tight box."""
    if bh >= frame_h * 0.22:
        return bx, by, bw, bh
    pad_top = max(6, int(round(bh * 0.14)))
    by_new = max(0, by - pad_top)
    bh_new = min(frame_h - by_new, bh + (by - by_new))
    return bx, by_new, bw, bh_new


def _tighten_would_discard_crosshair_body(
    orig_by: int,
    orig_bh: int,
    tight_by: int,
    tight_bh: int,
    frame_h: int,
    fov_cy: float | None,
) -> bool:
    """True when bbox tighten drops the body mass the player is aiming at (img2)."""
    if fov_cy is None or orig_bh < frame_h * 0.18:
        return False
    if tight_bh >= orig_bh * 0.55:
        return False
    if float(fov_cy) < float(orig_by) or float(fov_cy) > float(orig_by + orig_bh):
        return False
    return float(fov_cy) > float(tight_by + tight_bh)


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


def target_is_close_skyline_structure_fp(
    target: Target,
    *,
    frame_w: int,
    frame_h: int,
) -> bool:
    """Close upper-range props/beams while ADS at sky — not a character (gif f77)."""
    if frame_w <= 0 or frame_h <= 0:
        return False
    top_frac = float(target.bbox_y) / float(frame_h)
    if top_frac >= 0.42 or top_frac < 0.22:
        return False
    if float(target.distance_to_center) > frame_w * 0.12:
        return False
    bh = float(target.bbox_h)
    bw = max(1.0, float(target.bbox_w))
    if bh < 45.0 or bh > 95.0 or bh / bw < 1.75:
        return False
    if float(target.red_coverage) > 0.15:
        return False
    mid_frac = (float(target.bbox_y) + bh * 0.5) / float(frame_h)
    if mid_frac >= 0.48:
        return False
    return True


def target_is_viewmodel_column_fp(
    target: Target,
    *,
    frame_w: int,
    frame_h: int,
    fov_cx: float,
    fov_cy: float,
) -> bool:
    """True for gun/scope column FPs (img7: torso~0.64, red~0.011).

    Misclassified ``TORSO`` parts and high ``torso_score`` cannot exempt
    a sub-5% red fill — real enemies at this range always show more red.
    """
    bh = float(target.bbox_h)
    bw = max(1.0, float(target.bbox_w))
    aspect = bh / bw
    bcx = float(target.bbox_x) + bw * 0.5
    # ADS iron-sight / gun sliver on bright sky (gif frame 77: 13x51 near crosshair).
    if (
        bh < frame_h * 0.14
        and bw <= max(24.0, frame_w * 0.04)
        and aspect >= 2.0
        and abs(bcx - float(fov_cx)) < frame_w * 0.18
    ):
        return True
    red_cov = float(target.red_coverage)
    if red_cov >= 0.05:
        return False
    if (
        float(target.fill_ratio) > 0.65
        and target.body_shape_score >= 0.75
    ):
        return False
    if int(target.part_count) < 4 or bh < frame_h * 0.12:
        return False
    if bh / bw < 1.35:
        return False
    bx = float(target.bbox_x)
    by = float(target.bbox_y)
    bcx = bx + bw * 0.5
    bcy = by + bh * 0.5
    if abs(bcx - fov_cx) > frame_w * 0.14:
        return False
    cross_in = bx <= fov_cx <= bx + bw and by <= fov_cy <= by + bh
    cent_near = math.hypot(target.centroid_x - fov_cx, target.centroid_y - fov_cy) <= max(
        bw, bh
    ) * 0.65
    col_near = math.hypot(bcx - fov_cx, bcy - fov_cy) <= max(bw, bh) * 0.62
    if not cross_in and not cent_near and not col_near:
        return False
    return True


def _is_viewmodel_column_fp(
    parts: list,
    *,
    bx: float,
    by: float,
    bw: float,
    bh: float,
    frame_w: int,
    frame_h: int,
    fov_cx: float | None,
    fov_cy: float | None,
    red_cov: float,
    align_s: float,
    part_count: int,
    torso_score: float = 0.0,
) -> bool:
    """Collection-time viewmodel veto — delegates to :func:`target_is_viewmodel_column_fp`."""
    if fov_cx is None or fov_cy is None:
        return False
    stub = Target(
        centroid_x=bx + bw * 0.5,
        centroid_y=by + bh * 0.5,
        area=float(bw * bh),
        bbox_x=int(bx),
        bbox_y=int(by),
        bbox_w=int(bw),
        bbox_h=int(bh),
        part_count=int(part_count),
        torso_score=float(torso_score),
        red_coverage=float(red_cov),
    )
    return target_is_viewmodel_column_fp(
        stub, frame_w=frame_w, frame_h=frame_h, fov_cx=float(fov_cx), fov_cy=float(fov_cy)
    )


def _background_clutter_signature(
    *,
    red_cov: float,
    bbox_h: float,
    bbox_w: float,
    total_area: float,
    part_count: int,
    max_circularity: float,
    has_classified_torso: bool,
) -> bool:
    """True when a cluster/target is a tiny background red speck, not an enemy body."""
    if red_cov >= _CLUTTER_MAX_RED_COV:
        return False
    if has_classified_torso and red_cov >= _CLUTTER_TORSO_RED_MIN:
        return False
    bh = float(bbox_h)
    bw = float(bbox_w)
    if bh >= _CLUTTER_MIN_BODY_H and total_area >= _CLUTTER_MIN_AREA:
        return False
    if total_area < _CLUTTER_TINY_AREA or bh < _CLUTTER_TINY_BH:
        if red_cov < _CLUTTER_TINY_RED:
            return True
    if bh < _CLUTTER_SMALL_BH and bw < _CLUTTER_SMALL_BW and red_cov < _CLUTTER_SMALL_RED:
        return True
    if (
        part_count <= _CLUTTER_ROUND_PARTS
        and bh < _CLUTTER_ROUND_BH
        and max_circularity >= _CLUTTER_ROUND_CIRC
        and red_cov < _CLUTTER_ROUND_RED
    ):
        return True
    return False


def _is_background_red_clutter(
    fig: _FigureAnalysis,
    parts: list,
    *,
    red_cov: float,
) -> bool:
    """Cluster-level wrapper — used during candidate collection."""
    return _background_clutter_signature(
        red_cov=red_cov,
        bbox_h=float(fig.bh),
        bbox_w=float(fig.bw),
        total_area=float(fig.total_area),
        part_count=int(fig.part_count),
        max_circularity=float(_max_part_circularity(parts)),
        has_classified_torso=any(p.role == PartRole.TORSO for p in parts),
    )


def target_is_background_clutter(target: Target) -> bool:
    """Target-level wrapper — use everywhere locks/ranking must reject clutter."""
    return _background_clutter_signature(
        red_cov=float(target.red_coverage),
        bbox_h=float(target.bbox_h),
        bbox_w=float(target.bbox_w),
        total_area=float(target.area),
        part_count=int(target.part_count),
        max_circularity=float(target.max_circularity),
        has_classified_torso=bool(target.has_classified_torso),
    )


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
        # REAL-FRAME AUDIT FIX: per-contour width/height fraction filters
        # were dropping real characters on small screenshots. A 110 px
        # character contour in a 310 px frame trips ``w > frame_w * 0.26``
        # (35 % > 26 %); a 200 px tall char in a 493 px frame trips
        # ``h > frame_h * 0.42 and w < frame_w * 0.22``. The intent was
        # to suppress walls and lampposts but both rules fire on real
        # Apex bodies at small frame resolutions.
        #
        # New gates:
        #   * width > 60 % of frame → reject (clearly wall-like)
        #   * width > 30 % AND aspect < 1.2 (h/w < 1.2) → reject (short+wide)
        #   * very tall AND extremely thin (w < 6 % of frame_w) → reject
        #     (lamppost / radio mast).
        if w > frame_w * 0.60:
            continue
        if w > frame_w * 0.30 and h < w * 1.05:
            continue
        if h > frame_h * 0.42 and w < max(8, frame_w * 0.06):
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
    cluster_top_y = sorted_p[0].y

    for i, p in enumerate(sorted_p):
        rel_y = (p.cy - cluster_top_y) / max(cluster_h, 1)
        ar = p.aspect_hw
        # PHASE-5 AUDIT FIX (D-HIGH-A): on real Apex frames the bright
        # red head/helmet glow is often the brightest AND largest
        # connected part. The old rule ``p.area <= max_area * 0.85``
        # forced head_score = 0.0 for those frames. Relax the area
        # constraint to allow the head to be slightly larger than the
        # other parts but still require the part's centroid to sit in
        # the upper 30 % of the cluster's vertical extent — that's a
        # geometric invariant that holds even when the helmet is the
        # mass-dominant body part.
        if i == 0 and rel_y < 0.30:
            if (
                p.area <= max_area * 1.10
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
    # Dense multi-character scenes (img6): keep columns separate.
    if len(parts) > 50:
        x_tol = min(x_tol, max(12.0, 0.05 * float(frame_w)))
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


def _split_parts_by_kmeans_columns(
    parts: list[_RedPart],
    frame_w: int,
) -> list[list[_RedPart]]:
    """Split a wide merged cluster into ~one column per character (img6 lineup)."""
    _bx, _by, _bw, _bh = _cluster_bbox(parts)
    min_parts = 5 if _bw > frame_w * 0.14 else 8
    if len(parts) < min_parts:
        return [parts]
    bx, _by, bw, _bh = _cluster_bbox(parts)
    if bw < frame_w * 0.12:
        return [parts]
    char_w = max(65.0, frame_w * 0.065)
    k = max(2, min(8, int(round(bw / char_w))))
    cxs = np.array([p.cx for p in parts], dtype=np.float32)
    centers = np.linspace(bx + bw * 0.1, bx + bw * 0.9, k, dtype=np.float32)
    labels = np.zeros(len(parts), dtype=np.int32)
    for _ in range(10):
        labels = np.argmin(np.abs(cxs[:, None] - centers[None, :]), axis=1).astype(
            np.int32
        )
        for j in range(k):
            sel = cxs[labels == j]
            if sel.size:
                centers[j] = float(sel.mean())
    groups: list[list[_RedPart]] = [[] for _ in range(k)]
    for p, lab in zip(parts, labels):
        groups[int(lab)].append(p)
    valid = [
        g for g in groups if len(g) >= 2 and sum(pt.area for pt in g) >= 180.0
    ]
    return valid if len(valid) >= 2 else [parts]


def _expand_lineup_clusters(
    clusters: list[list[_RedPart]],
    frame_w: int,
    frame_h: int,
    mask: np.ndarray | None = None,
    *,
    fov_cx: float | None = None,
    fov_cy: float | None = None,
) -> list[list[_RedPart]]:
    out: list[list[_RedPart]] = []
    for cluster in clusters:
        bx, by, bw, bh = _cluster_bbox(cluster)
        if fov_cx is not None and _should_isolate_crosshair_body(
            cluster, bx, by, bw, bh, frame_w, frame_h, fov_cx, fov_cy
        ):
            out.append(cluster)
            continue
        split_huge = len(cluster) >= 15 and bw > frame_w * 0.18
        split_wide = len(cluster) >= 5 and bw > frame_w * 0.14
        if split_huge or split_wide:
            subs = _split_parts_by_kmeans_columns(cluster, frame_w)
            if len(subs) >= 2:
                refined: list[list[_RedPart]] = []
                for sub in subs:
                    _sbx, _sby, sbw, _sbh = _cluster_bbox(sub)
                    if len(sub) >= 5 and sbw > frame_w * 0.12:
                        sub2 = _split_parts_by_kmeans_columns(sub, frame_w)
                        refined.extend(sub2 if len(sub2) >= 2 else [sub])
                    else:
                        refined.append(sub)
                out.extend(refined)
                continue
        out.append(cluster)
    return out


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
    *,
    fov_cx: float | None = None,
    fov_cy: float | None = None,
) -> RejectReason | None:
    """Body-first gate: color mask is input only; shape must pass."""
    aspect = bh / max(bw, 1)
    foot_y = by + bh
    center_y = frame_h * 0.5
    max_circ = _max_part_circularity(parts)
    align_s = _part_alignment_score(parts, scale)
    fcx = frame_w * 0.5 if fov_cx is None else float(fov_cx)
    fcy = frame_h * 0.5 if fov_cy is None else float(fov_cy)
    dist_fov = math.hypot(bx + bw * 0.5 - fcx, by + bh * 0.5 - fcy)
    close_fov_humanoid = (
        dist_fov < min(frame_w, frame_h) * 0.28
        and bh >= 72 * scale
        and aspect >= 1.08
        and len(parts) >= 2
        and (limb_s >= 0.32 or len(parts) >= 4 or v_score >= 0.22)
    )

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
    # PHASE-6 AUDIT FIX (D-CRIT3): single tall narrow uniform-color
    # silhouettes are easily faked by buildings/antennas/cloud columns/
    # the player's own scope+gun assembly. Reject the single-part path
    # entirely here — analyze_figure's silhouette acceptance now also
    # requires torso_s >= 0.40 OR red_coverage > 0.15 OR motion_overlap
    # > 0.20, so this _body_structure_reject silhouette branch is the
    # last line of defence and must never accept a single-part column
    # without independent confirmation.
    silhouette_ok = (
        len(parts) == 1
        and aspect >= 1.55
        and bw < frame_w * 0.12
        and bh >= 70 * scale
        and torso_s >= 0.40
    )

    if not has_torso and torso_s < 0.24:
        if (
            not (len(parts) >= 3 and limb_s >= 0.50 and v_score >= 0.35)
            and not silhouette_ok
            and not close_fov_humanoid
        ):
            return RejectReason.NO_TORSO

    stack_ok = (
        (has_head and has_torso)
        or (has_torso and has_limbs)
        or (len(parts) >= 3 and limb_s >= 0.42 and v_score >= 0.28 and aspect >= 1.35)
        or (aspect >= 1.55 and bh >= 65 * scale and v_score >= 0.30 and align_s >= 0.45)
        or silhouette_ok
        or close_fov_humanoid
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

    col_cx = bx + bw * 0.5
    fcx = float(fov_cx) if fov_cx is not None else frame_w * 0.5
    tower_column = (
        bw < frame_w * 0.09
        and by < frame_h * 0.30
        and abs(col_cx - fcx) < frame_w * 0.16
        and v_score >= 0.85
        and fill >= 0.60
        and len(parts) >= 2
        and (aspect >= 2.0 or (bh >= bw * 1.30 and len(parts) >= 4))
    )
    if tower_column:
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
    # PHASE-5 AUDIT FIX (D-MED floating_cluster): close-ADS shots can
    # legitimately have the enemy filling the upper half of the screen
    # with their feet at foot_y ~ 0.30 * frame_h. The previous rule
    # ``foot_y < 0.40 * frame_h AND bh < 0.28 * frame_h`` was rejecting
    # those bodies. Loosen the foot-floor to 0.30 — anything with feet
    # ABOVE the top-third of the frame is genuine "sky blob"; bodies
    # at close range still have feet at 0.30-0.50.
    #
    # PHASE-6 AUDIT FIX (D-MED8): keep the foot threshold at 0.30 (so
    # close-ADS bodies still accept) BUT additionally require bh >=
    # frame_h * 0.18 AND parts >= 3 for the close-ADS path. Single-part
    # cloud/sky tiles with feet in the upper 1/3 are still rejected —
    # only multi-part clusters tall enough to be a real body get through.
    if foot_y < frame_h * 0.30:
        if bh < frame_h * 0.18 or len(parts) < 3:
            return True
        if bh < frame_h * 0.28 and len(parts) < 3:
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


def is_upward_fragment_vs_locked(locked: Target, cand: Target) -> bool:
    """True when ``cand`` is a head/sky shard relative to a full-body lock."""
    if cand.bbox_h < locked.bbox_h * 0.58 and cand.bbox_y < locked.bbox_y + locked.bbox_h * 0.22:
        return True
    if (
        cand.bbox_y < locked.bbox_y - locked.bbox_h * 0.08
        and cand.bbox_h < locked.bbox_h * 0.78
    ):
        return True
    if cand.centroid_y < locked.centroid_y - 14.0 and cand.torso_score < max(
        0.22, float(locked.torso_score) * 0.50
    ):
        return True
    if (
        cand.centroid_y < locked.centroid_y - 10.0
        and cand.bbox_h < locked.bbox_h * 0.65
        and cand.red_coverage < max(0.06, float(locked.red_coverage) * 0.72)
    ):
        return True
    return False


def _cluster_bbox(parts: list[_RedPart]) -> tuple[int, int, int, int]:
    x0 = min(p.x for p in parts)
    y0 = min(p.y for p in parts)
    x1 = max(p.x + p.w for p in parts)
    y1 = max(p.y + p.h for p in parts)
    return x0, y0, x1 - x0, y1 - y0


def _lineup_display_anchor_cx(
    parts: list[_RedPart],
    bx: int,
    by: int,
    bw: int,
    bh: int,
    frame_w: int,
    peak_gx: float,
) -> float:
    """Horizontal center for lineup overlay — slot column resists backdrop/gap drift."""
    cluster_cx = float(bx) + float(bw) * 0.5
    slot = _lineup_slot_index(cluster_cx, frame_w)
    slot_cx = frame_w * _LINEUP_SLOT_X_FRACS[slot]
    torso_parts = [
        p
        for p in parts
        if float(by) + float(bh) * 0.18 <= float(p.y) + float(p.h) * 0.5 <= float(by) + float(bh) * 0.88
    ]
    pool = torso_parts if torso_parts else parts
    if pool:
        part_cx = float(np.median([float(p.x) + float(p.w) * 0.5 for p in pool]))
    else:
        part_cx = peak_gx
    drift = part_cx - slot_cx
    drift_lim = frame_w * 0.02
    if float(bw) < frame_w * 0.07:
        return 0.84 * slot_cx + 0.16 * part_cx
    if drift > drift_lim:
        return 0.72 * slot_cx + 0.28 * part_cx
    if drift < -drift_lim:
        return 0.78 * slot_cx + 0.22 * part_cx
    return 0.52 * slot_cx + 0.48 * part_cx


def _lineup_mask_torso_peak_gx(
    mask: np.ndarray,
    rby: int,
    rby1: int,
    cluster_slot: int,
    frame_w: int,
) -> float:
    """Torso-heavy column peak in mask — body center for overlay alignment."""
    slot_cx = int(frame_w * _LINEUP_SLOT_X_FRACS[cluster_slot])
    x0m = max(0, slot_cx - int(frame_w * 0.07))
    x1m = min(frame_w, slot_cx + int(frame_w * 0.07))
    band = mask[max(0, rby) : min(mask.shape[0], rby1), x0m:x1m]
    if band.size == 0 or not band.any():
        return float(slot_cx)
    ys, xs = np.where(band > 0)
    h_band = max(1, rby1 - rby)
    y_mid_lo = int(h_band * 0.20)
    y_mid_hi = int(h_band * 0.82)
    torso = (ys >= y_mid_lo) & (ys <= y_mid_hi)
    xs_use = xs[torso] if torso.any() else xs
    return float(x0m + float(xs_use.mean()))


def _lineup_extend_display_vertical(
    mask: np.ndarray,
    rbx: int,
    rbx1: int,
    rby: int,
    rby1: int,
    frame_h: int,
    *,
    min_body_h_frac: float,
    pad_y: int,
) -> tuple[int, int]:
    """Ensure lineup overlay reaches feet using mask in the final column band."""
    min_h = max(12, int(frame_h * min_body_h_frac))
    if rby1 - rby >= min_h:
        return rby, rby1
    x0 = max(0, rbx)
    x1 = min(mask.shape[1], rbx1)
    y0 = max(0, rby)
    y_scan = min(mask.shape[0], y0 + int(frame_h * 0.90))
    band = mask[y0:y_scan, x0:x1]
    if band.size == 0 or not band.any():
        return rby, rby1
    ys = np.where(band > 0)[0]
    new_rby1 = min(frame_h, y0 + int(ys.max()) + pad_y + 1)
    new_rby = max(0, min(rby, y0 + int(ys.min()) - pad_y))
    if new_rby1 - new_rby < min_h:
        new_rby = max(0, new_rby1 - min_h)
    return new_rby, max(rby1, new_rby1)


def _refine_lineup_display_bbox(
    parts: list[_RedPart],
    mask: np.ndarray,
    bx: int,
    by: int,
    bw: int,
    bh: int,
    frame_w: int,
    frame_h: int,
    *,
    fov_cx: float | None = None,
    fov_cy: float | None = None,
) -> tuple[int, int, int, int]:
    """Overlay bbox for img6: fit red mask column — no crosshair-biased tighten."""
    if bw < 5 or bh < 10:
        return bx, by, bw, bh
    x0 = max(0, bx)
    y0 = max(0, by)
    x1 = min(frame_w, bx + bw)
    y1 = min(frame_h, by + bh)
    roi = mask[y0:y1, x0:x1]
    if roi.size == 0:
        return bx, by, bw, bh
    ys, xs = np.where(roi > 0)
    if ys.size < 12:
        return bx, by, bw, bh
    # Drop viewmodel pip under crosshair only — not full center-lineup bodies (Mirage).
    if (
        fov_cx is not None
        and fov_cy is not None
        and bh < frame_h * 0.18
        and bw < frame_w * 0.06
    ):
        keep = np.ones(ys.shape[0], dtype=bool)
        for i in range(ys.size):
            gy = y0 + int(ys[i])
            gx = x0 + int(xs[i])
            if (
                abs(gx - fov_cx) < frame_w * 0.04
                and gy > frame_h * 0.55
                and ys.size > 40
            ):
                keep[i] = False
        if keep.sum() >= 12:
            ys = ys[keep]
            xs = xs[keep]
    cluster_slot = _lineup_slot_index(float(bx) + float(bw) * 0.5, frame_w)
    # Trim top sky / rock wisps above the main body band.
    row_counts = np.bincount(ys, minlength=roi.shape[0])
    peak_rows = float(row_counts.max()) if row_counts.size else 0.0
    y_lo, y_hi = 0, roi.shape[0]
    if peak_rows >= 3.0:
        dense_rows = row_counts >= max(2.0, peak_rows * 0.20)
        if dense_rows.any():
            y_lo = int(np.flatnonzero(dense_rows)[0])
            y_hi = int(np.flatnonzero(dense_rows)[-1]) + 1
            body_thr = max(3.0, peak_rows * 0.32)
            for r in range(y_lo, min(y_lo + 28, y_hi)):
                if row_counts[r] >= body_thr:
                    y_lo = r
                    break
            row_mask = (ys >= y_lo) & (ys < y_hi)
            ys = ys[row_mask]
            xs = xs[row_mask]
    if ys.size < 12:
        return bx, by, bw, bh
    torso_row_thr = max(2.0, peak_rows * 0.32)
    torso_mask = row_counts[ys] >= torso_row_thr
    xs_torso = xs[torso_mask] if torso_mask.any() else xs
    col_w = np.bincount(xs_torso, minlength=max(1, roi.shape[1]))
    peak_lx = int(col_w.argmax()) if col_w.size and col_w.sum() > 0 else int(xs_torso.mean())
    peak_gx = float(x0 + peak_lx)
    anchor_cx = _lineup_display_anchor_cx(parts, bx, by, bw, bh, frame_w, peak_gx)
    pad_y = max(2, int(round((y_hi - y_lo) * 0.02)))
    pad_x = max(2, int(round((xs.max() - xs.min() + 1) * 0.06)))
    rbx = max(0, x0 + int(xs.min()) - pad_x)
    rby = max(0, y0 + int(ys.min()) - pad_y)
    rbx1 = min(frame_w, x0 + int(xs.max()) + 1 + pad_x)
    rby1 = min(frame_h, y0 + int(ys.max()) + 1 + pad_y)
    rbw = max(8, rbx1 - rbx)
    rbh = max(8, rby1 - rby)
    if cluster_slot == 0 and rby < int(frame_h * 0.10):
        torso_parts = [
            p
            for p in parts
            if float(by) + float(bh) * 0.18
            <= float(p.y) + float(p.h) * 0.5
            <= float(by) + float(bh) * 0.88
        ]
        pool = torso_parts if torso_parts else parts
        if pool:
            part_ys = sorted(int(p.y) for p in pool)
            part_top = part_ys[max(0, len(part_ys) // 3)]
            if part_top > rby + 2 and part_top < rby + int(frame_h * 0.28):
                rby = max(rby, part_top - pad_y)
                rbh = max(8, rby1 - rby)
    # One humanoid column in the lineup is ~9–14 % of frame width (wider at center).
    max_col_w = max(48, int(frame_w * (0.14 if cluster_slot == 3 else 0.12)))
    if rbw > max_col_w:
        rbw = max_col_w
    # Center on slot/part anchor — raw mask argmax pulls onto backdrop gaps.
    center_x = int(round(anchor_cx))
    rbx = max(0, center_x - rbw // 2)
    rbx1 = min(frame_w, rbx + rbw)
    if rbx1 - rbx < rbw:
        rbx = max(0, rbx1 - rbw)
    elif abs((rbx + rbx1) * 0.5 - anchor_cx) > frame_w * 0.02:
        rbx = max(0, min(frame_w - rbw, int(round(anchor_cx - rbw * 0.5))))
        rbx1 = rbx + rbw
    min_h_frac = 0.40 if cluster_slot == 0 else (0.30 if cluster_slot == 3 else 0.26)
    rby, rby1 = _lineup_extend_display_vertical(
        mask, rbx, rbx1, rby, rby1, frame_h, min_body_h_frac=min_h_frac, pad_y=pad_y
    )
    if parts:
        part_bot = max(int(p.y + p.h) for p in parts)
        rby1 = min(frame_h, max(rby1, part_bot + pad_y))
    if rbw < int(frame_w * 0.08):
        slot_cx_i = int(frame_w * _LINEUP_SLOT_X_FRACS[cluster_slot])
        x0m = max(0, slot_cx_i - int(frame_w * 0.07))
        x1m = min(frame_w, slot_cx_i + int(frame_w * 0.07))
        band = mask[rby:rby1, x0m:x1m]
        if band.any():
            xs = np.where(band > 0)[1] + x0m
            rbx = max(0, int(xs.min()) - pad_x)
            rbx1 = min(frame_w, int(xs.max()) + 1 + pad_x)
            rbw = min(max_col_w, max(8, rbx1 - rbx))
    mask_cx = _lineup_mask_torso_peak_gx(mask, rby, rby1, cluster_slot, frame_w)
    anchor_cx = _lineup_display_anchor_cx(parts, bx, by, bw, bh, frame_w, peak_gx)
    slot_cx_f = frame_w * _LINEUP_SLOT_X_FRACS[cluster_slot]
    if cluster_slot == 5:
        mask_w = 0.62
    elif cluster_slot == 0:
        mask_w = 0.28
    else:
        mask_w = 0.48
    final_cx = (1.0 - mask_w) * anchor_cx + mask_w * mask_cx
    clamp_lo = slot_cx_f - frame_w * 0.025
    clamp_hi = slot_cx_f + (
        frame_w * 0.008 if cluster_slot == 0 else frame_w * 0.025
    )
    final_cx = max(clamp_lo, min(clamp_hi, final_cx))
    center_x = int(round(final_cx))
    rbx = max(0, min(frame_w - rbw, center_x - rbw // 2))
    rbx1 = rbx + rbw
    rbh = max(8, rby1 - rby)
    return rbx, rby, rbw, rbh


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
    fov_cx: float | None = None,
    fov_cy: float | None = None,
) -> tuple[float, float] | None:
    """Centroid of red pixels in upper-chest band — stable body mass, not sky above head.

    REAL-FRAME AUDIT FIX: when the cluster bbox is much taller than the
    actual character (e.g. a real Apex enemy has merged with HUD text
    or damage numbers above/below, producing a 1.2-aspect cluster that
    spans 30-40 % of frame height), the fixed 0.30-0.50 chest band of
    the BBOX lands above the actual body. We now scan multiple candidate
    Y bands across the bbox and pick the densest 20 %-tall band — that
    band is the true body chest regardless of bbox extent. For clean
    character clusters the densest band IS at 0.30-0.50, so behaviour
    is unchanged; for merged clusters the dot lands on the body, not on
    a HUD bar.
    """
    if bw < 4 or bh < 8:
        return None
    band_height = max(2, int(round(bh * max(0.05, min(0.40, y1f - y0f)))))
    if band_height >= bh:
        band_height = max(2, bh - 1)

    # Compute row-density profile across the full bbox once.
    full = mask[by : by + bh, bx : bx + bw]
    if full.size == 0:
        return None
    row_density = (full > 0).sum(axis=1).astype(np.int64)
    if row_density.sum() < 8:
        return None

    # Cumulative sum trick: a sliding-window sum across rows.
    csum = np.cumsum(row_density)
    if band_height >= len(csum):
        band_sums = csum[-1:]
        start_idx = 0
    else:
        # band_sums[i] = sum of row_density[i : i + band_height]
        band_sums = csum[band_height - 1 :] - np.concatenate(
            ([0], csum[: -band_height])
        )
        # REAL-FRAME AUDIT FIX: when the cluster bbox contains the
        # crosshair (FOV centre), the user is almost certainly aiming
        # AT the target — so the desired chest band is the dense band
        # NEAREST to ``fov_cy``, not necessarily the GLOBALLY densest
        # band. Without this bias a "merged" cluster (real character
        # joined with HUD pixels above) sees the HUD as the densest
        # area and anchors aim into empty sky above the body.
        if fov_cy is not None and by <= fov_cy <= by + bh:
            band_centers = np.arange(len(band_sums), dtype=np.float32) + (
                band_height * 0.5
            )
            dist_to_fov = (by + band_centers) - float(fov_cy)
            sigma = max(2.0, bh * 0.10)
            decay = np.exp(-(dist_to_fov ** 2) / (2.0 * sigma * sigma))
            band_sums = band_sums.astype(np.float32) * decay
        start_idx = int(np.argmax(band_sums))

    y0 = by + start_idx
    y1 = y0 + band_height
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
    # REAL-FRAME AUDIT FIX: when ``fov_cx`` is provided and lives inside
    # the bbox X range, AND the bbox is sparsely filled (the merged-
    # cluster signature — body + floating HUD bits), weight the column-
    # density peak toward the crosshair so the dot stays on the visible
    # character. For a CLEAN humanoid silhouette (sideways character
    # with a gun arm — D5 fixture) the bbox is densely filled and the
    # bias is skipped so the torso column stays the densest.
    full_dense_col_frac = 1.0
    if full.size > 0:
        rd_full = (full > 0).sum(axis=1).astype(np.float32)
        pk_full = float(rd_full.max()) if rd_full.size else 0.0
        if pk_full >= 2.0:
            thr_full = max(2.5, pk_full * 0.18)
            full_dense_col_frac = float((rd_full >= thr_full).sum()) / float(bh)
    if (
        fov_cx is not None
        and bx <= fov_cx <= bx + bw
        and full_dense_col_frac < 0.45
    ):
        col_idx = np.arange(col_density.size, dtype=np.float32)
        dist_to_cx = (col_idx + bx) - float(fov_cx)
        sigma_x = max(2.0, bw * 0.18)
        decay = np.exp(-(dist_to_cx ** 2) / (2.0 * sigma_x * sigma_x))
        col_density = col_density * decay
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
    """Keep aim inside upper-chest band; never above head plate or into sky above bbox.

    REAL-FRAME AUDIT FIX: when the cluster has many parts (real Apex
    character whose body forms 5+ disjoint red plates) the original
    upper-chest clamp is correct: aim should sit just below the head.
    When the cluster is wide-and-tall (merged with HUD text / damage
    numbers above the body), the geometric chest band of the BBOX is
    NOT the actual chest — the body sits lower in the bbox. In that
    case we widen the Y clamp to the whole bbox interior so the
    densest-band anchor (computed in ``_mask_chest_anchor``) is not
    forced upward into HUD pixels. ``_aim_inside_body_bbox`` still
    rejects an aim that lands outside the bbox bounds entirely.
    """
    frac = max(0.32, min(0.48, torso_fraction))
    aspect = bh / max(bw, 1)
    is_merged_cluster = len(parts) >= 6 and aspect < 2.20
    if is_merged_cluster:
        y_lo = by + bh * 0.12
        y_hi = by + bh * 0.85
    else:
        y_lo = by + bh * aim_y_lo_frac
        y_hi = by + bh * min(aim_y_hi_frac, frac + 0.10)
        heads = [p for p in parts if p.role == PartRole.HEAD]
        if heads:
            head_bottom = heads[0].y + heads[0].h
            y_lo = max(y_lo, head_bottom - bh * 0.02)
    ay = max(y_lo, min(y_hi, ay))
    x_lo = bx + bw * 0.20
    x_hi = bx + bw * 0.80
    ax = max(x_lo, min(x_hi, ax))
    return ax, ay


def _bbox_has_detached_upper_fringe(
    mask: np.ndarray,
    bx: int,
    by: int,
    bw: int,
    bh: int,
) -> bool:
    """True when sparse HUD/scope pixels sit above a gap-separated body band."""
    if bw <= 0 or bh < 16:
        return False
    roi = mask[by : by + bh, bx : bx + bw]
    if roi.size == 0:
        return False
    row_d = (roi > 0).sum(axis=1).astype(np.float32)
    peak = float(row_d.max()) if row_d.size else 0.0
    if peak < 2.0:
        return False
    thr = max(2.5, peak * 0.18)
    dense = row_d >= thr
    if not bool(dense.any()):
        return False
    idx = np.flatnonzero(dense)
    y0 = int(idx[0])
    y1 = int(idx[-1])
    if y0 > bh * 0.14:
        return True
    gap_allow = max(5, int(round(bh * 0.10)))
    gap = 0
    max_gap = 0
    for i in range(1, len(idx)):
        gap = int(idx[i] - idx[i - 1] - 1)
        if gap > max_gap:
            max_gap = gap
    return max_gap >= gap_allow and y0 < bh * 0.22


def _prune_upper_fringe_parts(
    parts: list[_RedPart],
    bx: int,
    by: int,
    bw: int,
    bh: int,
    mask: np.ndarray,
) -> list[_RedPart]:
    """Drop gap-separated HUD specks above the body column (frame 45 ADS class)."""
    if bw <= 0 or bh < 16 or len(parts) < 2:
        return parts
    if not _bbox_has_detached_upper_fringe(mask, bx, by, bw, bh):
        return parts
    cutoff = by + int(round(bh * 0.38))
    kept = [p for p in parts if (p.y + p.h * 0.65) >= cutoff]
    if kept and len(kept) < len(parts):
        return kept
    return parts


def _dense_row_segments(
    row_d: np.ndarray,
    thr: float,
    gap_allow: int,
) -> list[tuple[int, int]]:
    """Return (y0, y1) inclusive row spans where density >= thr, split on vertical gaps."""
    dense = row_d >= thr
    if not bool(dense.any()):
        return []
    segments: list[tuple[int, int]] = []
    start: int | None = None
    gap = 0
    for i in range(int(row_d.size)):
        if dense[i]:
            if start is None:
                start = i
            gap = 0
        elif start is not None:
            gap += 1
            if gap > gap_allow:
                segments.append((start, i - gap))
                start = None
                gap = 0
    if start is not None:
        segments.append((start, int(row_d.size) - 1))
    return segments


def _anchor_bbox_bottom_dense_band(
    mask: np.ndarray,
    bx: int,
    by: int,
    bw: int,
    bh: int,
    scale: float,
) -> tuple[int, int, int, int]:
    """Drop gap-separated HUD fringes; anchor bbox on the lower body column."""
    if bw <= 0 or bh < 12:
        return bx, by, bw, bh
    roi = mask[by : by + bh, bx : bx + bw]
    if roi.size == 0:
        return bx, by, bw, bh
    row_d = (roi > 0).sum(axis=1).astype(np.float32)
    peak = float(row_d.max()) if row_d.size else 0.0
    if peak < 2.0:
        return bx, by, bw, bh
    thr = max(2.5, peak * 0.18)
    gap_allow = max(5, int(round(bh * 0.10)))
    segments = _dense_row_segments(row_d, thr, gap_allow)
    min_seg_h = max(10, int(round(28 * scale)))
    if len(segments) >= 2:
        # Frame 45 class: HUD fringe band above a wide ADS gap, body band below.
        # Always take the lowest dense band (largest row index / feet), never
        # bridge upward through the gap (old min_bh bridge pulled top to y~166).
        segments = [s for s in segments if s[1] - s[0] + 1 >= min_seg_h]
        if not segments:
            segments = [_dense_row_segments(row_d, thr, gap_allow)[-1]]
        y0, y1 = max(segments, key=lambda s: s[1])
    elif segments:
        y0, y1 = segments[0]
    else:
        idx = np.flatnonzero(row_d >= thr)
        y0, y1 = int(idx[0]), int(idx[-1])
    any_red = row_d > 0
    gap_red = max(4, gap_allow)
    i = y0 - 1
    empty = 0
    while i >= 0:
        if any_red[i]:
            y0 = i
            empty = 0
        else:
            empty += 1
            if empty > gap_red:
                break
        i -= 1
    i = y1 + 1
    empty = 0
    while i < int(row_d.size):
        if any_red[i]:
            y1 = i
            empty = 0
        else:
            empty += 1
            if empty > gap_red:
                break
        i += 1

    new_by = by + y0
    new_bh = max(12, y1 - y0 + 1)

    sub = roi[y0 : y1 + 1]
    col_d = (sub > 0).sum(axis=0).astype(np.float32)
    cp = float(col_d.max()) if col_d.size else 0.0
    new_bx, new_bw = bx, bw
    if cp >= 2.0:
        ct = max(2.5, cp * 0.18)
        dx = np.flatnonzero(col_d >= ct)
        if dx.size >= 2:
            new_bx = bx + int(dx[0])
            new_bw = max(4, int(dx[-1] - dx[0] + 1))

    return new_bx, new_by, new_bw, new_bh


def _tighten_merged_body_bbox(
    parts: list[_RedPart],
    bx: int,
    by: int,
    bw: int,
    bh: int,
    *,
    head_score: float,
    has_torso_part: bool,
    frame_h: int,
    aim_y: float | None = None,
) -> tuple[int, int, int, int]:
    """Trim over-extended merged-mask bbox to its head/torso/limb core.

    The Apex red-enemy mask sometimes merges three independent red regions
    into one connected cluster:
        - score-panel banners / tower icon strips above the dummy
        - the dummy body itself
        - score-panel red wires / "18" display segments below the dummy
    The aggregate bbox covers 100+ px (~30 % of frame height) of background
    noise around the actual body.  The body parts are correctly classified
    (head/torso/limb roles), but the cluster-level ``bx/by/bw/bh`` includes
    every connected pixel, so the visible green box looks alarming.

    Two-stage trim when the cluster matches the merged-mask signature
    (very tall + no head signal + at least one classified torso part):

      1) Clamp the bbox vertically to the union of the classified body
         parts' y-spans (drops HUD pixels above/below).
      2) Cap the union at frame_h * 0.18 (~80 px on 450 h) centered on
         ``aim_y`` (or torso part centroid) — even the body parts can
         include limb-classified score-panel fragments that extend the
         span past the actual dummy.  Real close enemies have head
         score >= 0.10 so this cap never fires for them.

    Width is preserved (horizontal merging is rare).  Downstream chest-band
    aim and IoU/refine checks then operate on the tightened body rectangle
    rather than the score-panel-plus-banner column.

    Pre-existing tighteners (``_anchor_bbox_bottom_dense_band``,
    ``_prune_upper_fringe_parts``) require ROW-DENSITY GAPS in the mask to
    fire; the merged-mask scenario has dense rows throughout, so neither
    triggered.  This tightener works on the part-role classification
    instead — it is the only signal that can distinguish the body parts
    from the surrounding HUD/structural pixels in the no-gap case.
    """
    if bh < int(float(frame_h) * 0.22):
        return bx, by, bw, bh
    if float(head_score) >= 0.10:
        return bx, by, bw, bh
    if not has_torso_part:
        return bx, by, bw, bh
    body_parts = [
        p for p in parts
        if p.role in (PartRole.HEAD, PartRole.TORSO, PartRole.LIMB)
    ]
    if len(body_parts) < 2:
        return bx, by, bw, bh
    y_min = min(int(p.y) for p in body_parts)
    y_max = max(int(p.y) + int(p.h) for p in body_parts)

    # Stage 2: cap merged-mask span at frame_h * 0.18 centered on the
    # aim point (the chest target derived by score_target) so even
    # noise-limb parts that extend the part-union past the actual body
    # are clipped to a humanoid-shaped window.
    max_bh = int(float(frame_h) * 0.18)
    if (y_max - y_min) > max_bh:
        # Pick a vertical anchor: aim_y first (already chest-band aware),
        # else the centroid of the torso parts (most stable role),
        # else the midpoint of the y-union.
        anchor: int
        if aim_y is not None:
            anchor = int(round(float(aim_y)))
        else:
            torsos = [p for p in body_parts if p.role == PartRole.TORSO]
            anchor = (
                int(sum(p.y + p.h * 0.5 for p in torsos) / max(1, len(torsos)))
                if torsos
                else (y_min + y_max) // 2
            )
        half = max_bh // 2
        y_min = max(y_min, anchor - half)
        y_max = min(y_max, anchor + half)
        if (y_max - y_min) < 30:
            return bx, by, bw, bh

    new_by = max(int(by), y_min)
    new_bh = max(12, y_max - new_by)
    if new_by < int(by):
        new_by = int(by)
    if new_by + new_bh > int(by) + int(bh):
        new_bh = int(by) + int(bh) - new_by
    if new_bh < 12:
        return bx, by, bw, bh
    # Only return the tighter bbox if it actually trims meaningfully
    # (avoid no-op churn on already-tight bboxes that happen to pass the
    # head/torso/tall preconditions).
    if new_bh >= int(bh) - 4:
        return bx, by, bw, bh
    return bx, new_by, bw, new_bh


def _tighten_bbox_around_aim(
    mask: np.ndarray,
    bx: int,
    by: int,
    bw: int,
    bh: int,
    aim_x: float,
    aim_y: float,
    *,
    fov_cx: float | None = None,
    fov_cy: float | None = None,
) -> tuple[int, int, int, int, float, float]:
    """Shrink the bbox to the dense mass connected to ``(aim_x, aim_y)``.

    Returns the smallest bbox whose rows/columns around the aim point
    have row-/column-density at least 12 % of the bbox-local peak. This
    drops sparse outliers (HUD text, damage numbers) that inflated the
    cluster bbox during the part-clustering stage; the dense body mass
    around the actual character then forms a tight body bbox where the
    aim falls naturally into the 28-52 % chest band.

    REAL-FRAME AUDIT FIX: the user's screenshots showed clusters whose
    bbox extended through HUD pixels above/below the visible character.
    That left the aim near the body but OUTSIDE the chest band of the
    reported bbox. With the tight bbox the chest band invariant holds.
    """
    if bw <= 0 or bh <= 0:
        return bx, by, bw, bh, aim_x, aim_y
    roi = mask[by : by + bh, bx : bx + bw]
    if roi.size == 0:
        return bx, by, bw, bh, aim_x, aim_y

    row_density = (roi > 0).sum(axis=1).astype(np.float32)
    col_density = (roi > 0).sum(axis=0).astype(np.float32)
    rd_peak = float(row_density.max()) if row_density.size else 0.0
    cd_peak = float(col_density.max()) if col_density.size else 0.0
    if rd_peak < 2.0 or cd_peak < 2.0:
        return bx, by, bw, bh, aim_x, aim_y

    aiy = int(round(aim_y - by))
    aix = int(round(aim_x - bx))
    aiy = max(0, min(bh - 1, aiy))
    aix = max(0, min(bw - 1, aix))

    # Snap the walk start to the densest row within a chest-height
    # window around the aim. Without this snap an FOV-biased aim that
    # lands between body part plates (real Apex characters have sparse
    # gaps between head/torso/limb plates) starts the expansion in a
    # row below the density threshold and the tight bbox collapses.
    # When ``fov_cy`` is provided we Gaussian-weight the window scores
    # toward the crosshair so the snap pulls TOWARD the body, not back
    # into HUD pixels above the aim.
    win_h = max(8, int(round(bh * 0.22)))
    y_lo_w = max(0, aiy - win_h)
    y_hi_w = min(bh, aiy + win_h + 1)
    if y_hi_w > y_lo_w:
        local = row_density[y_lo_w:y_hi_w].astype(np.float32)
        if fov_cy is not None:
            idx = np.arange(local.size, dtype=np.float32) + y_lo_w + by
            sigma_y = max(2.0, bh * 0.08)
            decay = np.exp(-((idx - float(fov_cy)) ** 2) / (2.0 * sigma_y * sigma_y))
            local = local * decay
        aiy = int(np.argmax(local)) + y_lo_w
    win_w = max(8, int(round(bw * 0.22)))
    x_lo_w = max(0, aix - win_w)
    x_hi_w = min(bw, aix + win_w + 1)
    if x_hi_w > x_lo_w:
        local_x = col_density[x_lo_w:x_hi_w].astype(np.float32)
        if fov_cx is not None:
            idx = np.arange(local_x.size, dtype=np.float32) + x_lo_w + bx
            sigma_x = max(2.0, bw * 0.10)
            decay_x = np.exp(-((idx - float(fov_cx)) ** 2) / (2.0 * sigma_x * sigma_x))
            local_x = local_x * decay_x
        aix = int(np.argmax(local_x)) + x_lo_w

    # Use a low floor (5 % of bbox-local peak) and a small absolute
    # pixel floor so tiny clusters (firing-range dummies) do not
    # collapse to a 1×1 bbox. We then allow a few "gap" rows below the
    # threshold during the walk — real character bodies have sparse
    # rows between head/torso/limb plates that should NOT terminate
    # the walk, but a long stretch of empty rows (the gap between the
    # body and a HUD element) WILL terminate it.
    rd_thr = max(2.5, rd_peak * 0.18)
    cd_thr = max(2.5, cd_peak * 0.18)
    max_gap = max(4, int(round(bh * 0.05)))
    max_gap_x = max(4, int(round(bw * 0.05)))

    def _walk_up(arr: np.ndarray, idx: int, thr: float, gap: int) -> int:
        gap_count = 0
        out = idx
        i = idx - 1
        while i >= 0:
            if arr[i] >= thr:
                out = i
                gap_count = 0
            else:
                gap_count += 1
                if gap_count > gap:
                    break
            i -= 1
        return out

    def _walk_down(arr: np.ndarray, idx: int, thr: float, gap: int, n: int) -> int:
        gap_count = 0
        out = idx
        i = idx + 1
        while i < n:
            if arr[i] >= thr:
                out = i
                gap_count = 0
            else:
                gap_count += 1
                if gap_count > gap:
                    break
            i += 1
        return out

    y0 = _walk_up(row_density, aiy, rd_thr, max_gap)
    y1 = _walk_down(row_density, aiy, rd_thr, max_gap, bh)
    # Recompute X-density profile using ONLY the tight Y range so the
    # X walk doesn't get inflated by HUD elements that sit above or
    # below the body. Without this restriction the X expansion fuses
    # the body column with adjacent HUD columns (ammo numbers / damage
    # text) even when those HUDs are vertically disjoint from the
    # body.
    y0_glob = max(0, y0)
    y1_glob = min(bh - 1, y1)
    if y1_glob > y0_glob:
        col_density_tight = (roi[y0_glob : y1_glob + 1] > 0).sum(axis=0).astype(
            np.float32
        )
        cd_peak_tight = float(col_density_tight.max()) if col_density_tight.size else 0.0
        if cd_peak_tight >= 2.0:
            col_density = col_density_tight
            cd_thr = max(2.5, cd_peak_tight * 0.18)
            # Re-snap aix inside the new col_density (FOV-biased).
            x_lo_w = max(0, aix - win_w)
            x_hi_w = min(bw, aix + win_w + 1)
            if x_hi_w > x_lo_w:
                local_x = col_density[x_lo_w:x_hi_w].astype(np.float32)
                if fov_cx is not None:
                    idx = np.arange(local_x.size, dtype=np.float32) + x_lo_w + bx
                    sigma_x = max(2.0, bw * 0.10)
                    decay_x = np.exp(
                        -((idx - float(fov_cx)) ** 2) / (2.0 * sigma_x * sigma_x)
                    )
                    local_x = local_x * decay_x
                aix = int(np.argmax(local_x)) + x_lo_w
    x0 = _walk_up(col_density, aix, cd_thr, max_gap_x)
    x1 = _walk_down(col_density, aix, cd_thr, max_gap_x, bw)

    new_bx = bx + x0
    new_by = by + y0
    new_bw = max(4, x1 - x0 + 1)
    new_bh = max(8, y1 - y0 + 1)
    snapped_aim_x = float(bx + aix)
    snapped_aim_y = float(by + aiy)
    # Snapped aim sits at (aix, aiy) inside the dense band by
    # construction; if for any reason the walk produced a degenerate
    # bbox that doesn't contain it, fall back to the input bbox/aim.
    if not (
        new_bx <= snapped_aim_x <= new_bx + new_bw
        and new_by <= snapped_aim_y <= new_by + new_bh
    ):
        return bx, by, bw, bh, aim_x, aim_y
    return new_bx, new_by, new_bw, new_bh, snapped_aim_x, snapped_aim_y


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
    # REAL-FRAME AUDIT FIX: the previous ``bw > frame_w * 0.32`` cap
    # rejected every real Apex character on small screenshots (310-684 px
    # frame width) — a 110 px-wide character in a 310 px frame is 35 % of
    # frame, well over the 32 % cap, and was tagged ARCHITECTURE_PANEL.
    # The intent of this gate is "walls and panels are extremely wide
    # relative to the play area"; that's only true if the silhouette is
    # also SHORT (wider than tall). A tall+wide silhouette with aspect>=1
    # is a near-camera body, not a panel. Reject only on:
    #   * extreme width (> 60 % of frame), regardless of aspect, OR
    #   * 32-60 % width AND aspect < 1.20 (short+wide → wall)
    if bw > frame_w * 0.60:
        return RejectReason.ARCHITECTURE_PANEL
    merged_lineup_body = (
        len(parts) >= 10
        and aspect >= 0.85
        and bh >= frame_h * 0.22
    )
    if bw > frame_w * 0.32 and aspect < 1.05 and not merged_lineup_body:
        return RejectReason.ARCHITECTURE_PANEL
    # PHASE-5 AUDIT FIX (D-MED horizontal_stripe): a back-view crouch
    # close-range body sits at aspect ~0.91 in img1 and only barely
    # survives. Bypass the borderline aspect < 0.95 gate when the
    # cluster has many parts AND a clear vertical extent — both
    # signals that this is a humanoid silhouette, not a HUD bar /
    # horizontal stripe.
    scale_hs = _scale(frame_w, frame_h)
    crouch_humanoid = (
        len(parts) >= 8
        and bh >= 30 * scale_hs
    )
    if aspect < 0.95 and not crouch_humanoid:
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
    #
    # PHASE-6 AUDIT FIX (D-CRIT3): the previous gate (aspect >= 1.55,
    # bw < 12% frame_w, bh > 11% frame_h) accepted any tall column —
    # buildings, antennas, cloud columns, the player's own scope+gun
    # vertical assembly. Tighten bh from 0.11 → 0.20 so sky/cloud tiles
    # of moderate width are no longer admitted. The len(parts) >= 2 AND
    # torso_s / red_coverage / motion_overlap confirmations live in
    # _body_structure_reject and analyze_figure (where those scores are
    # in scope). Here we keep the gate permissive enough that synthetic
    # single-part body tests (and real one-blob short-range silhouettes)
    # still survive the SOLID_WALL bypass, and rely on downstream
    # stages to enforce the structure / torso / colour confirmation.
    humanoid_silhouette = (
        aspect >= 1.55
        and bw < frame_w * 0.12
        and bh > frame_h * 0.20
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
    fov_cx: float | None = None,
    fov_cy: float | None = None,
) -> tuple[float, float]:
    """
    Upper-chest anchor from mask mass in torso band; clamped below head, inside bbox.
    """
    frac = max(0.34, min(0.46, torso_fraction))
    ax = bx + bw * 0.5
    ay = by + bh * frac

    chest = _mask_chest_anchor(
        mask, bx, by, bw, bh, y0f=0.32, y1f=0.50,
        fov_cx=fov_cx, fov_cy=fov_cy,
    )
    if chest is not None:
        ax = 0.40 * ax + 0.60 * chest[0]
        ay = 0.50 * ay + 0.50 * chest[1]

    # REAL-FRAME AUDIT FIX: when the crosshair sits inside this cluster
    # bbox AND the cluster is *demonstrably* merged with HUD pixels —
    # row-density coverage drops well below a normal contiguous body —
    # the user is aiming AT the body, so blend the densest-band anchor
    # toward the crosshair so the dot lands on the visible character,
    # not on a HUD bar at the geometric 38 % position. Gate strictly on
    # row sparsity so that a clean humanoid silhouette (sideways
    # character with extended gun arm — the D5 fixture) keeps the
    # mask-mass anchor and doesn't drift toward the crosshair, which
    # otherwise sits on the weapon arm.
    if (
        fov_cx is not None
        and fov_cy is not None
        and bx <= fov_cx <= bx + bw
        and by <= fov_cy <= by + bh
    ):
        bbox_roi_for_blend = mask[by : by + bh, bx : bx + bw]
        dense_blend_frac = 1.0
        if bbox_roi_for_blend.size > 0:
            rd = (bbox_roi_for_blend > 0).sum(axis=1).astype(np.float32)
            pk = float(rd.max()) if rd.size else 0.0
            if pk >= 2.0:
                thr = max(2.5, pk * 0.18)
                dense_blend_frac = float((rd >= thr).sum()) / float(bh)
        cluster_is_merged = (
            dense_blend_frac < 0.40
            and len(parts) >= 6
            and any(p.role == PartRole.TORSO for p in parts)
        )
        if cluster_is_merged:
            ax = 0.35 * ax + 0.65 * float(fov_cx)
            ay = 0.35 * ay + 0.65 * float(fov_cy)

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
            # PHASE-5 AUDIT FIX (D-HIGH-B): when no part is classified
            # TORSO and we average cx of all parts, an extended gun arm
            # in a side-view frame pulls aim X laterally onto the
            # weapon. Reject outlier parts whose cx deviates from the
            # cluster's median cx by more than 1.4 * MAD and whose
            # area is < 60 % of the median area — those are the
            # weapon-arm parts. Keep at least 2 parts in the average
            # so we don't collapse to a single fragment.
            cx_arr = np.array([p.cx for p in parts], dtype=np.float32)
            area_arr = np.array([p.area for p in parts], dtype=np.float32)
            median_cx = float(np.median(cx_arr))
            median_area = float(np.median(area_arr))
            mad_cx = float(np.median(np.abs(cx_arr - median_cx)))
            if mad_cx <= 0.0:
                mad_cx = 1.0
            kept = [
                p
                for p in parts
                if not (
                    abs(p.cx - median_cx) > 1.4 * mad_cx
                    and p.area < 0.6 * median_area
                )
            ]
            if len(kept) < 2:
                kept = list(parts)
            cx_parts = float(np.mean([p.cx for p in kept]))
            cy_parts = float(np.mean([max(y_lo, min(y_hi, p.cy)) for p in kept]))
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
    fov_cx: float | None = None,
    fov_cy: float | None = None,
    lineup_display: bool = False,
) -> _FigureAnalysis:
    scale = _scale(frame_w, frame_h)
    bx, by, bw, bh = _cluster_bbox(parts)
    pruned = _prune_upper_fringe_parts(parts, bx, by, bw, bh, mask)
    if len(pruned) < len(parts):
        parts = pruned
        bx, by, bw, bh = _cluster_bbox(parts)
    isolated_crosshair_body = False
    if _should_isolate_crosshair_body(
        parts, bx, by, bw, bh, frame_w, frame_h, fov_cx, fov_cy
    ):
        isolated = _isolate_crosshair_body_parts(
            parts, frame_w, frame_h, float(fov_cx), float(fov_cy)  # type: ignore[arg-type]
        )
        if len(isolated) < len(parts):
            parts = isolated
            bx, by, bw, bh = _cluster_bbox(parts)
            isolated_crosshair_body = True
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
        fov_cx=fov_cx,
        fov_cy=fov_cy,
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
    fcx = frame_w * 0.5 if fov_cx is None else float(fov_cx)
    fcy = frame_h * 0.5 if fov_cy is None else float(fov_cy)
    dist_fov = math.hypot(bx + bw * 0.5 - fcx, by + bh * 0.5 - fcy)
    close_fov_humanoid = (
        dist_fov < min(frame_w, frame_h) * 0.28
        and bh >= 72 * scale
        and aspect >= 1.08
        and len(parts) >= 2
        and (limb_s >= 0.30 or has_head_part)
    )
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
            # PHASE-6 AUDIT FIX (D-CRIT3): tall narrow uniform-color
            # silhouette acceptance now requires bh > frame_h * 0.20 AND
            # torso_s >= 0.40. Without these tightenings a single tall
            # column (building, antenna, scope/gun assembly column, cloud)
            # was admitted whenever it filled ~70 px and had torso ~0.25.
            # The new thresholds permit only bodies that show real torso
            # mass AND are at least ~1/5 of the frame in height.
            aspect >= 1.55
            and bw < frame_w * 0.12
            and bh >= 70 * scale
            and bh > frame_h * 0.20
            and torso_s >= 0.40
        ):
            structure_ok = True
        elif v_score < 0.35 or fill > 0.88:
            structure_ok = False
    foot_y = by + bh
    if foot_y < frame_h * 0.42 and len(parts) < 3 and not close_fov_humanoid:
        structure_ok = False
    if close_fov_humanoid and body_shape >= 0.55 and limb_s >= 0.28:
        structure_ok = True

    has_classified_torso = any(p.role == PartRole.TORSO for p in parts)
    if structure_ok and len(parts) >= 3 and not has_classified_torso and align_s < 0.35:
        structure_ok = False

    accepted = body_shape >= min_accept and structure_ok
    reason = RejectReason.OK if accepted else RejectReason.NO_BODY_STRUCTURE
    if accepted and body_shape < min_accept:
        reason = RejectReason.LOW_SCORE

    fcx = float(fov_cx) if fov_cx is not None else frame_w * 0.5
    tower_column = (
        bw < frame_w * 0.09
        and by < frame_h * 0.30
        and abs(bx + bw * 0.5 - fcx) < frame_w * 0.16
        and v_score >= 0.85
        and fill >= 0.60
        and len(parts) >= 2
        and (aspect >= 2.0 or (bh >= bw * 1.30 and len(parts) >= 4))
    )
    if accepted and tower_column:
        accepted = False
        reason = RejectReason.ARCHITECTURE_PANEL

    if accepted and (
        close_fov_humanoid
        or _bbox_has_detached_upper_fringe(mask, bx, by, bw, bh)
    ):
        abx, aby, abw, abh = _anchor_bbox_bottom_dense_band(
            mask, bx, by, bw, bh, scale
        )
        if abw >= 4 and abh >= 12:
            bx, by, bw, bh = abx, aby, abw, abh

    ax, ay = _figure_aim_point(
        parts, mask, bx, by, bw, bh, torso_aim_fraction,
        aim_y_min_fraction, aim_y_max_fraction,
        fov_cx=fov_cx, fov_cy=fov_cy,
    )

    # REAL-FRAME AUDIT FIX: when the cluster bbox is "merged" (many
    # disjoint red parts in one column — character + HUD damage text +
    # weapon ammo etc.), the OUTER bbox spans the full mask extent,
    # which leaves the aim near the body but OUTSIDE the chest band of
    # that outer bbox. Tighten the reported bbox to the dense mass
    # around the aim so the chest-band invariant holds for downstream
    # tracker / overlay code. Clean small-character clusters skip this
    # (the tight bbox returns the same coordinates because the mask is
    # already dense edge to edge).
    #
    # IMPORTANT: gate the tighten by row-density coverage. A real
    # character cluster (synthetic firing-range dummy with separated red
    # plates, or a real Apex body whose mask is fully connected) keeps
    # >= 45 % of bbox rows above the 18 %-of-peak density floor. A
    # merged cluster (body + floating HUD text inside one outer bbox)
    # drops well below 30 %. Only the latter should be tightened — the
    # former is already tight and shrinking would lose body extent.
    bbox_roi = mask[by : by + bh, bx : bx + bw] if (bw > 0 and bh > 0) else None
    dense_row_frac = 1.0
    if bbox_roi is not None and bbox_roi.size > 0:
        row_d = (bbox_roi > 0).sum(axis=1).astype(np.float32)
        peak_d = float(row_d.max()) if row_d.size else 0.0
        if peak_d >= 2.0:
            thr = max(2.5, peak_d * 0.18)
            dense_row_frac = float((row_d >= thr).sum()) / float(bh)
    bbox_aspect = bh / max(bw, 1)
    # PHASE-5 AUDIT FIX (D-MED tighten): the previous gate required
    # >= 5 parts which skipped img2's body+HUD merger at parts=4 and
    # left a 190 px bbox with only ~13 % red coverage. Lower the part
    # floor to >= 3, and also bypass the part-count gate when the
    # cluster shows clear sparsity (dense_row_frac < 0.30) — that
    # alone is a strong tighten signal regardless of part count.
    cluster_is_merged_for_tighten = (
        not lineup_display
        and (
            dense_row_frac < 0.30
            or (
                len(parts) >= 3
                and (
                    dense_row_frac < 0.45  # sparse rows → body + floating HUD text
                    or bbox_aspect < 1.65  # wide-ish → body + horizontal HUD merge
                )
            )
        )
    )
    if cluster_is_merged_for_tighten:
        orig_by, orig_bh = by, bh
        tbx, tby, tbw, tbh, snap_ax, snap_ay = _tighten_bbox_around_aim(
            mask, bx, by, bw, bh, ax, ay,
            fov_cx=fov_cx, fov_cy=fov_cy,
        )
        if tbw >= 8 and tbh >= 12 and (tbw != bw or tbh != bh):
            collapsed = tbh < max(12, int(bh * 0.22)) and bh >= frame_h * 0.12
            if not _tighten_would_discard_crosshair_body(
                orig_by, orig_bh, tby, tbh, frame_h, fov_cy
            ) and not collapsed:
                bx, by, bw, bh = tbx, tby, tbw, tbh
                aspect = bh / max(bw, 1)
                # Use the snapped aim and re-clamp into the tight bbox
                # chest band so the chest-band invariant (28-52 %) holds.
                # The clamp margins must match ``_aim_inside_body_bbox``
                # (margin_x=0.12, margin_y_top=0.08, margin_y_bot=0.12) so
                # the next-line sky-blob check doesn't reject the aim by a
                # one-pixel rounding error.
                chest_y_lo = tby + tbh * 0.28
                chest_y_hi = tby + tbh * 0.52
                x_lo = tbx + tbw * 0.13
                x_hi = tbx + tbw - tbw * 0.13
                ax = max(x_lo, min(x_hi, snap_ax))
                ay = max(chest_y_lo, min(chest_y_hi, snap_ay))

    if isolated_crosshair_body:
        bx, by, bw, bh = _expand_bbox_scope_headroom(bx, by, bw, bh, frame_h)

    if lineup_display:
        bx, by, bw, bh = _refine_lineup_display_bbox(
            parts, mask, bx, by, bw, bh, frame_w, frame_h,
            fov_cx=fov_cx, fov_cy=fov_cy,
        )
        aspect = bh / max(bw, 1)
        ax = bx + bw * 0.5
        ay = by + bh * 0.38

    if not lineup_display and not _aim_inside_body_bbox(ax, ay, bx, by, bw, bh):
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
    lineup_wide_fov = _is_lineup_wide_fov(float(fov_radius), w, h)
    clusters = _expand_lineup_clusters(
        _cluster_parts(parts, w, h), w, h, mask=mask, fov_cx=cx, fov_cy=cy
    )
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
            fov_cx=cx,
            fov_cy=cy,
            lineup_display=lineup_wide_fov,
        )
        dist = float(np.hypot(fig.aim_x - cx, fig.aim_y - cy))
        accepted = fig.accepted and dist <= fov_radius
        reason = fig.reject_reason.value
        if fig.accepted and dist > fov_radius:
            accepted = False
            reason = RejectReason.OUTSIDE_FOV.value
        if accepted and _is_sparse_rim_fp(
            fig.fill_ratio,
            fig.solidity,
            dist,
            float(fov_radius),
            part_count=fig.part_count,
            body_shape_score=fig.body_shape_score,
            lineup_wide_fov=lineup_wide_fov,
        ):
            accepted = False
            reason = "sparse_rim_fp"
        if accepted and _is_damage_glyph_fp(
            fig.part_count, fig.by, fig.bh, h, fov_cy=cy
        ):
            accepted = False
            reason = "damage_glyph"
        if accepted and lineup_wide_fov:
            wide_reject = _lineup_wide_cluster_reject(fig, w, h)
            if wide_reject is not None:
                accepted = False
                reason = wide_reject

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
    if lineup_wide_fov:
        _prune_lineup_candidates(out, w, h)
    return out, mask, parts


def render_debug_artifacts(
    frame_bgr: np.ndarray,
    candidates: list[CandidateInfo],
    selected: Target | None,
    fov_radius: int,
    fov_center_x: float,
    fov_center_y: float,
    mask: np.ndarray | None = None,
    *,
    display_fov_radius: int | None = None,
) -> np.ndarray:
    """Composite: original + mask tint + candidate bboxes + upper-chest aim point."""
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    cx, cy = int(round(fov_center_x)), int(round(fov_center_y))

    if mask is not None:
        tint = np.zeros_like(out)
        tint[:, :] = (0, 255, 0)
        out = np.where(mask[:, :, None] > 0, cv2.addWeighted(out, 0.55, tint, 0.45, 0), out)

    ring_r = int(display_fov_radius) if display_fov_radius is not None else int(fov_radius)
    cv2.circle(out, (cx, cy), ring_r, (0, 255, 0), 2)
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
        build_red_enemy_mask(
            frame_bgr,
            hsv=(context._frame_hsv if context is not None else None),
        )
        if _normalize_detection_mode(detection_mode) == DETECTION_MODE_APEX
        else None
    )

    lineup_wide_fov = _is_lineup_wide_fov(float(fov_radius), w, h)
    parts = _extract_parts(mask, w, h, fov_cx=cx, fov_cy=cy)
    clusters = _expand_lineup_clusters(
        _cluster_parts(parts, w, h), w, h, mask=mask, fov_cx=cx, fov_cy=cy
    )
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
            fov_cx=cx,
            fov_cy=cy,
            lineup_display=lineup_wide_fov,
        )
        mc = _max_part_circularity(body_parts)
        dist_c = float(np.hypot(fig.aim_x - cx, fig.aim_y - cy))
        dbg_reject = fig.reject_reason.value
        if fig.accepted and lineup_wide_fov:
            wide_reject = _lineup_wide_cluster_reject(fig, w, h)
            if wide_reject is not None:
                dbg_reject = wide_reject
        line = (
            f"cand[{idx}] reject={dbg_reject} body={fig.body_shape_score:.2f} "
            f"head={fig.head_score:.2f} torso={fig.torso_score:.2f} limb={fig.limb_stack_score:.2f} "
            f"aspect={fig.aspect:.2f} fill={fig.fill_ratio:.2f} circ={mc:.2f} "
            f"bbox=({fig.bx},{fig.by},{fig.bw}x{fig.bh}) area={fig.total_area:.0f} "
            f"dist={dist_c:.0f} parts={fig.part_count} | {fig.debug_detail}"
        )
        if debug:
            lines.append(line)

        if not fig.accepted:
            continue
        if lineup_wide_fov:
            wide_rej = _lineup_wide_cluster_reject(fig, w, h)
            if wide_rej is not None:
                if debug:
                    lines.append(f"cand[{idx}] drop {wide_rej}")
                continue

        if _is_damage_glyph_fp(
            len(body_parts), fig.by, fig.bh, h, fov_cy=cy
        ):
            if debug:
                lines.append(
                    f"cand[{idx}] drop damage_glyph parts={len(body_parts)} "
                    f"bh={fig.bh} y={fig.by}"
                )
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
        if _is_sparse_rim_fp(
            fig.fill_ratio,
            fig.solidity,
            dist,
            float(fov_radius),
            part_count=fig.part_count,
            body_shape_score=fig.body_shape_score,
            lineup_wide_fov=lineup_wide_fov,
        ):
            if debug:
                lines.append(
                    f"cand[{idx}] drop sparse_rim_fp fill={fig.fill_ratio:.2f} "
                    f"solidity={fig.solidity:.2f} dist={dist:.0f}"
                )
            continue
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

        # PHASE-6 AUDIT FIX (D-CRIT3 post-acceptance veto): the
        # geometry-only / parts-stack structure_ok branches in
        # analyze_figure accept tall multi-part clusters whose torso
        # score is essentially zero (e.g. img7 gun-column at parts=9
        # aspect=2.05 head=1.00 torso=0.00 limb=1.00 — a perfectly
        # stacked column of head/limb-shaped fragments with NO wide
        # central mass band). Real Apex characters ALWAYS have at
        # least one of:
        #   * torso_s >= 0.20 (their plate carrier / chest mass registers), OR
        #   * red_coverage >= 0.05 (enemy outline highlights their body), OR
        #   * motion_overlap >= 0.20 (they are moving — confirmed live).
        # The player's own viewmodel column has none of those, so we
        # reject it here BEFORE it can become a lock candidate. The
        # thresholds are empirically calibrated against the 7 real
        # Apex screenshots (img3 red_cov=0.098 — keeps; img7 gun
        # red_cov=0.004 — drops).
        has_torso_part = any(p.role == PartRole.TORSO for p in body_parts)
        align_for_vm = _part_alignment_score(body_parts, _scale(w, h))
        if red_filled is not None and fig.accepted and _is_viewmodel_column_fp(
            body_parts,
            bx=float(fig.bx),
            by=float(fig.by),
            bw=float(fig.bw),
            bh=float(fig.bh),
            frame_w=w,
            frame_h=h,
            fov_cx=cx,
            fov_cy=cy,
            red_cov=red_cov,
            align_s=align_for_vm,
            part_count=fig.part_count,
            torso_score=fig.torso_score,
        ):
            if debug:
                lines.append(
                    f"cand[{idx}] drop {RejectReason.VIEWMODEL_COLUMN.value} "
                    f"red_cov={red_cov:.3f} align={align_for_vm:.2f}"
                )
            continue
        if red_filled is not None:
            if fig.accepted and red_cov < MIN_ENEMY_RED_COVERAGE:
                mov_ov = 0.0
                if context is not None:
                    mov_ov = context.motion_coverage_ratio(
                        fig.bx, fig.by, fig.bw, fig.bh
                    )
                if mov_ov < 0.25:
                    if debug:
                        lines.append(
                            f"cand[{idx}] drop low_red_cov={red_cov:.3f} "
                            f"mov={mov_ov:.2f}"
                        )
                    continue
            if fig.accepted and fig.fill_ratio < 0.12 and red_cov < 0.05:
                if debug:
                    lines.append(
                        f"cand[{idx}] drop sparse_red fill={fig.fill_ratio:.2f} "
                        f"red_cov={red_cov:.3f}"
                    )
                continue
        if (
            fig.accepted
            and red_filled is not None
            and red_cov < 0.05
            and not has_enemy_red_torso_evidence(
                body_parts, torso_score=fig.torso_score, red_cov=red_cov
            )
        ):
            mov_ov = 0.0
            if context is not None:
                mov_ov = context.motion_coverage_ratio(fig.bx, fig.by, fig.bw, fig.bh)
            if mov_ov < 0.20:
                if debug:
                    lines.append(
                        f"cand[{idx}] drop no_torso_no_red torso_part={has_torso_part} "
                        f"torso={fig.torso_score:.2f} red_cov={red_cov:.3f} mov={mov_ov:.2f}"
                    )
                continue

        if red_filled is not None and fig.accepted and _is_background_red_clutter(
            fig, body_parts, red_cov=red_cov
        ):
            if debug:
                lines.append(
                    f"cand[{idx}] drop {RejectReason.BACKGROUND_CLUTTER.value} "
                    f"area={fig.total_area:.0f} bh={fig.bh} red_cov={red_cov:.3f}"
                )
            continue

        tight_bx, tight_by, tight_bw, tight_bh = _tighten_merged_body_bbox(
            body_parts,
            fig.bx, fig.by, fig.bw, fig.bh,
            head_score=fig.head_score,
            has_torso_part=has_torso_part,
            frame_h=int(h),
            aim_y=aim_y,
        )
        targets.append(
            Target(
                centroid_x=aim_x,
                centroid_y=aim_y,
                area=fig.total_area,
                distance_to_center=dist,
                bbox_x=tight_bx,
                bbox_y=tight_by,
                bbox_w=tight_bw,
                bbox_h=tight_bh,
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
                has_classified_torso=has_torso_part,
                reject_reason=fig.reject_reason.value,
            )
        )

    if debug:
        lines.append(f"accepted={len(targets)} from {len(clusters)} clusters / {len(parts)} parts")
    return targets, lines


def target_is_range_board_fp(
    target: Target,
    *,
    motion_overlap: float = 0.0,
    frame_w: int = 0,
    frame_h: int = 0,
    fov_cx: float | None = None,
) -> bool:
    """Firing-range hazard boards: saturated static red columns, not humanoids."""
    bw = max(1.0, float(target.bbox_w))
    bh = max(1.0, float(target.bbox_h))
    if (
        frame_w > 0
        and frame_h > 0
        and fov_cx is not None
        and target_is_central_tower_banner_fp(
            target,
            frame_w=frame_w,
            frame_h=frame_h,
            fov_cx=float(fov_cx),
            motion_overlap=motion_overlap,
        )
    ):
        return True
    if target.has_classified_torso and int(target.part_count) >= 3:
        if not (
            bw <= max(32.0, frame_w * 0.07 if frame_w > 0 else 32.0)
            and bh >= bw * 2.4
            and float(target.red_coverage) >= 0.14
            and motion_overlap < 0.09
        ):
            return False
    if (
        target.body_shape_score >= 0.78
        and target.head_score >= 0.30
        and int(target.part_count) >= 2
    ):
        return False
    red = float(target.red_coverage)
    if red < 0.18:
        return False
    bh = float(target.bbox_h)
    bw = max(1.0, float(target.bbox_w))
    if bh < bw * 1.12:
        return False
    if float(target.fill_ratio) < 0.40 and red < 0.30:
        return False
    if motion_overlap >= 0.07:
        return False
    if int(target.part_count) <= 2 and (
        not target.has_classified_torso or float(target.torso_score) < 0.32
    ):
        return red >= 0.20
    return False


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
    # Pure screen-height proxy for depth — do NOT multiply by center distance;
    # that was letting small central background blobs beat a larger edge dummy.
    return closeness_unit * fov_radius * 0.58


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
    dist_term = max(0.0, fov_radius - target.distance_to_center) * distance_weight * 0.28
    area_term = min(math.sqrt(target.area), 80.0) * area_weight
    closeness_term = _closeness_bonus(target, fov_radius) if include_closeness else 0.0
    # Motion-overlap bonus rewards candidates whose bbox covers an inter-frame
    # diff region — these are confirmed *moving* silhouettes (real Apex enemies)
    # vs static red walls/panels whose shape may look humanoid by accident.
    #
    # PHASE-6 AUDIT FIX (D-CRIT2): the motion_bonus is BOOSTING for real
    # bodies but was previously also RESCUING weak candidates (sky, walls,
    # HUD numerals). A cloud edge with body_shape_score=0.35 could overtake
    # a body=0.70 real target purely on motion overlap. Gate the bonus on a
    # body-score floor of 0.55 so motion only ever boosts a candidate that
    # already independently looks like a body.
    if target.body_shape_score < 0.55:
        motion_bonus = 0.0
    else:
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
    rim_frac = target.distance_to_center / max(fov_radius, 1.0)
    if rim_frac > 0.68:
        penalty += fov_radius * 0.50 * ((rim_frac - 0.68) / 0.32) ** 1.4
    if target.fill_ratio < 0.32 and rim_frac > 0.52:
        penalty += fov_radius * 0.85
    if target_is_range_board_fp(
        target,
        motion_overlap=motion_overlap,
        frame_w=0,
        frame_h=0,
        fov_cx=None,
    ):
        penalty += fov_radius * 1.35
    if motion_overlap < 0.04 and red_cov >= 0.24 and int(target.part_count) <= 2:
        if not target.has_classified_torso or float(target.torso_score) < 0.34:
            penalty += fov_radius * 0.75
    if int(target.part_count) >= 3 and target.has_classified_torso:
        penalty -= fov_radius * 0.12
    if center_y is not None:
        # REAL-FRAME AUDIT FIX: the previous flat "centroid_y < center_y"
        # penalty (*2.8 per pixel) preferred small clusters BELOW the
        # crosshair over big visible characters whose chest sat slightly
        # above the crosshair. In Apex the player commonly aims AT the
        # chest band, which means a target whose centroid is at or a
        # little above the screen-center y is the EXPECTED case, not an
        # outlier. The penalty now only fires when the centroid is far
        # above the crosshair (more than a third of the FOV radius), and
        # scales gently so a chest just above center keeps a winning
        # score. Targets clearly below the FOV (e.g. floor litter) still
        # incur the foot-above-center penalty below.
        above_dead = center_y - fov_radius * 0.35
        if target.centroid_y < above_dead:
            penalty += (above_dead - target.centroid_y) * 1.2
        if target.bbox_h > 0:
            # REAL-FRAME AUDIT FIX: the previous formulation
            # ``chest_hi = bbox_y + bbox_h * 0.48`` and penalizing any aim
            # above that fired on real Apex enemies whose cluster bbox
            # had merged with HUD text above the body. For such bboxes
            # the actual chest sits in the LOWER half of the bbox, so a
            # correct chest aim was wrongly punished by hundreds of
            # pixels of penalty. Penalize only when the aim lands in the
            # clear HEAD zone (top 20 % of bbox) instead.
            head_zone_bottom = target.bbox_y + target.bbox_h * 0.20
            if target.centroid_y < head_zone_bottom:
                penalty += (head_zone_bottom - target.centroid_y) * 4.5
        foot_y = target.bbox_y + target.bbox_h
        # REAL-FRAME AUDIT FIX: previous threshold (`fov_radius * 0.10`)
        # punished any enemy whose feet sat above centre-y, which is the
        # NORMAL framing when the player looks slightly down at a target
        # (the back-view enemy in img1 stands on a barrier 40 px above
        # centre — perfectly valid yet got hit by a +2.2*fov penalty,
        # ~420 px of cost on a 192 px FOV, enough to drop a body=1.0
        # head=1.0 target below noise). Only treat *very* floating bodies
        # (feet > 0.45 * fov above centre) as sky blobs.
        if foot_y < center_y - fov_radius * 0.45:
            penalty += fov_radius * 2.2
        if bbox_mid_in_sky_band(target.bbox_y, target.bbox_h, center_y):
            penalty += fov_radius * 1.8
    # REAL-FRAME AUDIT FIX (img1 dummy): when ``analyze_figure`` has
    # already produced a strong humanoid signature (body >= 0.80 with at
    # least one of head / torso / limb confirming structure), the cluster
    # IS a humanoid silhouette by independent geometric evidence — don't
    # re-punish it for the bbox aspect, which is a much coarser signal.
    # Walls / pipes / health bars register body_shape near 0 already so
    # they are unaffected. Without this gate, tight 45x41 dummies (img1)
    # got slapped with +1.8*fov even though body=1.0 head=1.0 limb=1.0.
    strong_humanoid = (
        target.body_shape_score >= 0.80
        and (
            target.head_score >= 0.70
            or target.torso_score >= 0.70
            or target.limb_stack_score >= 0.70
        )
    )
    bbox_h_small = target.bbox_h < 50
    if not strong_humanoid:
        if target.bbox_w >= target.bbox_h * 1.25:
            # Clearly horizontal — wall / pipe / health bar geometry.
            penalty += fov_radius * 0.8
        elif target.bbox_w >= target.bbox_h * 1.05 and not bbox_h_small:
            # Marginal aspect — gentle penalty on bigger bboxes only.
            penalty += fov_radius * 0.20
    elif target.bbox_w >= target.bbox_h * 1.40:
        # Even a "strong humanoid" cluster shouldn't be drastically wider
        # than tall. Apply a modest cap so a real wall that somehow nudges
        # body>=0.80 still loses points.
        penalty += fov_radius * 0.35
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
    # REAL-FRAME AUDIT FIX (img1): split the wider-than-tall penalty so
    # ``analyze_figure``-confirmed humanoids (body>=0.80 + structure)
    # don't get the *1.8 cliff just because the tight bbox is slightly
    # stocky. Walls / boards still trip it because their body_shape is
    # near 0 and they fall into the un-gated branch.
    if not strong_humanoid:
        if target.bbox_h < target.bbox_w * 0.95:
            penalty += fov_radius * 1.8
        elif target.bbox_h < target.bbox_w * 1.25:
            penalty += fov_radius * 0.45
    else:
        if target.bbox_h < target.bbox_w * 0.75:
            # Even a "humanoid" cluster shouldn't be very stocky.
            penalty += fov_radius * 0.6
    # Thin-stripe penalty: a real character's bbox should be at least a
    # dozen pixels wide. A 9-px-wide cluster (e.g. crosshair pixel
    # column / vertical HUD bar fragment) is noise even when the body
    # score smells humanoid because the analyser sees a tall narrow
    # column. Gate by ``bbox_w`` and ``part_count`` so distance targets
    # (still tall + multi-part) are unaffected.
    if target.bbox_w < 14 and target.part_count <= 2:
        penalty += fov_radius * 1.2
    if target_is_background_clutter(target):
        penalty += fov_radius * 2.0
  # Wide solid props (crates, panels) in the firing range — not humanoid columns.
    if (
        target.fill_ratio > 0.82
        and target.part_count <= 2
        and target.bbox_h < target.bbox_w * 1.10
    ):
        penalty += fov_radius * 1.6
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


def _should_retarget_closer_humanoid(
    locked: Target,
    challenger: Target,
    *,
    detect_fov: float,
    display_fov: float,
    fov_cx: float,
    fov_cy: float,
    frame_w: int,
    frame_h: int,
) -> bool:
    """Allow breaking HIGH6 empty-pool hold when a much closer in-ring humanoid exists."""
    if challenger.body_shape_score < 0.52:
        return False
    if target_is_background_clutter(challenger) or target_is_environment_column(
        challenger, center_y=fov_cy, frame_w=frame_w, frame_h=frame_h
    ):
        return False
    if target_is_viewmodel_column_fp(
        challenger, frame_w=frame_w, frame_h=frame_h, fov_cx=fov_cx, fov_cy=fov_cy
    ):
        return False
    if bbox_mid_in_sky_band(challenger.bbox_y, challenger.bbox_h, fov_cy):
        return False
    sd = float(locked.distance_to_center)
    cd = float(challenger.distance_to_center)
    if cd >= sd * 0.58 and cd >= 58.0:
        return False
    # ADS-close silhouettes can be narrow after bottom-band anchor; do not
    # require challenger taller than a stale rim lock when much closer in-ring.
    if cd < sd * 0.52:
        min_ch_h = max(24, int(locked.bbox_h * 0.24))
    else:
        min_ch_h = max(64, int(locked.bbox_h * 1.28))
    if challenger.bbox_h < min_ch_h:
        return False
    if sd < display_fov * 0.52:
        return False
    if cd > display_fov * 0.90:
        return False
    if _bbox_iou(
        locked.bbox_x, locked.bbox_y, locked.bbox_w, locked.bbox_h,
        challenger.bbox_x, challenger.bbox_y, challenger.bbox_w, challenger.bbox_h,
    ) >= 0.14:
        return False
    return True


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
    display_fov_radius: float | None = None,
) -> DetectionResult:
    h, w = frame_bgr.shape[:2]
    cx = w / 2.0 if fov_center_x is None else fov_center_x
    cy = h / 2.0 if fov_center_y is None else fov_center_y
    # PERF: pre-compute HSV once and stash on the context so the mask
    # builders (build_hsv_mask / build_red_enemy_mask /
    # build_chroma_spread_mask) all share it.  Profiled at ~0.17 ms per
    # cvtColor; eliminating 2-4 redundant conversions saves 0.3-0.7 ms
    # per detect call → directly reduces detect_ms p99 and lowers the
    # frame-overrun rate in the 60 fps capture loop.
    if context is not None:
        context._frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        context._frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    ring_fov = (
        float(display_fov_radius)
        if display_fov_radius is not None and display_fov_radius > 0
        else float(fov_radius) * 0.757
    )

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

    # Lazy red-mask coverage cache for pool-hold validation. Built only
    # when we need to validate that the sticky lock still has red mass
    # in the current frame (i.e. when we're about to return a synthesised
    # sticky_pool_hold lock). ~1-2 ms per build on 800x450 frames.
    _validation_mask: list[np.ndarray | None] = [None]

    def _sticky_has_red_evidence(t: Target) -> bool:
        if t is None:
            return False
        # When no HSV ranges are configured (e.g. shape-only callers, unit
        # tests with synthetic frames), there is no red mask to validate
        # against — fall back to the legacy "trust the sticky" behaviour
        # so we don't break shape-only detection pipelines.
        if not hsv_ranges:
            return True
        if _validation_mask[0] is None:
            _validation_mask[0] = build_hsv_mask(
                frame_bgr,
                hsv_ranges or [],
                hsv=(context._frame_hsv if context is not None else None),
            )
        cov = _bbox_red_coverage(
            _validation_mask[0], t.bbox_x, t.bbox_y, t.bbox_w, t.bbox_h
        )
        # Real enemy bodies score 0.10-0.30+ red coverage in their bbox.
        # The floor for "still red enough to hold" is half of the locked
        # target's original red_coverage, clamped to [0.05, 0.08]. This
        # lets brief partial occlusion (~50% red drop) keep the lock alive
        # while a sustained drop (no red mass where the lock thinks the
        # enemy is) instantly invalidates the pool-hold ghost.
        base = float(getattr(t, "red_coverage", 0.0))
        floor = max(0.05, min(0.08, base * 0.5))
        return cov >= floor

    if min_height_px is not None and min_height_px > 0:
        eff_min_h = float(min_height_px)
        if currently_locked and sticky_target is not None:
            # ADS-close silhouettes can be shorter than profile min_h after
            # bottom-band anchor; do not drop in-ring retarget challengers.
            eff_min_h = min(
                eff_min_h,
                max(26.0, float(sticky_target.bbox_h) * 0.82),
            )
        before = len(candidates)
        candidates = [t for t in candidates if t.bbox_h >= eff_min_h]
        if len(candidates) < before:
            dbg.append(
                f"min_height filter: {before} -> {len(candidates)} (min_h={eff_min_h:.0f})"
            )
    before_clutter = len(candidates)
    candidates = [t for t in candidates if not target_is_background_clutter(t)]
    if len(candidates) < before_clutter:
        dbg.append(
            f"background_clutter filter: {before_clutter} -> {len(candidates)}"
        )
    before_board = len(candidates)

    def _live_motion(t: Target) -> float:
        if context is None:
            return 0.0
        return context.motion_coverage_ratio(t.bbox_x, t.bbox_y, t.bbox_w, t.bbox_h)

    candidates = [
        t
        for t in candidates
        if not target_is_range_board_fp(
            t,
            motion_overlap=_live_motion(t),
            frame_w=w,
            frame_h=h,
            fov_cx=float(cx),
        )
    ]
    if len(candidates) < before_board:
        dbg.append(f"range_board filter: {before_board} -> {len(candidates)}")
    before_banner = len(candidates)
    candidates = [
        t
        for t in candidates
        if not target_is_central_tower_banner_fp(
            t,
            frame_w=w,
            frame_h=h,
            fov_cx=float(cx),
            motion_overlap=_live_motion(t),
        )
    ]
    if len(candidates) < before_banner:
        dbg.append(f"tower_banner filter: {before_banner} -> {len(candidates)}")
    before_skyline = len(candidates)
    candidates = [
        t
        for t in candidates
        if not target_is_close_skyline_structure_fp(t, frame_w=w, frame_h=h)
    ]
    if len(candidates) < before_skyline:
        dbg.append(f"skyline_structure filter: {before_skyline} -> {len(candidates)}")
    before_env = len(candidates)
    candidates = [
        t
        for t in candidates
        if not target_is_environment_column(
            t, center_y=float(cy), frame_w=w, frame_h=h
        )
    ]
    if len(candidates) < before_env:
        dbg.append(f"environment_column filter: {before_env} -> {len(candidates)}")
    if not candidates:
        # When locked on a plausible target, preserve the lock geometry even
        # when all post-filter candidates dropped out.  This is the symmetric
        # path to the existing PHASE-6 sticky_pool_hold (which is unreachable
        # when candidates=[] because the code never enters the sticky block).
        # Skip hold for tall-high-close FPs (top_frac<0.34, bbox_h>20% frame)
        # that should be purged by apply_target_lock's close_upper stale gate.
        _sph_top_frac = float(sticky_target.bbox_y) / float(h) if (currently_locked and sticky_target is not None and h > 0) else 1.0
        _sph_high_close = (
            currently_locked and sticky_target is not None
            and _sph_top_frac < 0.34
            and float(sticky_target.bbox_h) > float(h) * 0.20
        )
        if currently_locked and sticky_target is not None and not _sph_high_close:
            # Hard evidence gate: if the locked bbox has no red mass in
            # the CURRENT frame, the enemy is gone (panned off / walked
            # offscreen) and holding the lock geometry as active=True
            # produces the gif_166_proof F39-F46 ghost class — green
            # box + red dot sitting on empty firing-range floor. Skip
            # the hold entirely when the bbox has < ~5% red coverage.
            if not _sticky_has_red_evidence(sticky_target):
                dbg.append(
                    f"sticky_pool_drop_no_red dist={sticky_target.distance_to_center:.0f} "
                    f"h={sticky_target.bbox_h}"
                )
                return DetectionResult(None, 0, 0.0, debug_lines=dbg, active=False)
            _s_mov = (
                context.motion_coverage_ratio(
                    sticky_target.bbox_x, sticky_target.bbox_y,
                    sticky_target.bbox_w, sticky_target.bbox_h,
                )
                if context is not None
                else 0.0
            )
            _hold = (
                sticky_target.body_shape_score >= 0.55
                and sticky_target.bbox_h >= 26
                and not target_is_close_skyline_structure_fp(
                    sticky_target, frame_w=w, frame_h=h
                )
                and not target_is_viewmodel_column_fp(
                    sticky_target,
                    frame_w=w,
                    frame_h=h,
                    fov_cx=float(cx),
                    fov_cy=float(cy),
                )
                and not target_is_central_tower_banner_fp(
                    sticky_target,
                    frame_w=w,
                    frame_h=h,
                    fov_cx=float(cx),
                    motion_overlap=_s_mov,
                )
            )
            dbg.append(
                f"sticky_pool_hold dist={sticky_target.distance_to_center:.0f} "
                f"h={sticky_target.bbox_h}"
            )
            return DetectionResult(
                sticky_target,
                0,
                sticky_target.confidence,
                debug_lines=dbg,
                active=_hold,
            )
        return DetectionResult(None, 0, 0.0, debug_lines=dbg, active=False)

    before_vm = len(candidates)
    candidates = [
        t
        for t in candidates
        if not target_is_viewmodel_column_fp(
            t, frame_w=w, frame_h=h, fov_cx=float(cx), fov_cy=float(cy)
        )
    ]
    if len(candidates) < before_vm:
        dbg.append(f"viewmodel_column filter: {before_vm} -> {len(candidates)}")
    if not candidates:
        _sph_top_frac2 = float(sticky_target.bbox_y) / float(h) if (currently_locked and sticky_target is not None and h > 0) else 1.0
        _sph_high_close2 = (
            currently_locked and sticky_target is not None
            and _sph_top_frac2 < 0.34
            and float(sticky_target.bbox_h) > float(h) * 0.20
        )
        if currently_locked and sticky_target is not None and not _sph_high_close2:
            if not _sticky_has_red_evidence(sticky_target):
                dbg.append(
                    f"sticky_pool_drop_no_red dist={sticky_target.distance_to_center:.0f} "
                    f"h={sticky_target.bbox_h}"
                )
                return DetectionResult(None, 0, 0.0, debug_lines=dbg, active=False)
            _s_mov2 = (
                context.motion_coverage_ratio(
                    sticky_target.bbox_x, sticky_target.bbox_y,
                    sticky_target.bbox_w, sticky_target.bbox_h,
                )
                if context is not None
                else 0.0
            )
            _hold2 = (
                sticky_target.body_shape_score >= 0.55
                and sticky_target.bbox_h >= 26
                and not target_is_close_skyline_structure_fp(
                    sticky_target, frame_w=w, frame_h=h
                )
                and not target_is_viewmodel_column_fp(
                    sticky_target,
                    frame_w=w,
                    frame_h=h,
                    fov_cx=float(cx),
                    fov_cy=float(cy),
                )
                and not target_is_central_tower_banner_fp(
                    sticky_target,
                    frame_w=w,
                    frame_h=h,
                    fov_cx=float(cx),
                    motion_overlap=_s_mov2,
                )
            )
            dbg.append(
                f"sticky_pool_hold dist={sticky_target.distance_to_center:.0f} "
                f"h={sticky_target.bbox_h}"
            )
            return DetectionResult(
                sticky_target,
                0,
                sticky_target.confidence,
                debug_lines=dbg,
                active=_hold2,
            )
        return DetectionResult(None, 0, 0.0, debug_lines=dbg, active=False)

    pool_max_h = max(int(t.bbox_h) for t in candidates)

    def _motion_overlap(t: Target) -> float:
        if context is None:
            return 0.0
        live = context.motion_coverage_ratio(t.bbox_x, t.bbox_y, t.bbox_w, t.bbox_h)
        # Do not let motion memory inflate ranking for sub-threshold red FPs
        # (collection already used live ratio; memory-only boost caused img7 wins).
        if float(t.red_coverage) < MIN_ENEMY_RED_COVERAGE:
            return live
        return context.motion_coverage_with_memory(t.bbox_x, t.bbox_y, t.bbox_w, t.bbox_h)

    def rank(t: Target) -> float:
        # Ranking includes closeness so a clearly larger (closer) enemy beats a
        # similarly-positioned smaller one. The bonus is NOT applied to the
        # confidence calculation — confidence must measure detection quality,
        # not target proximity, otherwise small far enemies would never make
        # the threshold and large but low-quality blobs would always pass.
        raw = score_target(
            t,
            float(fov_radius),
            distance_weight,
            area_weight,
            center_y=cy,
            motion_overlap=_motion_overlap(t),
            include_closeness=True,
        )
        if pool_max_h > 0:
            h_ratio = float(t.bbox_h) / float(pool_max_h)
            if h_ratio < 0.82:
                raw -= float(fov_radius) * 0.78 * (0.82 - h_ratio)
            if (
                h_ratio < 0.72
                and t.distance_to_center < float(fov_radius) * 0.42
                and int(t.part_count) <= 2
            ):
                raw -= float(fov_radius) * 0.55
            if (
                h > 0
                and t.bbox_y < h * 0.40
                and t.distance_to_center < float(fov_radius) * 0.55
            ):
                high_frac = max(0.0, 0.40 - float(t.bbox_y) / float(h))
                raw -= float(fov_radius) * high_frac * 2.8
        return raw

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
        if float(t.red_coverage) < MIN_ENEMY_RED_COVERAGE:
            return
        if t.body_shape_score < 0.55:
            return
        live = context.motion_coverage_ratio(t.bbox_x, t.bbox_y, t.bbox_w, t.bbox_h)
        if live >= context.motion_validate_threshold:
            context.note_motion_validated(t.bbox_x, t.bbox_y, t.bbox_w, t.bbox_h)

    if sticky_target is not None and stickiness_pixels > 0:
        pool = []
        for t in candidates:
            if target_is_background_clutter(t):
                continue
            if target_is_range_board_fp(
                t,
                motion_overlap=_live_motion(t),
                frame_w=w,
                frame_h=h,
                fov_cx=float(cx),
            ):
                continue
            if target_is_central_tower_banner_fp(
                t,
                frame_w=w,
                frame_h=h,
                fov_cx=float(cx),
                motion_overlap=_live_motion(t),
            ):
                continue
            if currently_locked and is_upward_fragment_vs_locked(sticky_target, t):
                continue
            iou = _bbox_iou(
                sticky_target.bbox_x, sticky_target.bbox_y, sticky_target.bbox_w, sticky_target.bbox_h,
                t.bbox_x, t.bbox_y, t.bbox_w, t.bbox_h,
            )
            dist = math.hypot(t.centroid_x - sticky_target.centroid_x, t.centroid_y - sticky_target.centroid_y)
            min_pool_body = 0.50 if currently_locked else 0.45
            min_pool_iou = 0.26 if currently_locked else 0.20
            sticky_dist = float(sticky_target.distance_to_center)
            near_lock = (
                iou >= min_pool_iou
                or (
                    dist <= stickiness_pixels
                    and t.body_shape_score >= min_pool_body
                    and (
                        not currently_locked
                        or t.distance_to_center <= sticky_dist * 1.18
                    )
                )
            )
            if near_lock:
                pool.append(t)
        if (
            currently_locked
            and sticky_target.bbox_h <= 56
            and sticky_target.distance_to_center > ring_fov * 0.45
        ):
            for t in candidates:
                if t in pool:
                    continue
                if (
                    t.bbox_h >= max(64, int(sticky_target.bbox_h * 1.35))
                    and t.distance_to_center < sticky_target.distance_to_center * 0.62
                    and t.body_shape_score >= 0.52
                    and not target_is_background_clutter(t)
                ):
                    pool.append(t)
        if pool and currently_locked:
            overlap_pool = [
                t
                for t in pool
                if _bbox_iou(
                    sticky_target.bbox_x,
                    sticky_target.bbox_y,
                    sticky_target.bbox_w,
                    sticky_target.bbox_h,
                    t.bbox_x,
                    t.bbox_y,
                    t.bbox_w,
                    t.bbox_h,
                )
                >= 0.12
                or t.distance_to_center <= sticky_target.distance_to_center * 1.15
            ]
            if not overlap_pool:
                dbg.append("sticky pool drop (no lock overlap)")
                pool = []
        if pool:
            sticky_best = max(pool, key=rank)
            global_best = max(candidates, key=rank)
            closer_challengers = [
                t
                for t in candidates
                if _should_retarget_closer_humanoid(
                    sticky_target,
                    t,
                    detect_fov=float(fov_radius),
                    display_fov=ring_fov,
                    fov_cx=cx,
                    fov_cy=cy,
                    frame_w=w,
                    frame_h=h,
                )
            ]
            if currently_locked and closer_challengers:
                chosen = finalize(max(closer_challengers, key=rank))
                if not target_is_background_clutter(chosen):
                    dbg.append(
                        f"closer_retarget_pool dist={chosen.distance_to_center:.0f} "
                        f"h={chosen.bbox_h} was={sticky_target.distance_to_center:.0f}"
                    )
                    _refresh_validation(chosen)
                    return DetectionResult(
                        chosen,
                        len(candidates),
                        chosen.confidence,
                        debug_lines=dbg,
                        active=True,
                    )
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
            # Frame lock (target_lock.apply_target_lock) owns enemy switches
            # during grace — do not let detection steal the lock in one frame.
            if switch_allowed and not currently_locked:
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
            if target_is_background_clutter(chosen):
                dbg.append(
                    f"sticky reject {RejectReason.BACKGROUND_CLUTTER.value} "
                    f"red={chosen.red_coverage:.3f} h={chosen.bbox_h}"
                )
                return DetectionResult(None, len(candidates), chosen.confidence, debug_lines=dbg, active=False)
            if (
                bbox_mid_in_sky_band(chosen.bbox_y, chosen.bbox_h, cy)
                and chosen.body_shape_score < 0.70
            ):
                chosen_mid_y = chosen.bbox_y + chosen.bbox_h * 0.5
                dbg.append(
                    f"sticky reject sky_band mid_y={chosen_mid_y:.0f} cy={cy:.0f} "
                    f"body={chosen.body_shape_score:.2f}"
                )
                return DetectionResult(None, len(candidates), chosen.confidence, debug_lines=dbg, active=False)
            iou_lock = _bbox_iou(
                sticky_target.bbox_x, sticky_target.bbox_y,
                sticky_target.bbox_w, sticky_target.bbox_h,
                chosen.bbox_x, chosen.bbox_y, chosen.bbox_w, chosen.bbox_h,
            )
            lock_dist = math.hypot(
                chosen.centroid_x - sticky_target.centroid_x,
                chosen.centroid_y - sticky_target.centroid_y,
            )
            lock_overlap = currently_locked and (iou_lock >= 0.5 or lock_dist < 40.0)
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
        if currently_locked and sticky_target is not None:
            probe, probe_dbg = _collect_candidates(
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
                debug=False,
                detection_mode=resolved_mode,
                context=None,
                min_aspect=min_aspect,
                max_aspect=max_aspect,
                min_solidity=min_solidity,
            )
            if probe_dbg:
                dbg.extend(probe_dbg[:2])
            seen = {
                (t.bbox_x, t.bbox_y, t.bbox_w, t.bbox_h) for t in candidates
            }
            for t in probe:
                key = (t.bbox_x, t.bbox_y, t.bbox_w, t.bbox_h)
                if key not in seen:
                    candidates.append(t)
                    seen.add(key)
            before_clutter = len(candidates)
            candidates = [t for t in candidates if not target_is_background_clutter(t)]
            if len(candidates) < before_clutter:
                dbg.append(
                    f"rim_retarget_clutter: {before_clutter} -> {len(candidates)}"
                )
            if currently_locked and candidates:
                lost_challengers = [
                    t
                    for t in candidates
                    if _should_retarget_closer_humanoid(
                        sticky_target,
                        t,
                        detect_fov=float(fov_radius),
                        display_fov=ring_fov,
                        fov_cx=cx,
                        fov_cy=cy,
                        frame_w=w,
                        frame_h=h,
                    )
                ]
                if lost_challengers:
                    chosen = finalize(max(lost_challengers, key=rank))
                    if not target_is_background_clutter(chosen):
                        dbg.append(
                            f"closer_retarget dist={chosen.distance_to_center:.0f} "
                            f"h={chosen.bbox_h} was={sticky_target.distance_to_center:.0f}"
                        )
                        _refresh_validation(chosen)
                        return DetectionResult(
                            chosen,
                            len(candidates),
                            chosen.confidence,
                            debug_lines=dbg,
                            active=True,
                        )
                if sticky_target.distance_to_center < ring_fov * 0.72:
                    in_ring = [
                        t
                        for t in candidates
                        if t.distance_to_center < ring_fov * 0.55
                        and t.body_shape_score >= 0.50
                        and not target_is_background_clutter(t)
                        and not target_is_close_skyline_structure_fp(
                            t, frame_w=w, frame_h=h
                        )
                        and not target_is_viewmodel_column_fp(
                            t,
                            frame_w=w,
                            frame_h=h,
                            fov_cx=float(cx),
                            fov_cy=float(cy),
                        )
                    ]
                    if in_ring:
                        chosen = finalize(
                            min(in_ring, key=lambda t: t.distance_to_center)
                        )
                        dbg.append(
                            f"in_ring_nearest dist={chosen.distance_to_center:.0f} "
                            f"h={chosen.bbox_h}"
                        )
                        _refresh_validation(chosen)
                        return DetectionResult(
                            chosen,
                            len(candidates),
                            chosen.confidence,
                            debug_lines=dbg,
                            active=True,
                        )
            identity = [
                t
                for t in candidates
                if _bbox_iou(
                    sticky_target.bbox_x,
                    sticky_target.bbox_y,
                    sticky_target.bbox_w,
                    sticky_target.bbox_h,
                    t.bbox_x,
                    t.bbox_y,
                    t.bbox_w,
                    t.bbox_h,
                )
                >= 0.12
                or math.hypot(
                    t.centroid_x - sticky_target.centroid_x,
                    t.centroid_y - sticky_target.centroid_y,
                )
                < max(55.0, float(sticky_target.bbox_h) * 0.85)
            ]
            if identity:
                chosen = finalize(max(identity, key=rank))
                if (
                    currently_locked
                    and chosen.distance_to_center
                    > sticky_target.distance_to_center * 1.22
                ):
                    dbg.append(
                        f"sticky_identity_skip_far dist={chosen.distance_to_center:.0f} "
                        f"locked={sticky_target.distance_to_center:.0f}"
                    )
                elif not target_is_background_clutter(chosen):
                    dbg.append(
                        f"sticky_identity dist={chosen.distance_to_center:.0f} "
                        f"h={chosen.bbox_h}"
                    )
                    _refresh_validation(chosen)
                    return DetectionResult(
                        chosen,
                        len(candidates),
                        chosen.confidence,
                        debug_lines=dbg,
                        active=True,
                    )
        # PHASE-6 (D-HIGH6): empty sticky pool while locked — hold last lock
        # geometry (do not return None or apply_target_lock drops the lock).
        if currently_locked and sticky_target is not None:
            if not _sticky_has_red_evidence(sticky_target):
                dbg.append(
                    f"sticky_pool_drop_no_red dist={sticky_target.distance_to_center:.0f} "
                    f"h={sticky_target.bbox_h}"
                )
                return DetectionResult(
                    None, len(candidates), 0.0, debug_lines=dbg, active=False
                )
            hold_active = (
                sticky_target.body_shape_score >= 0.55
                and sticky_target.bbox_h >= 26
                and not target_is_close_skyline_structure_fp(
                    sticky_target, frame_w=w, frame_h=h
                )
                and not target_is_viewmodel_column_fp(
                    sticky_target,
                    frame_w=w,
                    frame_h=h,
                    fov_cx=float(cx),
                    fov_cy=float(cy),
                )
                and not target_is_central_tower_banner_fp(
                    sticky_target,
                    frame_w=w,
                    frame_h=h,
                    fov_cx=float(cx),
                    motion_overlap=_motion_overlap(sticky_target),
                )
            )
            dbg.append(
                f"sticky_pool_hold dist={sticky_target.distance_to_center:.0f} "
                f"h={sticky_target.bbox_h}"
            )
            return DetectionResult(
                sticky_target,
                len(candidates),
                sticky_target.confidence,
                debug_lines=dbg,
                active=hold_active,
            )

    best = finalize(max(candidates, key=rank))
    if target_is_background_clutter(best):
        dbg.append(
            f"free-max reject {RejectReason.BACKGROUND_CLUTTER.value} "
            f"red={best.red_coverage:.3f} h={best.bbox_h}"
        )
        return DetectionResult(None, len(candidates), best.confidence, debug_lines=dbg, active=False)
    if best.confidence < min_confidence:
        dbg.append(f"selected reject low_conf={best.confidence:.2f}")
        return DetectionResult(None, len(candidates), best.confidence, debug_lines=dbg, active=False)
    if bbox_mid_in_sky_band(best.bbox_y, best.bbox_h, cy) and best.body_shape_score < 0.70:
        best_mid_y = best.bbox_y + best.bbox_h * 0.5
        dbg.append(f"free-max reject sky_band mid_y={best_mid_y:.0f} cy={cy:.0f} body={best.body_shape_score:.2f}")
        return DetectionResult(None, len(candidates), best.confidence, debug_lines=dbg, active=False)
    _refresh_validation(best)
    dbg.append(
        f"SELECTED body={best.body_shape_score:.2f} head={best.head_score:.2f} "
        f"dist={best.distance_to_center:.0f} bbox={best.bbox_w}x{best.bbox_h}"
    )
    return DetectionResult(best, len(candidates), best.confidence, debug_lines=dbg, active=True)


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
    detection_mode: str | None = None,
    exclude_bottom_frac: float = _VIEWMODEL_EXCLUDE_FRAC,
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
    # PHASE-7 AUDIT FIX (MED10): use the LIVE detection mode (passed
    # from the runtime) so the green-tinted mask in the debug window
    # reflects what the runtime is actually doing. The previous
    # hardcoded ``DETECTION_MODE_SHAPE`` made the apex/hybrid masks
    # invisible in the debug viewer, misleading users debugging Apex
    # detection by showing only the shape channel.
    dbg_mode = detection_mode if detection_mode is not None else DETECTION_MODE_SHAPE
    mask = build_detection_mask(frame_bgr, hsv_ranges, detection_mode=dbg_mode)
    mask_fov = (
        int(display_fov_radius)
        if display_fov_radius is not None
        else int(fov_radius)
    )
    fov = _build_fov_mask(h, w, cx_f, cy_f, mask_fov)
    vm = _build_viewmodel_exclude_mask(h, w, exclude_bottom_frac)
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
