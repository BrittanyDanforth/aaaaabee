"""Phase-5 hard regression: all SIX real Apex Legends screenshots.

Loads img1-5 (single-character framings) plus img6 (7-character
lineup) and pins down the detector's empirical behaviour:

* For each of the 5 single-character images: ``find_best_target``
  returns ``active=True`` with the anchor inside the bbox chest
  band (28 % - 52 % of bbox height) and a body_shape_score >= 0.55.
* For img6 (the lineup): at least 3 candidates are accepted by
  ``enumerate_candidates`` at full-frame FOV, and the selected
  target's bbox center lies inside the FOV.

Synthetic frames cannot subvert these checks — the JPEG/PNG/WEBP
bytes on disk are the source of truth.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

import detector


REPO_ROOT = Path(__file__).resolve().parents[1]
INPUTS_DIR = REPO_ROOT / "artifacts" / "real_apex_test" / "_inputs"


_SINGLE_CHAR_IMAGES: list[tuple[str, bool]] = [
    ("img1_back_view.png", False),
    ("img2_shooting_26.png", False),
    ("img3_side_view.webp", True),
    ("img4_dummy_not_detected.webp", False),
    ("img5_close_ads.webp", True),
]

_LINEUP_IMAGE = "img6_seven_characters.png"


def _fov_radius_single(frame_w: int, frame_h: int, ads: bool) -> int:
    base = min(frame_w, frame_h)
    if ads:
        return max(80, int(0.42 * base))
    return max(80, int(0.36 * base))


def _fov_radius_lineup(frame_w: int, frame_h: int) -> int:
    """Wide FOV for the lineup so all seven characters enumerate."""
    base = max(frame_w, frame_h)
    return int(0.95 * base / 2.0)


def _make_ctx(gray_now: np.ndarray, w: int, h: int) -> detector.DetectionContext:
    ctx = detector.DetectionContext()
    ctx.prev_gray = gray_now.copy()
    ctx.prev_size = (w, h)
    return ctx


@pytest.mark.parametrize("filename, ads_hint", _SINGLE_CHAR_IMAGES)
def test_single_character_image_detects_with_chest_anchor(
    filename: str, ads_hint: bool
) -> None:
    path = INPUTS_DIR / filename
    assert path.exists(), f"Real-image fixture missing: {path}"

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert img is not None, f"cv2 could not decode {path}"
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    fov_r = _fov_radius_single(w, h, ads_hint)
    min_area_floor = max(40.0, 60.0 * (min(w, h) / 1080.0) ** 2)

    gray_now = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    result = detector.find_best_target(
        img,
        None,
        fov_r,
        min_area_floor,
        cx,
        cy,
        debug=False,
        exclude_bottom_frac=0.30,
        detection_mode=detector.DETECTION_MODE_APEX,
        context=_make_ctx(gray_now, w, h),
    )

    assert result.active, f"{filename}: detector active=False on real screenshot"
    target = result.target
    assert target is not None, f"{filename}: active=True but target is None"

    bx, by, bw, bh = target.bbox_x, target.bbox_y, target.bbox_w, target.bbox_h
    assert bw > 0 and bh > 0, f"{filename}: degenerate bbox"

    cxx, cyy = bx + bw / 2.0, by + bh / 2.0
    dist = math.hypot(cxx - cx, cyy - cy)
    assert dist <= fov_r + 2.0, (
        f"{filename}: bbox center ({cxx:.1f},{cyy:.1f}) outside FOV r={fov_r}"
    )

    ax, ay = target.centroid_x, target.centroid_y
    assert bx <= ax <= bx + bw, (
        f"{filename}: anchor x={ax:.1f} outside bbox [{bx},{bx+bw}]"
    )
    assert by <= ay <= by + bh, (
        f"{filename}: anchor y={ay:.1f} outside bbox [{by},{by+bh}]"
    )

    y_frac = (ay - by) / float(max(1, bh))
    assert 0.27 <= y_frac <= 0.53, (
        f"{filename}: anchor y_frac={y_frac:.3f} not in chest band [0.28,0.52]"
    )

    assert target.body_shape_score >= 0.55, (
        f"{filename}: body_shape_score {target.body_shape_score:.2f} too low"
    )


def test_seven_character_lineup_produces_three_plus_candidates() -> None:
    path = INPUTS_DIR / _LINEUP_IMAGE
    assert path.exists(), f"Lineup fixture missing: {path}"

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert img is not None, f"cv2 could not decode {path}"
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    fov_r = _fov_radius_lineup(w, h)
    min_area_floor = max(40.0, 60.0 * (min(w, h) / 1080.0) ** 2)

    gray_now = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ctx = _make_ctx(gray_now, w, h)

    cands, _mask, _parts = detector.enumerate_candidates(
        img,
        None,
        fov_r,
        min_area_floor,
        cx,
        cy,
        exclude_bottom_frac=0.05,
        detection_mode=detector.DETECTION_MODE_APEX,
        context=ctx,
    )

    accepted = [c for c in cands if c.accepted]
    assert len(accepted) >= 3, (
        f"{_LINEUP_IMAGE}: only {len(accepted)} accepted candidates "
        f"(expected >= 3 of 7 characters). All candidates: "
        f"{[(c.idx, c.accepted, c.reject_reason) for c in cands]}"
    )

    accepted_x_centers = sorted(
        c.bbox_x + c.bbox_w / 2.0 for c in accepted
    )
    assert len(accepted_x_centers) >= 3
    span = accepted_x_centers[-1] - accepted_x_centers[0]
    assert span >= 200, (
        f"{_LINEUP_IMAGE}: accepted candidates span only {span:.0f} px; "
        "lineup detection should cover characters across the frame"
    )

    result = detector.find_best_target(
        img,
        None,
        fov_r,
        min_area_floor,
        cx,
        cy,
        debug=False,
        exclude_bottom_frac=0.05,
        detection_mode=detector.DETECTION_MODE_APEX,
        context=_make_ctx(gray_now, w, h),
    )
    assert result.active, f"{_LINEUP_IMAGE}: detector active=False"
    target = result.target
    assert target is not None
    sx = target.bbox_x + target.bbox_w / 2.0
    sy = target.bbox_y + target.bbox_h / 2.0
    dist = math.hypot(sx - cx, sy - cy)
    assert dist <= fov_r + 2.0, (
        f"{_LINEUP_IMAGE}: selected bbox center ({sx:.1f},{sy:.1f}) "
        f"outside FOV r={fov_r}"
    )

    accepted_bboxes = [
        (c.bbox_x, c.bbox_y, c.bbox_w, c.bbox_h) for c in accepted
    ]
    matched = any(
        (
            target.bbox_x >= bx - 4
            and target.bbox_y >= by - 4
            and target.bbox_x + target.bbox_w <= bx + bw + 8
            and target.bbox_y + target.bbox_h <= by + bh + 8
        )
        for bx, by, bw, bh in accepted_bboxes
    )
    assert matched, (
        f"{_LINEUP_IMAGE}: selected bbox "
        f"{(target.bbox_x, target.bbox_y, target.bbox_w, target.bbox_h)} "
        f"does not match any accepted candidate {accepted_bboxes}"
    )
