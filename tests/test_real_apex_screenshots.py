"""Hard-pinned regression suite using the 5 REAL Apex Legends screenshots.

These tests load the actual PNG/WEBP files the user supplied as evidence
that the previous "fixed" detector still missed bright red enemies. Each
image is the source of truth — synthetic frames lied, so these assertions
exist precisely to prevent that regression from ever happening again.

Each image must:

1. Produce ``active=True`` from :func:`detector.find_best_target`.
2. The selected target's bbox center must lie within ``fov_radius``
   of the FOV center.
3. The anchor (centroid_x, centroid_y) must lie INSIDE the bbox in the
   chest band — i.e. between 28 % and 52 % of the bbox height.

Any failure here means the detector has regressed on REAL gameplay
frames; no number of passing synthetic tests can save it.
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


# (filename, ads_hint). The audit harness uses identical FOV proportions
# so the runtime exercise here mirrors what the empirical audit shows.
_IMAGES: list[tuple[str, bool]] = [
    ("img1_back_view.png", False),
    ("img2_shooting_26.png", False),
    ("img3_side_view.webp", True),
    ("img4_dummy_not_detected.webp", False),
    ("img5_close_ads.webp", True),
]


def _fov_radius(frame_w: int, frame_h: int, ads: bool) -> int:
    base = min(frame_w, frame_h)
    if ads:
        return max(80, int(0.42 * base))
    return max(80, int(0.36 * base))


def _make_ctx(gray_now: np.ndarray, w: int, h: int) -> detector.DetectionContext:
    """Detection context with zero synthetic motion (still screenshot)."""
    ctx = detector.DetectionContext()
    ctx.prev_gray = gray_now.copy()
    ctx.prev_size = (w, h)
    return ctx


@pytest.mark.parametrize("filename, ads_hint", _IMAGES)
def test_real_apex_screenshot_detects_character(filename: str, ads_hint: bool) -> None:
    path = INPUTS_DIR / filename
    assert path.exists(), f"Real-image fixture missing: {path}"

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert img is not None, f"cv2 could not decode {path}"
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    fov_r = _fov_radius(w, h, ads_hint)
    min_area_floor = max(40.0, 60.0 * (min(w, h) / 1080.0) ** 2)

    gray_now = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Larger viewmodel exclusion because these are full-screen
    # screenshots that include the gun barrel / ammo HUD.
    exclude_bottom = 0.30

    result = detector.find_best_target(
        img,
        None,
        fov_r,
        min_area_floor,
        cx,
        cy,
        debug=False,
        exclude_bottom_frac=exclude_bottom,
        detection_mode=detector.DETECTION_MODE_APEX,
        context=_make_ctx(gray_now, w, h),
    )

    assert result.active, (
        f"{filename}: detector returned active=False — real image must "
        "produce a target"
    )
    target = result.target
    assert target is not None, f"{filename}: active=True but target is None"

    bx, by, bw, bh = target.bbox_x, target.bbox_y, target.bbox_w, target.bbox_h
    assert bw > 0 and bh > 0, f"{filename}: degenerate bbox"

    cxx, cyy = bx + bw / 2.0, by + bh / 2.0
    dist = math.hypot(cxx - cx, cyy - cy)
    assert dist <= fov_r + 2.0, (
        f"{filename}: bbox center ({cxx:.1f},{cyy:.1f}) outside FOV "
        f"radius {fov_r} (distance {dist:.1f})"
    )

    ax, ay = target.centroid_x, target.centroid_y
    assert bx <= ax <= bx + bw, (
        f"{filename}: anchor x={ax:.1f} outside bbox x=[{bx},{bx+bw}]"
    )
    assert by <= ay <= by + bh, (
        f"{filename}: anchor y={ay:.1f} outside bbox y=[{by},{by+bh}]"
    )

    y_frac = (ay - by) / float(max(1, bh))
    # Allow a tiny floating-point slack at the edges of the band.
    assert 0.27 <= y_frac <= 0.53, (
        f"{filename}: anchor y_frac={y_frac:.3f} not inside chest band "
        "[0.28, 0.52]"
    )

    assert target.body_shape_score >= 0.55, (
        f"{filename}: body_shape_score {target.body_shape_score:.2f} too "
        "low — analyzer no longer accepts a humanoid silhouette"
    )
