"""IMG2: lock must be on crosshair enemy, not distant sand motion FP."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import pytest

import detector

INPUT = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "real_apex_test"
    / "_inputs"
    / "img2_shooting_26.png"
)


@pytest.mark.skipif(not INPUT.exists(), reason="real img2 fixture missing")
def test_img2_target_near_crosshair_not_rim_sand() -> None:
    img = cv2.imread(str(INPUT))
    assert img is not None
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    fov_r = max(80, int(0.36 * min(w, h)))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ctx = detector.DetectionContext()
    ctx.prev_gray = gray
    ctx.prev_size = (w, h)
    r = detector.find_best_target(
        img,
        None,
        fov_r,
        max(40.0, 60.0 * (min(w, h) / 1080.0) ** 2),
        cx,
        cy,
        detection_mode=detector.DETECTION_MODE_APEX,
        context=ctx,
        exclude_bottom_frac=0.30,
    )
    assert r.active and r.target is not None
    t = r.target
    dist = math.hypot(t.centroid_x - cx, t.centroid_y - cy)
    assert dist < fov_r * 0.45, (
        f"img2 centroid must stay near crosshair (dist={dist:.0f}, fov={fov_r})"
    )
    assert t.bbox_x < w * 0.55, (
        f"img2 bbox must not be rim sand blob (bbox_x={t.bbox_x})"
    )
