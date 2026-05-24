"""img6: seven-character lineup must enumerate multiple per-character boxes."""

from __future__ import annotations

from pathlib import Path

import cv2
import pytest

import detector

INPUT = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "real_apex_test"
    / "_inputs"
    / "img6_seven_characters.png"
)


@pytest.mark.skipif(not INPUT.exists(), reason="real img6 fixture missing")
def test_img6_enumerates_multiple_characters_not_one_merge() -> None:
    img = cv2.imread(str(INPUT))
    assert img is not None
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    fov_r = int(0.95 * max(w, h) / 2.0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ctx = detector.DetectionContext()
    ctx.prev_gray = gray
    ctx.prev_size = (w, h)
    cands, _, _ = detector.enumerate_candidates(
        img,
        None,
        fov_r,
        max(40.0, 60.0 * (min(w, h) / 1080.0) ** 2),
        cx,
        cy,
        exclude_bottom_frac=0.05,
        detection_mode=detector.DETECTION_MODE_APEX,
        context=ctx,
    )
    accepted = [c for c in cands if c.accepted]
    assert len(accepted) >= 6
    assert sum(1 for c in accepted if c.bbox_w > w * 0.15) <= 2
    xs = sorted(c.bbox_x + c.bbox_w * 0.5 for c in accepted)
    assert xs[-1] - xs[0] >= 500
