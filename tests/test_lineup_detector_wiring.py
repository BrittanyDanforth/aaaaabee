"""Lineup-wide FOV path: enumerate + analyze_figure + refine stay connected."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

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
def test_lineup_wide_fov_enables_display_refine() -> None:
    """img6 FOV must set lineup_display so overlay uses _refine_lineup_display_bbox."""
    img = cv2.imread(str(INPUT))
    assert img is not None
    h, w = img.shape[:2]
    fov_r = int(0.95 * max(w, h) / 2.0)
    assert detector._is_lineup_wide_fov(float(fov_r), w, h)
    refine_calls: list[bool] = []
    orig_refine = detector._refine_lineup_display_bbox

    def _track_refine(*args, **kwargs):
        refine_calls.append(True)
        return orig_refine(*args, **kwargs)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ctx = detector.DetectionContext()
    ctx.prev_gray = gray
    ctx.prev_size = (w, h)
    with patch.object(
        detector, "_refine_lineup_display_bbox", side_effect=_track_refine
    ):
        cands, _, _ = detector.enumerate_candidates(
            img,
            None,
            fov_r,
            max(40.0, 60.0 * (min(w, h) / 1080.0) ** 2),
            w / 2.0,
            h / 2.0,
            exclude_bottom_frac=0.05,
            detection_mode=detector.DETECTION_MODE_APEX,
            context=ctx,
        )
    assert refine_calls, "lineup_display refine never invoked"
    accepted = [c for c in cands if c.accepted]
    assert 6 <= len(accepted) <= 8


@pytest.mark.skipif(not INPUT.exists(), reason="real img6 fixture missing")
def test_enumerate_and_collect_share_lineup_wide_reject() -> None:
    """Both entry points must agree on lineup_wide_fov for the same frame/FOV."""
    img = cv2.imread(str(INPUT))
    assert img is not None
    h, w = img.shape[:2]
    fov_r = int(0.95 * max(w, h) / 2.0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ctx = detector.DetectionContext()
    ctx.prev_gray = gray
    ctx.prev_size = (w, h)
    min_area = max(40.0, 60.0 * (min(w, h) / 1080.0) ** 2)
    cx, cy = w / 2.0, h / 2.0
    enum_cands, _, _ = detector.enumerate_candidates(
        img,
        None,
        fov_r,
        min_area,
        cx,
        cy,
        exclude_bottom_frac=0.05,
        detection_mode=detector.DETECTION_MODE_APEX,
        context=ctx,
    )
    with patch.object(detector, "_prune_lineup_candidates") as mock_prune:
        detector.find_best_target(
            img,
            None,
            fov_r,
            min_area,
            cx,
            cy,
            exclude_bottom_frac=0.05,
            detection_mode=detector.DETECTION_MODE_APEX,
            context=ctx,
        )
    assert not mock_prune.called, "find_best_target must not prune (enumerate-only)"
    octane = sorted([c for c in enum_cands if c.accepted], key=lambda c: c.bbox_x)[5]
    slot_cx = w * detector._LINEUP_SLOT_X_FRACS[5]
    oct_cx = octane.bbox_x + octane.bbox_w * 0.5
    assert abs(oct_cx - slot_cx) <= w * 0.015
