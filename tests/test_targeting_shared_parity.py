"""Shared targeting helpers match production math."""

from __future__ import annotations

import math

from targeting_shared import ring_clamp_distance_limit, ring_clamp_frame_point


def test_ring_clamp_uses_min_detect_and_display_fov() -> None:
    detect, display = 208.0, 185.0
    lim = ring_clamp_distance_limit(detect, display)
    assert math.isclose(lim, min(detect, display) * 0.96)
    cx, cy = 100.0, 100.0
    ox, oy = 100.0 + 300.0, 100.0
    px, py = ring_clamp_frame_point(
        ox, oy, cx, cy, detect_fov=detect, display_fov=display
    )
    assert math.hypot(px - cx, py - cy) <= lim + 0.01


def test_ring_clamp_inside_limit_unchanged() -> None:
    px, py = ring_clamp_frame_point(
        105.0, 100.0, 100.0, 100.0, detect_fov=200.0, display_fov=180.0
    )
    assert px == 105.0 and py == 100.0


def test_harness_must_not_use_pre_scaled_display_fov() -> None:
    """_runtime_overlay_fov is ring_inner (×0.96); clamp needs full overlay radius."""
    detect, display_inner = 208.0, 134.4
    full_display = 140.0
    wrong = ring_clamp_frame_point(
        300.0, 100.0, 100.0, 100.0, detect_fov=detect, display_fov=display_inner
    )
    right = ring_clamp_frame_point(
        300.0, 100.0, 100.0, 100.0, detect_fov=detect, display_fov=full_display
    )
    assert wrong != right
