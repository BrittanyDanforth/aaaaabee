"""Sticky pool must not adopt upward fragments over a full-body lock."""

from __future__ import annotations

from dataclasses import replace

import detector
from detector import Target, is_upward_fragment_vs_locked


def _body() -> Target:
    return Target(
        centroid_x=435.0,
        centroid_y=290.0,
        area=10500.0,
        distance_to_center=50.0,
        confidence=0.9,
        bbox_x=400,
        bbox_y=220,
        bbox_w=70,
        bbox_h=150,
        body_shape_score=0.85,
        head_score=0.7,
        torso_score=0.65,
        limb_stack_score=0.5,
        part_count=8,
        red_coverage=0.12,
        reject_reason="ok",
    )


def test_is_upward_fragment_vs_locked() -> None:
    locked = _body()
    frag = replace(
        locked,
        bbox_y=180,
        bbox_h=60,
        centroid_y=210.0,
        torso_score=0.1,
        body_shape_score=0.55,
    )
    assert is_upward_fragment_vs_locked(locked, frag)


def test_sticky_pool_skips_fragment_when_locked() -> None:
    locked = _body()
    frag = replace(
        locked,
        bbox_y=190,
        bbox_h=55,
        centroid_x=438.0,
        centroid_y=215.0,
        torso_score=0.08,
        body_shape_score=0.50,
        red_coverage=0.04,
    )
    pool = [locked, frag]
    filtered = [
        t
        for t in pool
        if not detector.is_upward_fragment_vs_locked(locked, t)
    ]
    assert filtered == [locked]
