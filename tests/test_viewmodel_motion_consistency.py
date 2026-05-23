"""Viewmodel + motion-ranking consistency regressions (img7 audit)."""

from __future__ import annotations

import detector


def _img7_gun_column() -> detector.Target:
    return detector.Target(
        centroid_x=535.5,
        centroid_y=369.0,
        area=3216.0,
        distance_to_center=61.0,
        confidence=0.85,
        bbox_x=485,
        bbox_y=269,
        bbox_w=106,
        bbox_h=246,
        body_shape_score=1.0,
        head_score=0.87,
        torso_score=0.64,
        limb_stack_score=1.0,
        part_count=14,
        red_coverage=0.011,
    )


def test_motion_memory_does_not_boost_low_red_ranking() -> None:
    """Ranking must not use motion_memory when red < MIN_ENEMY_RED_COVERAGE."""
    ctx = detector.DetectionContext(motion_assist=True)
    import numpy as np

    h, w = 736, 1193
    ctx.last_motion_mask = np.zeros((h, w), dtype=np.uint8)
    ctx.note_motion_validated(485, 269, 106, 246)
    t = _img7_gun_column()
    live = ctx.motion_coverage_ratio(t.bbox_x, t.bbox_y, t.bbox_w, t.bbox_h)
    mem = ctx.motion_coverage_with_memory(t.bbox_x, t.bbox_y, t.bbox_w, t.bbox_h)
    assert live < 0.08, "precondition: bbox has little live motion"
    assert mem >= 0.18, "precondition: memory inflates overlap for validated bbox"
    assert float(t.red_coverage) < detector.MIN_ENEMY_RED_COVERAGE


def test_fake_torso_high_score_still_viewmodel() -> None:
    t = _img7_gun_column()
    assert t.torso_score >= 0.32
    assert t.red_coverage < detector.MIN_ENEMY_RED_COVERAGE
    assert detector.target_is_viewmodel_column_fp(
        t, frame_w=1193, frame_h=736, fov_cx=1193 / 2, fov_cy=736 / 2
    )
