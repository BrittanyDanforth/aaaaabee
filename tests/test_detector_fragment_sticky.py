"""Fragment guard must be importable from detector (runtime startup)."""

from __future__ import annotations

from detector import Target, is_upward_fragment_vs_locked


def test_is_upward_fragment_vs_locked_importable() -> None:
    locked = Target(
        centroid_x=400.0,
        centroid_y=300.0,
        area=18000.0,
        bbox_x=360,
        bbox_y=180,
        bbox_w=90,
        bbox_h=200,
        confidence=0.9,
        body_shape_score=0.8,
        red_coverage=0.35,
        torso_score=0.75,
        limb_stack_score=0.7,
        part_count=4,
    )
    frag = Target(
        centroid_x=402.0,
        centroid_y=195.0,
        area=2750.0,
        bbox_x=375,
        bbox_y=120,
        bbox_w=50,
        bbox_h=55,
        confidence=0.88,
        body_shape_score=0.85,
        red_coverage=0.4,
        torso_score=0.8,
        limb_stack_score=0.5,
        part_count=2,
    )
    assert is_upward_fragment_vs_locked(locked, frag)


def test_runtime_import_chain() -> None:
    from runtime import AssistRuntime  # noqa: F401
    from target_lock import apply_target_lock  # noqa: F401
