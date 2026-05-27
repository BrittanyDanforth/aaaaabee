"""VALOAI-inspired tracking fusion helpers."""

from __future__ import annotations

from dataclasses import dataclass

from tracking_fusion import (
    ExternalBox,
    apply_external_box_fusion,
    associate_boxes_iou,
    bbox_iou,
    pick_nearest_crosshair,
    select_ranked_target,
)


@dataclass
class _FakeTarget:
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float
    body_shape_score: float = 0.8
    distance_to_center: float = 50.0


def test_bbox_iou_overlap() -> None:
    assert bbox_iou(0, 0, 10, 10, 5, 5, 10, 10) > 0.1
    assert bbox_iou(0, 0, 10, 10, 100, 100, 10, 10) == 0.0


def test_associate_boxes_iou() -> None:
    prev = [ExternalBox(0, 0, 20, 40, confidence=0.9)]
    curr = [ExternalBox(2, 2, 22, 42, confidence=0.85)]
    pairs = associate_boxes_iou(prev, curr, iou_threshold=0.3)
    assert len(pairs) == 1
    assert pairs[0][2] > 0.5


def test_fusion_boost_when_yolo_agrees() -> None:
    t = _FakeTarget(10, 10, 40, 80)
    boxes = [ExternalBox(12, 12, 50, 88, confidence=0.9)]
    boosts = apply_external_box_fusion([t], boxes, fov_radius=140.0, boost_scale=0.5)
    assert id(t) in boosts
    assert boosts[id(t)] > 10.0


def test_pick_nearest_crosshair() -> None:
    far = _FakeTarget(0, 0, 20, 60, distance_to_center=120.0)
    near = _FakeTarget(50, 50, 20, 60, distance_to_center=15.0)
    picked = pick_nearest_crosshair([far, near])
    assert picked is near


def test_select_ranked_target_valoai_mode() -> None:
    far = _FakeTarget(0, 0, 20, 60, body_shape_score=0.9, distance_to_center=120.0)
    near = _FakeTarget(50, 50, 20, 60, body_shape_score=0.55, distance_to_center=15.0)

    def rank(t: _FakeTarget) -> float:
        return t.body_shape_score * 100.0 - t.distance_to_center * 0.1

    apex_pick = select_ranked_target([far, near], rank, selection_mode="apex")
    assert apex_pick is far
    valo_pick = select_ranked_target([far, near], rank, selection_mode="valoai")
    assert valo_pick is near
