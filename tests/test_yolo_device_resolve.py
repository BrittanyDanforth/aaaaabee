"""yolo_device auto/cuda must not break single-GPU load."""

from __future__ import annotations

from apexaimbot_bridge import resolve_yolo_device


def test_auto_maps_to_empty_for_select_device() -> None:
    assert resolve_yolo_device({"yolo_device": "auto"}) == ""
    assert resolve_yolo_device({"yolo_device": "cuda"}) == ""
    assert resolve_yolo_device({}) == ""


def test_cpu_and_gpu_index_preserved() -> None:
    assert resolve_yolo_device({"yolo_device": "cpu"}) == "cpu"
    assert resolve_yolo_device({"yolo_device": "0"}) == "0"
