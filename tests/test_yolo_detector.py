"""YOLO-primary detection path (mocked engine — no torch required in CI)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from detector import DETECTION_MODE_YOLO, find_best_target
from yolo_detector import YoloDetection, YoloEngine, YoloEngineConfig, reset_yolo_engine_cache


def _mock_engine(dets: list[YoloDetection]) -> MagicMock:
    eng = MagicMock(spec=YoloEngine)
    eng.infer.return_value = dets
    eng.config = YoloEngineConfig(
        weights_path=__import__("pathlib").Path("/tmp/fake.pt"),
        target_pick="nearest",
    )
    eng.find_best_target = YoloEngine.find_best_target.__get__(eng, YoloEngine)
    return eng


def test_find_best_target_yolo_delegates_without_cv() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cx, cy = 320.0, 240.0
    dets = [
        YoloDetection(280.0, 100.0, 360.0, 380.0, 0.88, "target"),
    ]
    eng = _mock_engine(dets)
    r = find_best_target(
        frame,
        [],
        250,
        100.0,
        cx,
        cy,
        detection_mode=DETECTION_MODE_YOLO,
        yolo_engine=eng,
    )
    assert r.active
    assert r.target is not None
    assert r.target.confidence == pytest.approx(0.88, rel=1e-3)
    assert "mode=yolo" in (r.debug_lines[0] if r.debug_lines else "")
    eng.infer.assert_called_once()


def test_find_best_target_yolo_missing_engine_inactive() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    r = find_best_target(
        frame,
        [],
        50,
        10.0,
        50.0,
        50.0,
        detection_mode=DETECTION_MODE_YOLO,
        yolo_engine=None,
    )
    assert not r.active
    assert r.target is None
    assert any("engine not loaded" in ln for ln in (r.debug_lines or []))


def test_yolo_mode_in_valid_detection_modes() -> None:
    from detector import _VALID_DETECTION_MODES

    assert "yolo" in _VALID_DETECTION_MODES


def test_config_validation_accepts_yolo_mode() -> None:
    from config_validation import validate_config

    cfg = validate_config(
        {
            "detection_mode": "yolo",
            "yolo_weights_path": "tests/fixtures/fake_yolo_weights.pt",
            "fov_radius_pixels": 180,
            "hsv_ranges": [],
            "min_target_area_pixels": 40,
            "pull_strength": 0.5,
            "max_pull_speed_pixels_per_frame": 20,
            "offline_dev_mode": True,
            "allow_live_mouse": False,
        }
    )
    assert cfg["detection_mode"] == "yolo"


def test_reset_yolo_engine_cache() -> None:
    reset_yolo_engine_cache()
    from yolo_detector import _yolo_engine_cache

    assert _yolo_engine_cache is None
