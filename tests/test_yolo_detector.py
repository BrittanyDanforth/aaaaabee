"""YOLO-primary detection path (mocked vendored runtime — no torch required in CI)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from detector import DETECTION_MODE_YOLO, DetectionResult, Target, find_best_target
from yolo_detector import reset_yolo_engine_cache


def test_find_best_target_yolo_delegates_without_cv() -> None:
    frame = np.zeros((416, 416, 3), dtype=np.uint8)
    cx, cy = 208.0, 208.0
    fake_t = Target(
        centroid_x=cx,
        centroid_y=cy - 20,
        area=8000,
        distance_to_center=20,
        confidence=0.88,
        bbox_x=150,
        bbox_y=80,
        bbox_w=80,
        bbox_h=160,
        solidity=0.75,
        humanoid_score=0.88,
        part_count=3,
        body_shape_score=0.88,
        head_score=0.88,
        torso_score=0.88,
        limb_stack_score=0.5,
        red_coverage=0.12,
        fill_ratio=0.65,
        max_circularity=0.45,
        has_classified_torso=True,
    )
    mock_rt = MagicMock()
    mock_rt.config.aim_offset_fraction = 0.2
    with patch(
        "yolo_detector.detect_frame",
        return_value=DetectionResult(fake_t, 1, 0.88, debug_lines=["mode=apexaimbot_vendored"], active=True),
    ):
        r = find_best_target(
            frame,
            [],
            250,
            100.0,
            cx,
            cy,
            detection_mode=DETECTION_MODE_YOLO,
            yolo_engine=mock_rt,
        )
    assert r.active
    assert r.target is not None
    assert "apexaimbot" in (r.debug_lines[0] if r.debug_lines else "")


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
            "yolo_yolov5_root": "third_party/apexaimbot",
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
    from apexaimbot_bridge import _engine_cache, get_apexaimbot_runtime

    cfg = {
        "detection_mode": "yolo",
        "yolo_yolov5_root": "third_party/apexaimbot",
        "yolo_weights_path": "third_party/apexaimbot/weights/APEX22W.pt",
    }
    get_apexaimbot_runtime(cfg)
    import apexaimbot_bridge as bridge

    assert bridge._engine_cache is not None
    reset_yolo_engine_cache()
    assert bridge._engine_cache is None
