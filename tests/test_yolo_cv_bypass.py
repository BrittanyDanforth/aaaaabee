"""YOLO-primary path must not depend on red-mask CV wiring."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from detector import DETECTION_MODE_YOLO, DetectionResult, Target, draw_debug
from profiles import PROFILE_APEXAIMBOT, PROFILE_DEFAULTS, is_yolo_detection, uses_apex_pid_pull
from yolo_detector import find_best_yolo_target


def test_apexaimbot_profile_flags() -> None:
    p = PROFILE_DEFAULTS[PROFILE_APEXAIMBOT]
    assert is_yolo_detection(p)
    assert uses_apex_pid_pull(p)
    assert p.get("detection_motion_assist") is False
    assert p.get("yolo_skip_motion_smooth") is True


def test_find_best_yolo_target_respects_confidence() -> None:
    frame = np.zeros((416, 416, 3), dtype=np.uint8)
    low = Target(
        200.0,
        200.0,
        5000.0,
        10.0,
        0.4,
        bbox_x=150,
        bbox_y=80,
        bbox_w=80,
        bbox_h=160,
        solidity=0.75,
        humanoid_score=0.4,
        part_count=3,
        body_shape_score=0.4,
        head_score=0.4,
        torso_score=0.4,
        limb_stack_score=0.5,
        red_coverage=0.12,
        fill_ratio=0.65,
        max_circularity=0.45,
        has_classified_torso=True,
    )
    mock_rt = MagicMock()
    with patch(
        "yolo_detector.detect_frame",
        return_value=DetectionResult(low, 1, 0.4, active=True),
    ):
        r = find_best_yolo_target(
            frame,
            208,
            100.0,
            208.0,
            208.0,
            engine=mock_rt,
            min_confidence=0.5,
        )
    assert r.target is None
    assert any("below_min_confidence" in ln for ln in (r.debug_lines or []))


def test_draw_debug_yolo_skips_cv_mask(monkeypatch) -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    called: list[str] = []

    def _boom(*_a, **_k):
        called.append("mask")
        raise AssertionError("build_detection_mask must not run for yolo debug")

    monkeypatch.setattr("detector.build_detection_mask", _boom)
    out = draw_debug(
        frame,
        None,
        40,
        detection_mode=DETECTION_MODE_YOLO,
        hsv_ranges=[],
    )
    assert out.shape == frame.shape
    assert not called


def test_runtime_lazy_pull_controller_source() -> None:
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "runtime.py").read_text(
        encoding="utf-8"
    )
    assert "uses_apex_pid_pull" in text
    assert "sync_config_subsystems" in text
    assert "uses_apex_pid_pull(cfg)" in text
