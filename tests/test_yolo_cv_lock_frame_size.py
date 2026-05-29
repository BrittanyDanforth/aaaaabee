"""Regression: CV apply_target_lock must receive (width, height) from yolo_detect_and_lock."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from detector import DetectionResult, Target
from target_lock import TargetLockState
from yolo_targeting import yolo_detect_and_lock


def test_yolo_cv_lock_receives_unswapped_frame_size() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    fake_t = Target(50.0, 50.0, 1000.0, 5.0, 0.9, bbox_w=20, bbox_h=40)
    state = TargetLockState()
    engine = MagicMock()
    locked = DetectionResult(fake_t, 1, 0.9, active=True)

    with patch(
        "yolo_targeting.find_best_yolo_target",
        return_value=DetectionResult(fake_t, 1, 0.9, active=True),
    ), patch(
        "yolo_targeting.apply_target_lock",
        return_value=(locked, False),
    ) as mock_cv_lock:
        yolo_detect_and_lock(
            {
                "detection_mode": "yolo",
                "yolo_confidence_min": 0.5,
                "min_target_area_pixels": 1,
                "target_stickiness_pixels": 90,
                "humanoid_min_height_pixels": 0,
                "yolo_apex_nearest_lock": False,
                "target_lost_frames_before_unlock": 10,
            },
            frame,
            engine,
            state,
            fov_radius=50,
            center_x=50.0,
            center_y=50.0,
            frame_size=(1920, 1080),
        )
    assert mock_cv_lock.call_count == 1
    assert mock_cv_lock.call_args.kwargs["frame_size"] == (1920, 1080)
