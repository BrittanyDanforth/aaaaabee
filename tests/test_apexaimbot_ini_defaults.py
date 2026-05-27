"""ABA uses 1bit.ai.config defaults, not main.py stale globals."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VENDOR = REPO / "third_party" / "apexaimbot"


def test_ini_config_matches_shipped_1bit() -> None:
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    from ini_config import load_1bit_defaults, merge_app_config

    ini = load_1bit_defaults()
    assert ini.get("iou_thres") == 0.25
    assert ini.get("grab_width") == 416
    assert ini.get("weight") == "APEX416SFP32.engine"
    merged = merge_app_config({})
    assert merged["yolo_iou_thres"] == 0.25
    assert merged["yolo_max_det"] == 3
