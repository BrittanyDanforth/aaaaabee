"""Vendored ApexAimBot detect helpers (no torch / weights required)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
VENDOR = REPO / "third_party" / "apexaimbot"


@pytest.fixture(scope="module")
def vendor_detect():
    if not (VENDOR / "detect.py").is_file():
        pytest.skip("vendored apexaimbot not present")
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    import nearest

    return nearest


def test_send_nearest_matches_apexaimbot_math(vendor_detect) -> None:
    # Normalized xywh: center (0.5, 0.5), size 0.2 x 0.4
    box_list = [("target", 0.5, 0.5, 0.2, 0.4, 88)]
    out = vendor_detect.send_nearest_pos_to_mouse_ctrl(  # type: ignore[attr-defined]
        box_list, grab_width=416, grab_height=416
    )
    assert out is not None
    pos, bw, bh, conf = out
    assert pos == pytest.approx((0.0, 0.0), abs=1e-6)
    assert bw == pytest.approx(416 * 0.2, rel=1e-3)
    assert bh == pytest.approx(416 * 0.4, rel=1e-3)
    assert conf == pytest.approx(0.88, rel=1e-3)


def test_vendor_tree_present() -> None:
    assert (VENDOR / "models" / "common.py").is_file()
    assert (VENDOR / "utils" / "general.py").is_file()
    assert (VENDOR / "PID.py").is_file()


def test_default_vendor_root_resolves() -> None:
    from apexaimbot_bridge import VENDOR_DEFAULT, _resolve_vendor_root

    cfg = {"yolo_yolov5_root": ""}
    if VENDOR_DEFAULT.is_dir():
        root = _resolve_vendor_root(cfg)
        assert root == VENDOR_DEFAULT.resolve()
