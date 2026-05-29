"""Shipped YOLO/ApexAimBot path on committed real Apex screenshots."""

from __future__ import annotations

import pytest

from scripts.audit_yolo_real_apex import INPUTS, run_audit

REQUIRED_FIXTURES = [
    "img4_dummy_not_detected.webp",
    "img5_close_ads.webp",
    "img6_seven_characters.png",
    "img7_sky_dot_bug.png",
]


@pytest.mark.skipif(
    not all((INPUTS / name).exists() for name in REQUIRED_FIXTURES),
    reason="real Apex YOLO fixtures missing",
)
def test_yolo_detects_real_apex_close_and_lineup(tmp_path) -> None:
    rows = {row["image"]: row for row in run_audit(tmp_path, write_images=False)}

    for name in (
        "img4_dummy_not_detected.webp",
        "img5_close_ads.webp",
        "img6_seven_characters.png",
    ):
        row = rows[name]
        assert row["active"], f"{name}: YOLO inactive; debug={row.get('debug')}"
        assert row["candidates"] >= 1
        assert row["score"] >= 0.50
        assert row.get("target"), f"{name}: active without target"

    # Regression for the sky-dot false positive screenshot: YOLO must not invent a lock.
    row = rows["img7_sky_dot_bug.png"]
    assert not row["active"], f"img7 should stay inactive; target={row.get('target')}"
