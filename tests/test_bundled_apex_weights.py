"""Bundled ApexAimBot TensorRT engine ships in-repo."""

from __future__ import annotations

import hashlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ENGINE = REPO / "third_party" / "apexaimbot" / "weights" / "APEX416SFP32.engine"
EXPECTED = "e8d61fede4f6f59fb24cf1245394013e52d21a0a3a374cac39085db1c6e6dd16"


def test_bundled_engine_present_and_verified() -> None:
    assert ENGINE.is_file(), f"missing bundled weights: {ENGINE}"
    size = ENGINE.stat().st_size
    assert 5_000_000 < size < 15_000_000, f"unexpected size {size}"
    h = hashlib.sha256()
    with ENGINE.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    assert h.hexdigest() == EXPECTED
