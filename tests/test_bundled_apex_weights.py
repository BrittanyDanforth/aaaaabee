"""Bundled ApexAimBot TensorRT engine ships in-repo."""

from __future__ import annotations

import hashlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ENGINES = {
    "APEX416SFP32.engine": "e8d61fede4f6f59fb24cf1245394013e52d21a0a3a374cac39085db1c6e6dd16",
    "APEX22W.pt": "fea589b8d3d6097d1dd66659fb68dc8c90e23de93a27627458742cd9707aa144",
}


def test_bundled_engine_present_and_verified() -> None:
    name = "APEX416SFP32.engine"
    expected = ENGINES[name]
    ENGINE = REPO / "third_party" / "apexaimbot" / "weights" / name
    assert ENGINE.is_file(), f"missing bundled weights: {ENGINE}"
    size = ENGINE.stat().st_size
    assert 5_000_000 < size < 15_000_000, f"unexpected size {size}"
    h = hashlib.sha256()
    with ENGINE.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    assert h.hexdigest() == expected


def test_bundled_apex22w_pt_present() -> None:
    path = REPO / "third_party" / "apexaimbot" / "weights" / "APEX22W.pt"
    assert path.is_file()
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    assert h.hexdigest() == ENGINES["APEX22W.pt"]
