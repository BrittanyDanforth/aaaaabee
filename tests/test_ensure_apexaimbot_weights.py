"""ensure_apexaimbot_weights.py refuses wrong hashes and stray weight files."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "ensure_apexaimbot_weights.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("ensure_apexaimbot_weights", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_try_copy_refuses_wrong_hash(tmp_path, monkeypatch) -> None:
    mod = _load_module()
    bad = tmp_path / "APEX22W.pt"
    bad.write_bytes(b"not-a-real-pt-file")
    ok, msg = mod.try_copy(
        "APEX22W.pt",
        mod.APEX_WEIGHTS["APEX22W.pt"],
        tmp_path,
    )
    assert not ok
    assert "REFUSED" in msg
    assert mod.sha256_file(bad) != mod.APEX_WEIGHTS["APEX22W.pt"]


def test_reject_unknown_file_in_weights_dir(tmp_path, monkeypatch) -> None:
    mod = _load_module()
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "PUBG22WBODY.engine").write_bytes(b"x" * 100)
    monkeypatch.setattr(mod, "WEIGHTS_DIR", weights)
    ok, msgs = mod.reject_unknown_files_in_weights_dir()
    assert not ok
    assert any("PUBG22WBODY" in m for m in msgs)


def test_reject_unknown_random_filename(tmp_path, monkeypatch) -> None:
    mod = _load_module()
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "mystery.onnx").write_bytes(b"x")
    monkeypatch.setattr(mod, "WEIGHTS_DIR", weights)
    ok, msgs = mod.reject_unknown_files_in_weights_dir()
    assert not ok
    assert any("mystery.onnx" in m for m in msgs)


def test_manifest_matches_bundled_engine() -> None:
    mod = _load_module()
    engine = REPO / "third_party" / "apexaimbot" / "weights" / "APEX416SFP32.engine"
    assert engine.is_file()
    assert mod.sha256_file(engine) == mod.APEX_WEIGHTS["APEX416SFP32.engine"]
