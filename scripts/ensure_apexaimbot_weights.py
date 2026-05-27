#!/usr/bin/env python3
"""Verify or install bundled ApexAimBot TensorRT weights (safe copy with SHA-256 check)."""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
DEST = APP_ROOT / "third_party" / "apexaimbot" / "weights" / "APEX416SFP32.engine"
EXPECTED_SHA256 = "e8d61fede4f6f59fb24cf1245394013e52d21a0a3a374cac39085db1c6e6dd16"
DEFAULT_SRC_NAME = "APEX416SFP32.engine"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find_upstream_engine() -> Path | None:
    env = os.environ.get("APEXAIMBOT_WEIGHTS_SRC", "").strip()
    if env:
        p = Path(env)
        if p.is_file():
            return p
        candidate = p / "function" / "weights" / DEFAULT_SRC_NAME
        if candidate.is_file():
            return candidate
    for base in (
        APP_ROOT.parent / "ApexAimBot",
        Path.home() / "ApexAimBot",
        Path("/tmp/ApexAimBot"),
    ):
        candidate = base / "function" / "weights" / DEFAULT_SRC_NAME
        if candidate.is_file():
            return candidate
    return None


def main() -> int:
    if DEST.is_file():
        digest = sha256_file(DEST)
        if digest == EXPECTED_SHA256:
            print(f"OK  {DEST} ({DEST.stat().st_size} bytes, sha256 match)")
            return 0
        print(f"FAIL sha256 mismatch for {DEST}", file=sys.stderr)
        print(f"     got {digest}", file=sys.stderr)
        print(f"     expected {EXPECTED_SHA256}", file=sys.stderr)
        if "--force" not in sys.argv:
            return 1

    src = find_upstream_engine()
    if src is None:
        print(
            "FAIL weights missing and no upstream copy found.\n"
            "  Set APEXAIMBOT_WEIGHTS_SRC to ApexAimBot/function/weights/APEX416SFP32.engine\n"
            "  or git pull — weights should ship in third_party/apexaimbot/weights/.",
            file=sys.stderr,
        )
        return 1

    src_digest = sha256_file(src)
    if src_digest != EXPECTED_SHA256:
        print(
            f"FAIL upstream file hash does not match known-good engine:\n"
            f"  {src}\n"
            f"  got {src_digest}\n"
            f"  expected {EXPECTED_SHA256}",
            file=sys.stderr,
        )
        return 1

    DEST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, DEST)
    print(f"OK  copied {src} -> {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
