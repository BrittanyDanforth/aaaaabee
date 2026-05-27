#!/usr/bin/env python3
"""Verify or safely install Apex-only weights (per-file SHA-256; refuse unknown files)."""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = APP_ROOT / "third_party" / "apexaimbot" / "weights"

# Manifest: third_party/apexaimbot/weights/WEIGHTS.md
APEX_WEIGHTS: dict[str, str] = {
    "APEX416SFP32.engine": "e8d61fede4f6f59fb24cf1245394013e52d21a0a3a374cac39085db1c6e6dd16",
    "APEX22W.pt": "fea589b8d3d6097d1dd66659fb68dc8c90e23de93a27627458742cd9707aa144",
}

# Upstream ~120MB folder includes these; never copy (wrong game or wrong hash).
OTHER_GAME_WEIGHTS = (
    "PUBG22WBODY.engine",
    "PUBG22WBODY.onnx",
    "PUBG22WBODY.pt",
    "PUBG模型2W图body单分类.pt",
    "csgo_5n_48000.pt",
    "瓦 紫色盲 标识0.pt",
    "yolov5s.pt",
    "A.onnx",
    "BJGFP32.onnx",
)

ALLOWED_WEIGHTS_DIR_NAMES = frozenset(APEX_WEIGHTS) | {".gitkeep", "WEIGHTS.md"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def upstream_weights_dir() -> Path | None:
    env = os.environ.get("APEXAIMBOT_SRC", "").strip()
    if env:
        p = Path(env)
        if (p / "function" / "weights").is_dir():
            return p / "function" / "weights"
        if p.name == "weights" and p.is_dir():
            return p
    for base in (
        APP_ROOT.parent / "ApexAimBot",
        Path.home() / "ApexAimBot",
        Path("/tmp/ApexAimBot"),
    ):
        d = base / "function" / "weights"
        if d.is_dir():
            return d
    return None


def verify_one(name: str, expected: str) -> tuple[bool, str]:
    dest = WEIGHTS_DIR / name
    if not dest.is_file():
        return False, f"missing {dest}"
    got = sha256_file(dest)
    if got != expected:
        return False, f"sha256 mismatch {name}: got {got} expected {expected}"
    return True, f"OK {name} ({dest.stat().st_size} bytes)"


def try_copy(name: str, expected: str, src_dir: Path) -> tuple[bool, str]:
    src = src_dir / name
    if not src.is_file():
        return False, f"upstream missing {src}"
    got = sha256_file(src)
    if got != expected:
        return (
            False,
            f"REFUSED copy {name}: hash {got} != manifest {expected}",
        )
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, WEIGHTS_DIR / name)
    return True, f"copied {src} -> {WEIGHTS_DIR / name}"


def reject_unknown_files_in_weights_dir() -> tuple[bool, list[str]]:
    """Refuse extra blobs in weights/ (e.g. PUBG/CSGO copied by mistake)."""
    if not WEIGHTS_DIR.is_dir():
        return True, []
    problems: list[str] = []
    for entry in WEIGHTS_DIR.iterdir():
        if not entry.is_file():
            continue
        if entry.name in ALLOWED_WEIGHTS_DIR_NAMES:
            continue
        if entry.name in OTHER_GAME_WEIGHTS:
            problems.append(f"REFUSED other-game weight in bundle dir: {entry.name}")
        else:
            problems.append(f"REFUSED unknown file in weights/: {entry.name}")
    return len(problems) == 0, problems


def ensure_apex_weights(*, verbose: bool = True) -> bool:
    ok_all = True
    ok_extra, extra_msgs = reject_unknown_files_in_weights_dir()
    for msg in extra_msgs:
        if verbose:
            print(f"  {msg}", file=sys.stderr)
        ok_all = False
    if not ok_extra:
        return False

    src_dir = upstream_weights_dir()
    for name, expected in APEX_WEIGHTS.items():
        ok, msg = verify_one(name, expected)
        if ok:
            if verbose:
                print(f"  {msg}")
            continue
        if verbose:
            print(f"  {msg}")
        if src_dir is not None:
            ok2, msg2 = try_copy(name, expected, src_dir)
            if verbose:
                print(f"  {msg2}")
            if ok2:
                ok, _ = verify_one(name, expected)
            else:
                ok_all = False
        else:
            ok_all = False
    return ok_all


def main() -> int:
    print("ApexAimBot weights (Apex-only; SHA-256 per WEIGHTS.md)")
    print(f"  Upstream folder has ~120MB; we allow only: {', '.join(APEX_WEIGHTS)}")
    if not ensure_apex_weights():
        print("FAIL — fix weights or set APEXAIMBOT_SRC to a clean ApexAimBot clone", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
