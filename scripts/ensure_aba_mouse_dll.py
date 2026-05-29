#!/usr/bin/env python3
"""Verify bundled aba_mouse.dll or build from open-source C."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
DLL = APP_ROOT / "third_party" / "apexaimbot" / "driver" / "aba_mouse.dll"
EXPECTED_SHA = "4f9c8d6e5b7de1a0d9eb2bebea92352e43ced56812758727fbfc05a34df35992"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def try_build() -> bool:
    sh = APP_ROOT / "scripts" / "build_aba_mouse_dll.sh"
    if not sh.is_file():
        return False
    try:
        subprocess.run(["bash", str(sh)], check=True, cwd=APP_ROOT)
        return DLL.is_file()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def main() -> int:
    print("aba_mouse.dll (open-source SendInput driver)")
    if not DLL.is_file():
        print("  missing — attempting build from C source...")
        if not try_build():
            print(
                "  FAIL: run scripts\\build_aba_mouse_dll.bat on Windows "
                "or install mingw-w64 and run build_aba_mouse_dll.sh",
                file=sys.stderr,
            )
            return 1
    got = sha256_file(DLL)
    if got != EXPECTED_SHA:
        print(f"  WARN sha256 {got} != manifest {EXPECTED_SHA} (rebuilt locally?)")
    else:
        print(f"  OK sha256 {got[:16]}...")
    print(f"  OK {DLL} ({DLL.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
