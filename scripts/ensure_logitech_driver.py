#!/usr/bin/env python3
"""Copy Logitech ghub_mouse.dll from ApexAimBot clone (optional; Windows aim parity)."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
DEST = APP_ROOT / "third_party" / "apexaimbot" / "driver"
DLL_NAMES = ("ghub_mouse.dll", "logitech.driver.dll")


def upstream_function_dir() -> Path | None:
    env = os.environ.get("APEXAIMBOT_SRC", "").strip()
    if env:
        p = Path(env)
        if (p / "function" / "ghub_mouse.dll").is_file():
            return p / "function"
        if (p / "ghub_mouse.dll").is_file():
            return p
    for base in (
        APP_ROOT.parent / "ApexAimBot",
        Path.home() / "ApexAimBot",
        Path("/tmp/ApexAimBot"),
    ):
        d = base / "function"
        if (d / "ghub_mouse.dll").is_file():
            return d
    return None


def main() -> int:
    print("Logitech G HUB driver (optional — copy from ApexAimBot function/)")
    dest_has = (DEST / "ghub_mouse.dll").is_file()
    if dest_has:
        print(f"  OK bundled {DEST / 'ghub_mouse.dll'}")
        for name in DLL_NAMES[1:]:
            p = DEST / name
            if p.is_file():
                print(f"  OK {name}")
        return 0

    src = upstream_function_dir()
    if src is None:
        print(
            "  SKIP no ghub_mouse.dll — set APEXAIMBOT_SRC or copy DLLs to\n"
            f"       {DEST}\n"
            "       mouse_backend=apexaimbot will use Win32 SendInput instead.",
            file=sys.stderr,
        )
        return 0

    DEST.mkdir(parents=True, exist_ok=True)
    copied = False
    for name in DLL_NAMES:
        s = src / name
        if not s.is_file():
            continue
        shutil.copy2(s, DEST / name)
        print(f"  copied {s} -> {DEST / name}")
        copied = True
    if not copied:
        print("  FAIL upstream has no ghub_mouse.dll", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
