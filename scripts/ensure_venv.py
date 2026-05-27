#!/usr/bin/env python3
"""Create or repair .venv under install_root; used by run_windows.bat."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

MIN_PYTHON = (3, 10)
MAX_PATH_HINT = 200


def _venv_python(venv_dir: Path) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _venv_ok(py: Path) -> bool:
    if not py.is_file():
        return False
    try:
        r = subprocess.run(
            [str(py), "-c", "import sys; assert sys.version_info[:2] >= (3, 10)"],
            capture_output=True,
            timeout=60,
        )
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def ensure_venv(install_root: Path) -> int:
    install_root = install_root.resolve()
    venv_dir = install_root / ".venv"
    py = _venv_python(venv_dir)

    if venv_dir.is_dir() and _venv_ok(py):
        print("venv_ok")
        return 0

    if venv_dir.exists():
        print("removing_broken_venv", flush=True)
        shutil.rmtree(venv_dir, ignore_errors=True)

    if len(str(venv_dir)) > MAX_PATH_HINT:
        print(
            f"warning: install path is long ({len(str(venv_dir))} chars). "
            f"If venv fails, move the folder to e.g. C:\\OverlayAssist",
            flush=True,
        )

    print(f"creating_venv at {venv_dir}", flush=True)
    try:
        subprocess.run(
            [sys.executable, "-m", "venv", str(venv_dir)],
            check=True,
            timeout=600,
        )
    except subprocess.CalledProcessError as exc:
        print(f"venv create failed: {exc}", file=sys.stderr)
        return 1

    if not _venv_ok(py):
        print(f"error: venv python missing or broken: {py}", file=sys.stderr)
        if len(str(py)) > MAX_PATH_HINT:
            print(
                "hint: Windows often breaks venv on very long paths. "
                "Move the whole folder to C:\\OverlayAssist and run run_windows.bat again.",
                file=sys.stderr,
            )
        shutil.rmtree(venv_dir, ignore_errors=True)
        return 1

    print("venv_created")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if len(args) != 1:
        print("usage: ensure_venv.py <install_root>", file=sys.stderr)
        return 2
    if sys.version_info < MIN_PYTHON:
        print(f"need Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+", file=sys.stderr)
        return 2
    return ensure_venv(Path(args[0]))


if __name__ == "__main__":
    raise SystemExit(main())
