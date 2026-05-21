"""Path helpers for ABA / OverlayAssist."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent
LOGS_DIR = APP_ROOT / "logs"
SETUP_LOG = LOGS_DIR / "setup.log"
SELFCHECK_LOG = LOGS_DIR / "selfcheck.log"


def resolve_config_path(path: Path | str) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = (APP_ROOT / p).resolve()
    else:
        p = p.resolve()
    if not p.is_file():
        raise FileNotFoundError(f"config not found: {p}")
    return p


def resolve_log_path(log_file: str) -> str:
    p = Path(log_file)
    if not p.is_absolute():
        p = LOGS_DIR / p.name
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    return str(p.resolve())


def open_logs_folder() -> tuple[bool, str]:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    path = str(LOGS_DIR.resolve())
    try:
        if sys.platform == "win32":
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", path], check=False)
        else:
            subprocess.run(["xdg-open", path], check=False)
        return True, path
    except Exception as exc:
        return False, str(exc)
