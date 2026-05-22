"""Launch external debug HUD (OpenCV) in a subprocess."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from path_utils import APP_ROOT


def launch_debug_hud(config_path: Path) -> tuple[bool, str, subprocess.Popen | None]:
    script = APP_ROOT / "scripts" / "save_detection_artifacts.py"
    if not script.is_file():
        return False, f"Missing {script.name}", None
    cmd = [
        sys.executable,
        str(script),
        "--all-references",
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(APP_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return (
            True,
            f"Debug artifacts running (PID {proc.pid}). Check artifacts/ folder.",
            proc,
        )
    except Exception as exc:
        return False, str(exc), None
