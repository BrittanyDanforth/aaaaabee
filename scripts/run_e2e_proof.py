#!/usr/bin/env python3
"""Run all proof tests + write detection artifacts. Exit 0 only if everything passes."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    steps = [
        (["python3", "-m", "unittest", "discover", "-s", "tests", "-q"], "unit tests"),
        (
            ["python3", str(ROOT / "scripts" / "save_detection_artifacts.py"), "--all-references"],
            "detection artifacts",
        ),
    ]
    for cmd, label in steps:
        print(f"\n--- {label} ---")
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0:
            print(f"FAILED: {label}")
            return r.returncode
    print("\nAll proofs passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
