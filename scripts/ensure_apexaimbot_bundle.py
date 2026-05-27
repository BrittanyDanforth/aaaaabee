#!/usr/bin/env python3
"""Verify vendored ApexAimBot tree + Apex-only weights (delegates to ensure_apexaimbot_weights)."""

from __future__ import annotations

import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    # Import sibling module by path (scripts/ is not a package).
    import importlib.util

    weights_path = Path(__file__).resolve().parent / "ensure_apexaimbot_weights.py"
    spec = importlib.util.spec_from_file_location("ensure_apexaimbot_weights", weights_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    print("ApexAimBot bundle check (weights + vendor tree)")
    ok_all = mod.ensure_apex_weights(verbose=True)

    ini = APP_ROOT / "third_party" / "apexaimbot" / "1bit.ai.config"
    if ini.is_file():
        print("  OK 1bit.ai.config present")
    else:
        print("  FAIL missing third_party/apexaimbot/1bit.ai.config", file=sys.stderr)
        ok_all = False

    vendor = APP_ROOT / "third_party" / "apexaimbot" / "models" / "common.py"
    if vendor.is_file():
        print("  OK vendored yolov5 models/common.py")
    else:
        print("  FAIL missing vendored yolov5 tree", file=sys.stderr)
        ok_all = False

    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
