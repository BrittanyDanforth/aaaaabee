#!/usr/bin/env python3
"""ABA entry — polished control panel + overlay runtime."""

from __future__ import annotations

import sys
from pathlib import Path

from assist import CONFIG_PATH, load_config, setup_logging
from path_utils import APP_ROOT


def main() -> int:
    try:
        cfg = load_config(CONFIG_PATH)
    except Exception as exc:
        print(f"[ABA] Config error: {exc}", file=sys.stderr)
        return 1
    setup_logging(cfg)
    from aba_gui import run_gui

    run_gui(cfg, CONFIG_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
