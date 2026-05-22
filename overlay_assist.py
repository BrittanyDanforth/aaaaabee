#!/usr/bin/env python3
"""
OverlayAssist entry point — external screen-only aim assist.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from assist import load_config, setup_logging
from config_validation import ConfigError
from path_utils import resolve_config_path
from platform_info import enable_dpi_awareness
CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="OverlayAssist — external screen aim assist")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--overlay", action="store_true", help="Transparent FOV overlay")
    parser.add_argument("--debug", action="store_true", help="OpenCV debug + stats HUD")
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Verify deps, capture, config; exit without assist loop",
    )
    args = parser.parse_args()

    try:
        resolve_config_path(args.config)
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 1

    enable_dpi_awareness()

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"Invalid config: {exc}", file=sys.stderr)
        return 1

    if args.overlay:
        config["enable_overlay"] = True
    if args.debug:
        config["show_debug_window"] = True
        config["verbose_logging"] = True
        config["stats_log_interval_frames"] = min(
            int(config.get("stats_log_interval_frames", 60)), 30
        )

    setup_logging(config)

    if args.self_check:
        from self_check import run_self_check

        return run_self_check(config)

    from runtime import AssistRuntime

    runtime = AssistRuntime(config, resolve_config_path(args.config))
    return runtime.run()


if __name__ == "__main__":
    raise SystemExit(main())
