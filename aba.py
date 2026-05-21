#!/usr/bin/env python3
"""ABA — Apex-style external aim assist launcher."""

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
    parser = argparse.ArgumentParser(
        description="ABA — Apex Legends reference external aim assist (no injection)"
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--cli", action="store_true", help="Run assist loop without GUI")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--doctor", action="store_true", help="Run setup_doctor.py checks")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Measure capture/detect timing only (no mouse movement)",
    )
    args = parser.parse_args()

    try:
        resolve_config_path(args.config)
    except (ConfigError, FileNotFoundError) as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 1

    enable_dpi_awareness()

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"Invalid config: {exc}", file=sys.stderr)
        return 1

    setup_logging(config)

    if args.doctor:
        from setup_doctor import main as doctor_main

        return doctor_main()

    if args.self_check:
        from self_check import run_self_check

        return run_self_check(config)

    if args.benchmark:
        from ban_safety import block_extra_capture_reason
        from perf_benchmark import run_perf_benchmark

        block = block_extra_capture_reason(config)
        if block:
            print(f"[ABA] Benchmark blocked: {block}", file=sys.stderr)
            return 1
        report = run_perf_benchmark(config, duration_sec=3.0)
        print(f"[ABA] Benchmark: {report.summary()}")
        verdict = report.gameplay_readiness.upper()
        if "FAILED" in verdict or "CANNOT" in verdict or "INSUFFICIENT" in verdict:
            return 1
        return 0

    if args.cli or args.debug:
        if args.debug:
            config["show_debug_window"] = True
            config["verbose_logging"] = True
            config["stats_log_interval_frames"] = min(
                int(config.get("stats_log_interval_frames", 60)), 15
            )
        from runtime import AssistRuntime

        resolved = resolve_config_path(args.config)
        return AssistRuntime(config, resolved).run()

    from aba_gui import run_gui

    try:
        run_gui(config, resolve_config_path(args.config))
    except Exception as exc:
        print(f"[ABA] GUI failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
