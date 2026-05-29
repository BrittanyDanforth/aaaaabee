#!/usr/bin/env python3
"""Shared config load + logging for ABA and OverlayAssist entry points."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from config_pipeline import DEFAULT_CONFIG_PATH, load_app_config
from config_validation import ConfigError

CONFIG_PATH = DEFAULT_CONFIG_PATH


def load_config(path: Path | str | None = None) -> dict:
    return load_app_config(path)


def setup_logging(cfg: dict) -> None:
    level = logging.DEBUG if cfg.get("verbose_logging") else logging.INFO
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    log_file = cfg.get("log_file")
    if isinstance(log_file, str) and log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )
