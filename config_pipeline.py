"""Single config path: profile → INI merge (YOLO) → validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config_validation import ConfigError, validate_config
from path_utils import resolve_config_path
from profiles import apply_profile

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


def merge_apex_ini(cfg: dict[str, Any]) -> dict[str, Any]:
    """Apply vendored 1bit.ai.config when YOLO mode is active."""
    if str(cfg.get("detection_mode", "apex")).strip().lower() != "yolo":
        return cfg
    from apexaimbot_bridge import prepare_apex_cfg

    return prepare_apex_cfg(cfg)


def normalize_app_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Profile defaults, Apex INI merge, validated keys — use everywhere."""
    cfg = apply_profile(dict(raw))
    cfg = merge_apex_ini(cfg)
    return validate_config(cfg)


def load_app_config(path: Path | str | None = None) -> dict[str, Any]:
    resolved = resolve_config_path(path or DEFAULT_CONFIG_PATH)
    with resolved.open(encoding="utf-8") as f:
        data = json.load(f)
    for entry in data.get("hsv_ranges", []):
        if isinstance(entry, dict):
            entry.pop("comment", None)
    return normalize_app_config(data)
