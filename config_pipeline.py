"""Single config path: profile → INI merge (YOLO) → validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config_validation import ConfigError, validate_config
from path_utils import resolve_config_path
from profiles import apply_profile

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"

# Keys applied when leaving YOLO + ApexAimBot PID (GUI presets, mode dropdown, normalize).
LEAVE_YOLO_STACK_PATCH: dict[str, Any] = {
    "pull_mode": "aba",
    "mouse_backend": "auto",
    "profile": "apex_style_live_trace",
}

# Back-compat alias (docs / older branches).
CV_LEAVE_YOLO_PATCH = LEAVE_YOLO_STACK_PATCH


def merge_leave_yolo_stack(patch: dict[str, Any]) -> dict[str, Any]:
    """Return patch with CV stack keys when detection is not YOLO."""
    mode = str(patch.get("detection_mode", "")).strip().lower()
    if mode == "yolo":
        return dict(patch)
    out = dict(patch)
    out.update(LEAVE_YOLO_STACK_PATCH)
    return out


def reconcile_detection_stack(cfg: dict[str, Any]) -> dict[str, Any]:
    """Fix inconsistent CV detection + leftover ApexAimBot pull/mouse/profile."""
    from profiles import is_yolo_detection

    if is_yolo_detection(cfg):
        return cfg
    out = dict(cfg)
    pull = str(out.get("pull_mode", "aba")).strip().lower()
    mouse = str(out.get("mouse_backend", "auto")).strip().lower()
    profile = str(out.get("profile", "")).strip().lower()
    if pull == "apexaimbot_pid" or mouse == "apexaimbot" or profile == "apexaimbot":
        out.update(LEAVE_YOLO_STACK_PATCH)
    return out


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
    cfg = validate_config(cfg)
    return reconcile_detection_stack(cfg)


def load_app_config(path: Path | str | None = None) -> dict[str, Any]:
    resolved = resolve_config_path(path or DEFAULT_CONFIG_PATH)
    with resolved.open(encoding="utf-8") as f:
        data = json.load(f)
    for entry in data.get("hsv_ranges", []):
        if isinstance(entry, dict):
            entry.pop("comment", None)
    return normalize_app_config(data)
