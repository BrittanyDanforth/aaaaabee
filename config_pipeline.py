"""Single config path: profile → INI merge (YOLO) → validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config_validation import ConfigError, validate_config
from path_utils import resolve_config_path
from profiles import apply_profile

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"

# Keys applied when leaving YOLO + ApexAimBot PID (GUI mode dropdown, explicit CV switch).
LEAVE_YOLO_STACK_PATCH: dict[str, Any] = {
    "pull_mode": "aba",
    "mouse_backend": "auto",
    "profile": "apex_style_live_trace",
}

# Keys applied when detection_mode=yolo but stack was left on CV / incomplete.
ENTER_YOLO_STACK_PATCH: dict[str, Any] = {
    "profile": "apexaimbot",
    "pull_mode": "apexaimbot_pid",
    "mouse_backend": "apexaimbot",
    "detection_motion_assist": False,
    "humanoid_min_height_pixels": 0,
    "min_target_area_pixels": 1,
    "body_shape_min_score": 0.40,
    "yolo_direct_overlay": True,
    "yolo_skip_motion_smooth": True,
}

# Back-compat alias (docs / older branches).
CV_LEAVE_YOLO_PATCH = LEAVE_YOLO_STACK_PATCH


def merge_leave_yolo_stack(patch: dict[str, Any]) -> dict[str, Any]:
    """Return patch with CV stack keys when patch explicitly leaves YOLO."""
    out = dict(patch)
    if "detection_mode" not in out:
        return out
    mode = str(out.get("detection_mode", "")).strip().lower()
    if mode == "yolo":
        return out
    out.update(LEAVE_YOLO_STACK_PATCH)
    return out


def merge_enter_yolo_stack(patch: dict[str, Any]) -> dict[str, Any]:
    """Return patch with full ApexAimBot stack when patch sets detection_mode=yolo."""
    out = dict(patch)
    if str(out.get("detection_mode", "")).strip().lower() != "yolo":
        return out
    for key, value in ENTER_YOLO_STACK_PATCH.items():
        out.setdefault(key, value)
    return out


def reconcile_detection_stack(cfg: dict[str, Any]) -> dict[str, Any]:
    """Fix inconsistent CV↔YOLO stack (pull, mouse, profile, YOLO post-filters)."""
    from profiles import is_yolo_detection, uses_apex_pid_pull

    if is_yolo_detection(cfg):
        out = dict(cfg)
        if not uses_apex_pid_pull(out):
            out.update(ENTER_YOLO_STACK_PATCH)
        out["detection_motion_assist"] = False
        out["humanoid_min_height_pixels"] = 0
        if int(out.get("min_target_area_pixels", 40)) > 40:
            out["min_target_area_pixels"] = 1
        return out
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
    seeded = merge_enter_yolo_stack(dict(raw))
    cfg = apply_profile(seeded)
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
