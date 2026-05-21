"""Config profiles and loading — live safe + trace variants."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

CONFIG_DIR = Path(__file__).resolve().parent / "config"

PROFILE_CHAIN: dict[str, list[str]] = {
    "apex_style_live_trace": ["apex_style_live_safe.json", "apex_style_live_trace.json"],
}

PROFILE_FILES: dict[str, str] = {
    "apex_style_live_safe": "apex_style_live_safe.json",
    "apex_style_live_trace": "apex_style_live_trace.json",
    "apex_style_dry_run": "apex_style_dry_run.json",
}


def effective_capture_fps(config: dict[str, Any]) -> float:
    fps = float(config.get("capture_fps", 60))
    return max(1.0, min(fps, 240.0))


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, val in overlay.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def _apply_profile(merged: dict[str, Any], prof: str) -> dict[str, Any]:
    if prof in PROFILE_CHAIN:
        for fname in PROFILE_CHAIN[prof]:
            pth = CONFIG_DIR / fname
            if pth.is_file():
                merged = _deep_merge(merged, json.loads(pth.read_text(encoding="utf-8")))
    elif prof in PROFILE_FILES:
        prof_path = CONFIG_DIR / PROFILE_FILES[prof]
        if prof_path.is_file():
            merged = _deep_merge(merged, json.loads(prof_path.read_text(encoding="utf-8")))
    return merged


def load_config(path: Path | str | None = None, *, profile: str | None = None) -> dict[str, Any]:
    """
    Load config: defaults.json <- profile chain <- user path overrides.
    config.json with only {\"profile\": \"apex_style_live_safe\"} expands the full profile.
    """
    defaults_path = CONFIG_DIR / "defaults.json"
    merged: dict[str, Any] = {}
    if defaults_path.is_file():
        merged = json.loads(defaults_path.read_text(encoding="utf-8"))

    user_overlay: dict[str, Any] = {}
    if path is not None:
        user_path = Path(path)
        if user_path.is_file():
            user_overlay = json.loads(user_path.read_text(encoding="utf-8"))

    prof = profile or user_overlay.get("profile") or merged.get("profile")
    if prof:
        merged = _apply_profile(merged, str(prof))

    if user_overlay:
        merged = _deep_merge(merged, user_overlay)
        prof = merged.get("profile", prof)

    merged.setdefault("profile", prof or "apex_style_live_safe")
    return merged


def resolve_config_path(path: Path | str | None = None, *, profile: str | None = None) -> Path:
    if path is not None:
        p = Path(path)
        if p.is_file():
            return p.resolve()
    prof = profile or "apex_style_live_safe"
    if prof in PROFILE_FILES:
        candidate = CONFIG_DIR / PROFILE_FILES[prof]
        if candidate.is_file():
            return candidate.resolve()
    return (CONFIG_DIR / "apex_style_live_safe.json").resolve()
