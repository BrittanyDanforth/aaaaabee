"""Minimal config validation used by assist.py / aba.py entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class ConfigError(Exception):
    pass


_REQUIRED_KEYS = (
    "profile",
    "hsv_ranges",
    "fov_radius_pixels",
    "capture_fps",
    "max_pull_speed_pixels_per_frame",
    "pull_strength",
    "deadzone_pixels",
)


def validate_config(cfg: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(cfg, dict):
        raise ConfigError("config must be a JSON object")
    for key in _REQUIRED_KEYS:
        if key not in cfg:
            raise ConfigError(f"missing required config key: {key}")
    if not isinstance(cfg.get("hsv_ranges"), list) or not cfg["hsv_ranges"]:
        raise ConfigError("hsv_ranges must be a non-empty list")
    fps = int(cfg.get("capture_fps", 0))
    if fps < 1 or fps > 240:
        raise ConfigError(f"capture_fps out of range: {fps}")
    fov = int(cfg.get("fov_radius_pixels", 0))
    if fov < 60 or fov > 400:
        raise ConfigError(f"fov_radius_pixels out of range: {fov}")
    return cfg


def resolve_config_path(path: Path | str) -> Path:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise ConfigError(f"config not found: {p}")
    return p
