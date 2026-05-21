"""Runtime policy helpers — dry-run vs live mouse (minimal copy for repo tests)."""

from __future__ import annotations

from typing import Any


def live_assist_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("allow_live_mouse", False))


def dry_run_mode(config: dict[str, Any]) -> bool:
    if live_assist_enabled(config):
        return False
    return bool(config.get("dry_run", True))


def hooks_enabled(config: dict[str, Any]) -> bool:
    return live_assist_enabled(config) and bool(config.get("enable_hooks", False))


def kill_switch_active(config: dict[str, Any]) -> bool:
    return live_assist_enabled(config) and bool(config.get("kill_switch_active", False))


def validate_runtime_policy(config: dict[str, Any]) -> tuple[bool, str]:
    if not isinstance(config, dict):
        return False, "config must be a dict"
    return True, ""
