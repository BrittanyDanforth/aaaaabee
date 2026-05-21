"""Ban-risk policy — external assist is NOT safe on live anti-cheat titles."""

from __future__ import annotations

from typing import Any

from profiles import (
    PROFILE_APEX_STYLE_DRY_RUN,
    PROFILE_APEX_STYLE_LIVE_SAFE,
    PROFILE_APEX_STYLE_LIVE_TRACE,
    PROFILE_APEX_STYLE_PERF_TEST,
    PROFILE_OWNED_DEV_LIVE,
    normalize_profile_name,
)

# Executable names commonly protected by kernel anti-cheat (presence check only).
KNOWN_ANTICHEAT_PROCESS_NAMES = frozenset(
    {
        "r5apex.exe",
        "r5apex_dx12.exe",
        "fortniteclient-win64-shipping.exe",
        "valorant-win64-shipping.exe",
        "cod.exe",
        "pubg.exe",
        "tslgame.exe",
    }
)

_LIVE_INPUT_PROFILES = frozenset(
    {
        PROFILE_APEX_STYLE_LIVE_SAFE,
        PROFILE_APEX_STYLE_LIVE_TRACE,
    }
)

BAN_ACK_MESSAGE = """ABA WILL RISK A PERMANENT BAN IF USED WITH LIVE ANTI-CHEAT

This tool uses:
• Screen capture (mss) at high FPS
• Global input hooks (pynput) — NOT "safe because no injection"
• Synthetic mouse movement (Win32 / pynput)

Easy Anti-Cheat (Apex), BattlEye, and Vanguard detect external assist regardless of injection.

Only continue if you are on an OFFLINE / PRIVATE build with NO anti-cheat, or you accept total account loss.

Set allow_live_mouse: true only after reading SAFETY.md."""

DRY_RUN_FPS_NOTE = (
    "30 FPS dry-run (capped): safe silhouette/mask sanity check — NOT proof of flick tracking, "
    "fast reacquire, or prediction under combat. Run apex_style_perf_test + Benchmark for timing. "
    "Assume ban if you enable live input on protected online Apex."
)


def is_known_anticheat_process(name: str) -> bool:
    for part in name.split(","):
        n = part.strip().lower()
        if not n:
            continue
        if not n.endswith(".exe"):
            n = f"{n}.exe"
        if n in KNOWN_ANTICHEAT_PROCESS_NAMES:
            return True
    return False


def live_assist_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("allow_live_mouse", False))


def dry_run_mode(config: dict[str, Any]) -> bool:
    """No OS mouse movement and no global hooks."""
    return not live_assist_enabled(config)


def hooks_enabled(config: dict[str, Any]) -> bool:
    return live_assist_enabled(config)


def kill_switch_active(config: dict[str, Any]) -> bool:
    return live_assist_enabled(config)


def validate_runtime_policy(config: dict[str, Any]) -> tuple[bool, str]:
    if not bool(config.get("offline_dev_mode", True)):
        return (
            False,
            "offline_dev_mode must be true (private/offline anti-cheat acknowledgment — NOT dry-run). See SAFETY.md.",
        )

    profile = normalize_profile_name(str(config.get("profile", PROFILE_APEX_STYLE_LIVE_TRACE)))
    if profile in (PROFILE_APEX_STYLE_DRY_RUN, PROFILE_APEX_STYLE_PERF_TEST):
        if live_assist_enabled(config):
            return (
                False,
                f"Profile {profile} cannot enable allow_live_mouse. "
                "Use apex_style_live_trace for live input with GUI ban acknowledgment.",
            )

    if live_assist_enabled(config) and profile not in _LIVE_INPUT_PROFILES:
        proc = str(config.get("target_process_name", "")).strip().lower()
        if is_known_anticheat_process(proc):
            return (
                False,
                f"allow_live_mouse with {proc} is ban bait on live EAC/BattlEye clients. "
                "Use apex_style_live_trace (default) with ban acknowledgment, or dry-run profiles.",
            )
    return True, ""


def block_extra_capture_reason(config: dict[str, Any]) -> str | None:
    """Block benchmark/debug/self-check capture while the configured game exe is running."""
    proc = str(config.get("target_process_name", "")).strip()
    if not proc:
        return None
    from process_presence import is_target_process_running

    if is_target_process_running(proc):
        return (
            "Target game is running — close the game first to avoid extra screen capture on live EAC."
        )
    return None
