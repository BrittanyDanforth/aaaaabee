"""Reference profiles — Apex safe-by-default; owned dev requires explicit opt-in."""

from __future__ import annotations

from typing import Any

# Apex Legends PC executables (DX11 + DX12). Process presence only — no injection.
APEX_PROCESS_NAME = "r5apex.exe,r5apex_dx12.exe"

# Shared Apex tuning (HSV, pull, humanoid filters, pull/gate trace).
_APEX_TUNING: dict[str, Any] = {
    "target_process_name": APEX_PROCESS_NAME,
    "target_window_title": "",
    "target_process_required": False,
    "pause_on_target_closed": True,
    "offline_dev_mode": True,
    "hsv_ranges": [
        {"lower": [0, 80, 80], "upper": [15, 255, 255]},
        {"lower": [165, 80, 80], "upper": [179, 255, 255]},
        {"lower": [0, 150, 150], "upper": [25, 255, 255]},
    ],
    "fov_radius_pixels": 185,
    "fov_radius_ads_pixels": 255,
    "capture_fov_crop": True,
    "capture_crop_padding": 1.4,
    "max_pull_speed_pixels_per_frame": 18.0,
    "pull_strength": 0.78,
    "deadzone_pixels": 3,
    "velocity_smoothing": 0.50,
    "smoothing_curve": "ease_out",
    "magnetism_radius_pixels": 80,
    "magnetism_min_pull_scale": 0.70,
    "fov_edge_min_pull_scale": 0.65,
    "prediction_enabled": True,
    "prediction_lead_seconds": 0.055,
    "prediction_max_pixels": 36,
    "humanize_enabled": True,
    "humanize_amplitude_pixels": 0.20,
    "humanize_jerk_limit": 10.0,
    "target_stickiness_pixels": 60,
    "target_lost_frames_before_unlock": 18,
    "distance_score_weight": 2.6,
    "area_score_weight": 0.015,
    "min_target_area_pixels": 20,
    "humanoid_min_height_pixels": 16,
    "humanoid_min_aspect": 0.5,
    "humanoid_max_aspect": 5.5,
    "humanoid_min_solidity": 0.15,
    "torso_aim_fraction": 0.36,
    "mouse_backend": "auto",
    "stats_log_interval_frames": 45,
    "mouse_gate_stale_grace_frames": 12,
    "mouse_gate_pull_budget_scale": 3.5,
    "trace_pull": False,
    "trace_pull_console": False,
    "trace_pull_log_file": "logs/pull_trace.log",
    "trace_pull_interval_frames": 1,
    "trace_pull_max_frames": 900,
}

PROFILE_APEX_STYLE_DRY_RUN = "apex_style_dry_run"
PROFILE_APEX_STYLE_LIVE_SAFE = "apex_style_live_safe"
PROFILE_APEX_STYLE_LIVE_TRACE = "apex_style_live_trace"
PROFILE_APEX_STYLE_PERF_TEST = "apex_style_perf_test"
PROFILE_OWNED_DEV_LIVE = "owned_dev_live"
PROFILE_LEGACY_APEX = "apex_style"
PROFILE_CUSTOM = "custom"

VALID_PROFILES = frozenset(
    {
        PROFILE_APEX_STYLE_DRY_RUN,
        PROFILE_APEX_STYLE_LIVE_SAFE,
        PROFILE_APEX_STYLE_LIVE_TRACE,
        PROFILE_APEX_STYLE_PERF_TEST,
        PROFILE_OWNED_DEV_LIVE,
        PROFILE_LEGACY_APEX,
        PROFILE_CUSTOM,
    }
)

PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    PROFILE_APEX_STYLE_DRY_RUN: {
        **_APEX_TUNING,
        "profile": PROFILE_APEX_STYLE_DRY_RUN,
        "allow_live_mouse": False,
        "capture_fps": 30,
        "ads_input_mode": "disabled",
        "dry_run_force_detect": True,
    },
    PROFILE_APEX_STYLE_LIVE_SAFE: {
        **_APEX_TUNING,
        "profile": PROFILE_APEX_STYLE_LIVE_SAFE,
        "allow_live_mouse": True,
        "capture_fps": 30,
        "ads_input_mode": "both",
        "dry_run_force_detect": False,
    },
    PROFILE_APEX_STYLE_LIVE_TRACE: {
        **_APEX_TUNING,
        "profile": PROFILE_APEX_STYLE_LIVE_TRACE,
        "allow_live_mouse": True,
        "capture_fps": 30,
        "ads_input_mode": "both",
        "dry_run_force_detect": False,
        "trace_pull": True,
        "trace_pull_console": True,
        "verbose_logging": True,
        "stats_log_interval_frames": 30,
        "trace_pull_max_frames": 600,
    },
    PROFILE_APEX_STYLE_PERF_TEST: {
        **_APEX_TUNING,
        "profile": PROFILE_APEX_STYLE_PERF_TEST,
        "allow_live_mouse": False,
        "capture_fps": 60,
        "ads_input_mode": "disabled",
        "dry_run_force_detect": True,
    },
    PROFILE_OWNED_DEV_LIVE: {
        **_APEX_TUNING,
        "profile": PROFILE_OWNED_DEV_LIVE,
        "allow_live_mouse": False,
        "target_process_name": "",
        "capture_fps": 90,
        "ads_input_mode": "both",
        "dry_run_force_detect": False,
    },
    PROFILE_LEGACY_APEX: {
        **_APEX_TUNING,
        "profile": PROFILE_APEX_STYLE_DRY_RUN,
        "allow_live_mouse": False,
        "capture_fps": 30,
        "ads_input_mode": "disabled",
        "dry_run_force_detect": True,
    },
}

APEX_STYLE_DEFAULTS = PROFILE_DEFAULTS[PROFILE_APEX_STYLE_DRY_RUN].copy()
APEX_STYLE_DEFAULTS["profile"] = PROFILE_LEGACY_APEX


def normalize_profile_name(name: str) -> str:
    p = str(name).lower().strip()
    if p == PROFILE_LEGACY_APEX:
        return PROFILE_APEX_STYLE_DRY_RUN
    if p not in VALID_PROFILES:
        return PROFILE_APEX_STYLE_DRY_RUN
    return p


def effective_capture_fps(config: dict[str, Any]) -> int:
    """Configured FPS after profile + dry-run safety cap."""
    fps = int(config.get("capture_fps", 15))
    profile = normalize_profile_name(str(config.get("profile", PROFILE_APEX_STYLE_LIVE_SAFE)))
    if profile in (PROFILE_APEX_STYLE_DRY_RUN, PROFILE_APEX_STYLE_LIVE_SAFE, PROFILE_APEX_STYLE_LIVE_TRACE):
        return min(fps, 30)
    if not config.get("allow_live_mouse", False):
        if profile == PROFILE_APEX_STYLE_PERF_TEST:
            return min(fps, 120)
        return min(fps, 30)
    return fps


def effective_fov_radius(config: dict[str, Any], *, ads_active: bool) -> int:
    """
    Detection + capture radius. Larger while ADS so edge targets stay in frame.
    Falls back to fov_radius_pixels if fov_radius_ads_pixels unset.
    """
    base = int(config.get("fov_radius_pixels", 140))
    if not ads_active:
        return max(80, base)
    ads = int(config.get("fov_radius_ads_pixels", 0))
    if ads <= 0:
        scale = float(config.get("fov_ads_scale", 1.38))
        ads = int(base * scale)
    return max(base, ads)


def apply_profile(raw: dict[str, Any]) -> dict[str, Any]:
    """Merge profile defaults under user config (user keys win)."""
    profile = normalize_profile_name(str(raw.get("profile", PROFILE_APEX_STYLE_LIVE_SAFE)))
    merged: dict[str, Any] = {}
    if profile in PROFILE_DEFAULTS:
        merged.update(PROFILE_DEFAULTS[profile])
    elif profile == PROFILE_CUSTOM:
        merged.update(_APEX_TUNING)
        merged["profile"] = PROFILE_CUSTOM
    merged.update(raw)
    merged["profile"] = profile
    if not merged.get("allow_live_mouse"):
        merged["ads_input_mode"] = "disabled"
    return merged
