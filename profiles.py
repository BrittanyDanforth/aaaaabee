"""Reference profiles — Apex safe-by-default; owned dev requires explicit opt-in."""

from __future__ import annotations

from pathlib import Path
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
    "fov_radius_pixels": 168,
    "fov_radius_ads_pixels": 218,
    "detection_fov_margin_pixels": 36,
    "capture_fov_crop": True,
    "capture_crop_padding": 1.34,
    "max_pull_speed_pixels_per_frame": 22.0,
    "pull_strength": 0.82,
    "deadzone_pixels": 3,
    "velocity_smoothing": 0.50,
    "smoothing_curve": "ease_out",
    "magnetism_radius_pixels": 80,
    "magnetism_min_pull_scale": 0.70,
    "fov_edge_min_pull_scale": 0.88,
    "prediction_enabled": True,
    "prediction_lead_seconds": 0.034,
    "prediction_max_pixels": 12,
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
    "torso_aim_fraction": 0.40,
    "aim_body_y_min_fraction": 0.28,
    "aim_body_y_max_fraction": 0.52,
    "prediction_vertical_cap_pixels": 4.0,
    "smoothing_tau_still": 0.062,
    "smoothing_tau_moving": 0.028,
    "body_shape_min_score": 0.40,
    "head_score_weight": 0.26,
    "torso_score_weight": 0.26,
    "limb_stack_score_weight": 0.22,
    "debug_show_body_bbox": False,
    "debug_show_anchor": False,
    "debug_show_rejected": False,
    "debug_show_top_candidates": False,
    "debug_show_reject_reasons": False,
    "debug_show_mask_overlay": False,
    "debug_show_timing": True,
    "debug_frames_dir": "artifacts/debug_frames",
    "mouse_backend": "auto",
    "stats_log_interval_frames": 45,
    "mouse_gate_stale_grace_frames": 12,
    "mouse_gate_pull_budget_scale": 3.5,
    "trace_pull": True,
    "trace_pull_console": False,
    "trace_pull_log_file": "logs/pull_trace.log",
    "trace_pull_interval_frames": 2,
    "trace_pull_max_frames": 0,
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
        "trace_pull": False,
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
        "enable_overlay": True,
        "capture_fps": 30,
        "ads_input_mode": "both",
        "dry_run_force_detect": False,
        "trace_pull": True,
        "trace_pull_console": False,
        "verbose_logging": False,
        "stats_log_interval_frames": 45,
        "trace_pull_max_frames": 600,
        "humanize_enabled": False,
        "humanize_amplitude_pixels": 0.0,
        "prediction_lead_seconds": 0.020,
        "prediction_max_pixels": 12,
        "prediction_vertical_cap_pixels": 4.0,
        "target_stickiness_pixels": 92,
        "fov_edge_min_pull_scale": 0.90,
        "max_pull_speed_pixels_per_frame": 26.0,
        "pull_strength": 0.82,
        "deadzone_pixels": 2,
        "torso_aim_fraction": 0.40,
        "aim_body_y_min_fraction": 0.28,
        "aim_body_y_max_fraction": 0.50,
        "smoothing_tau_still": 0.048,
        "smoothing_tau_moving": 0.020,
        "velocity_smoothing": 0.48,
        "body_shape_min_score": 0.40,
        "mouse_gate_stale_grace_frames": 14,
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
    profile = resolve_profile_name(config)
    if profile in (PROFILE_APEX_STYLE_DRY_RUN, PROFILE_APEX_STYLE_LIVE_SAFE, PROFILE_APEX_STYLE_LIVE_TRACE):
        return min(fps, 30)
    if not config.get("allow_live_mouse", False):
        if profile == PROFILE_APEX_STYLE_PERF_TEST:
            return min(fps, 120)
        return min(fps, 30)
    return fps


def effective_fov_radius(config: dict[str, Any], *, ads_active: bool) -> int:
    """Overlay / HUD ring radius (hip-fire vs ADS)."""
    base = int(config.get("fov_radius_pixels", 140))
    if not ads_active:
        return max(80, base)
    ads = int(config.get("fov_radius_ads_pixels", 0))
    if ads <= 0:
        scale = float(config.get("fov_ads_scale", 1.30))
        ads = int(base * scale)
    return max(base, ads)


def effective_detection_fov_radius(config: dict[str, Any], *, ads_active: bool) -> int:
    """
    Detection + pull use this radius (larger than overlay ring).
    Keeps enemies at screen edge inside the HSV/FOV mask so pull does not drop off.
    """
    core = effective_fov_radius(config, ads_active=ads_active)
    margin = int(config.get("detection_fov_margin_pixels", 0))
    if margin <= 0:
        margin = int(core * float(config.get("detection_fov_margin_scale", 0.18)))
    return min(400, core + max(12, margin))


def effective_capture_fov_radius(config: dict[str, Any], *, ads_active: bool) -> int:
    """Capture crop must cover detection FOV + padding."""
    detect = effective_detection_fov_radius(config, ads_active=ads_active)
    extra = int(config.get("capture_extra_pixels", 8))
    return detect + extra


PROFILE_LIVE_DEFAULT = PROFILE_APEX_STYLE_LIVE_TRACE


def resolve_profile_name(raw: dict[str, Any]) -> str:
    """Default live runs use apex_style_live_trace unless config names a profile."""
    explicit = str(raw.get("profile", "")).strip()
    if explicit:
        return normalize_profile_name(explicit)
    if raw.get("allow_live_mouse"):
        return PROFILE_LIVE_DEFAULT
    return PROFILE_APEX_STYLE_DRY_RUN


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load config.json and merge profile defaults (trace profile when unset)."""
    import json as _json

    p = Path(path) if path is not None else Path(__file__).resolve().parent / "config.json"
    raw: dict[str, Any] = {}
    if p.is_file():
        raw = _json.loads(p.read_text(encoding="utf-8"))
    merged = apply_profile(raw)
    try:
        from config_validation import validate_config
        return validate_config(merged)
    except ImportError:
        return merged


def apply_profile(raw: dict[str, Any]) -> dict[str, Any]:
    """Merge profile defaults under user config (user keys win)."""
    profile = resolve_profile_name(raw)
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
