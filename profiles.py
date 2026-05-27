"""Reference profiles — Apex safe-by-default; owned dev requires explicit opt-in."""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Apex Legends PC executables (DX11 + DX12). Process presence only — no injection.
APEX_PROCESS_NAME = "r5apex.exe,r5apex_dx12.exe"

# Shared Apex tuning (shape detection default; optional HSV legacy).
_APEX_TUNING: dict[str, Any] = {
    "target_process_name": APEX_PROCESS_NAME,
    "target_process_required": False,
    "pause_on_target_closed": True,
    "offline_dev_mode": True,
    "hsv_ranges": [
        {"lower": [0, 80, 80], "upper": [15, 255, 255]},
        {"lower": [165, 80, 80], "upper": [179, 255, 255]},
        {"lower": [0, 150, 150], "upper": [25, 255, 255]},
    ],
    "fov_radius_pixels": 140,
    "fov_radius_ads_pixels": 185,
    # Single user-visible FOV: overlay ring, detection mask, pull clamp, and
    # capture crop all use effective_fov_radius() when unified_fov is True.
    # Set unified_fov=False and detection_fov_margin_pixels>0 only if you
    # need a hidden detection halo beyond the drawn ring.
    "unified_fov": True,
    # HUD ring stays hip-fire size/color on ADS (no second cyan/larger ring).
    "overlay_ring_fixed_hip": True,
    "detection_fov_margin_pixels": 0,
    "capture_fov_crop": True,
    "capture_crop_padding": 1.34,
    "max_pull_speed_pixels_per_frame": 22.0,
    "pull_strength": 0.82,
    "deadzone_pixels": 3,
    "velocity_smoothing": 0.50,
    "smoothing_curve": "ease_out",
    "magnetism_radius_pixels": 65,
    "magnetism_min_pull_scale": 0.70,
    "fov_edge_min_pull_scale": 0.88,
    "prediction_enabled": True,
    "prediction_lead_seconds": 0.034,
    "prediction_max_pixels": 12,
    "humanize_enabled": True,
    "humanize_amplitude_pixels": 0.20,
    "humanize_jerk_limit": 10.0,
    # Engagement-gated recoil compensator + jitter — defaults OFF on every
    # profile so existing tracking behaviour is preserved.  The Strong preset
    # in aba_gui.py is the only built-in shortcut that turns these on.
    "recoil_compensation_enabled": False,
    "recoil_pull_down_pixels_per_second": 0.0,
    "jitter_enabled": False,
    "jitter_amplitude_pixels": 0.0,
    "jitter_frequency_hz": 6.0,
    "target_stickiness_pixels": 60,
    "target_lost_frames_before_unlock": 18,
    "new_lock_confirm_frames": 2,
    "distance_score_weight": 2.6,
    "area_score_weight": 0.015,
    "min_target_area_pixels": 20,
    "viewmodel_exclude_bottom_frac": 0.28,
    # PHASE-7 AUDIT FIX (HIGH5): bumped from 16 → 40 — the previous 16-px
    # floor let sky/HUD 16-px fragments enter the candidate pool. The
    # Tracking preset overlays 60 (more aggressive) but profile defaults
    # need a sane mid-ground for users who never click the preset.
    "humanoid_min_height_pixels": 48,
    "humanoid_min_aspect": 0.5,
    "humanoid_max_aspect": 5.5,
    # PHASE-7 AUDIT FIX (HIGH6): lowered from 0.15 → 0.10. img2 in the
    # static audit set has body=1.00 at solidity=0.14; 0.15 dropped a
    # real body. The downstream ``_body_structure_reject`` and
    # ``body_shape_min_score`` together already gate humanoid shape; we
    # do not need this threshold to also be a brittle hard cutoff.
    "humanoid_min_solidity": 0.10,
    "torso_aim_fraction": 0.40,
    "aim_body_y_min_fraction": 0.28,
    "aim_body_y_max_fraction": 0.52,
    "prediction_vertical_cap_pixels": 4.0,
    "smoothing_tau_still": 0.042,
    "smoothing_tau_moving": 0.018,
    "body_shape_min_score": 0.40,
    "detection_mode": "apex",
    # YOLO primary when detection_mode=yolo; optional CV fusion when yolo_assist_enabled.
    "target_selection_mode": "apex",
    "yolo_assist_enabled": False,
    "yolo_weights_path": "",
    "yolo_yolov5_root": "third_party/apexaimbot",
    "yolo_grab_width": 416,
    "yolo_grab_height": 416,
    "pull_mode": "aba",
    "apexaimbot_pid_x_p": 0.36,
    "apexaimbot_pid_x_i": 0.032,
    "apexaimbot_pid_x_d": 0.01,
    "apexaimbot_pid_y_p": 0.2,
    "apexaimbot_pid_y_i": 0.0,
    "apexaimbot_pid_y_d": 0.0,
    "apexaimbot_min_step": 10,
    "apexaimbot_max_step": 6,
    "apexaimbot_lock_range_x": 1.0,
    "apexaimbot_lock_range_y": 0.5,
    "yolo_inference_size": 416,
    "yolo_confidence_min": 0.5,
    "yolo_iou_thres": 0.25,
    "yolo_max_det": 12,
    "yolo_target_pick": "nearest",
    "yolo_aim_fraction": 0.2,
    "yolo_exclude_labels": ["teammate"],
    "yolo_use_fp16": False,
    "yolo_fusion_boost": 0.30,
    "yolo_fusion_min_iou": 0.28,
    "yolo_device": "auto",
    "detection_motion_assist": True,
    "detection_motion_threshold": 10,
    "head_score_weight": 0.26,
    "torso_score_weight": 0.26,
    "limb_stack_score_weight": 0.22,
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
        # Pull sub-tick (see runtime._run_pull_subticks).  At
        # capture_fps=30 each detect frame is 33 ms wide, which is
        # exactly the cadence the user reported as "feels chunky";
        # 120 Hz sub-tick gives 4 mouse-moves between captures so
        # the cursor receives a smooth corrective stream instead of
        # one big jump per detect frame.
        "pull_subtick_hz": 120,
    },
    PROFILE_APEX_STYLE_LIVE_TRACE: {
        **_APEX_TUNING,
        "profile": PROFILE_APEX_STYLE_LIVE_TRACE,
        "allow_live_mouse": True,
        "enable_overlay": True,
        "capture_fps": 60,
        "overlay_fps": 90,
        "overlay_dot_smooth_alpha": 0.58,
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
        "fov_edge_min_pull_scale": 0.92,
        "max_pull_speed_pixels_per_frame": 30.0,
        "pull_strength": 0.88,
        "deadzone_pixels": 2,
        "torso_aim_fraction": 0.40,
        "aim_body_y_min_fraction": 0.28,
        "aim_body_y_max_fraction": 0.50,
        "smoothing_tau_still": 0.034,
        "smoothing_tau_moving": 0.014,
        "velocity_smoothing": 0.42,
        "detection_motion_assist": True,
        "detection_motion_threshold": 9,
        "body_shape_min_score": 0.40,
        "mouse_gate_stale_grace_frames": 14,
        # Sub-tick pull at 180 Hz so mouse motion stays continuous when
        # detect frame cost spikes above the 16.67 ms (60 fps) budget.
        # See runtime.py PULL-SUBTICK block for the extrapolation rules
        # (vy bounded ±1 px/subtick, vx bounded ±2.5 px/subtick, anchor
        # always clamped inside the chest band).  0 disables.
        "pull_subtick_hz": 180,
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
        # Higher capture rate → smaller sub-tick budget per frame;
        # 240 Hz keeps ~2-3 sub-ticks per 11 ms detect frame.
        "pull_subtick_hz": 240,
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


def effective_overlay_fps(config: dict[str, Any]) -> int:
    """Tk overlay redraw rate (FOV ring + target dot). Defaults to 90 Hz."""
    fps = int(config.get("overlay_fps", 90))
    capture = effective_capture_fps(config)
    # Never redraw slower than capture — dot coords only change each capture frame.
    return max(30, min(144, max(fps, capture)))


def effective_capture_fps(config: dict[str, Any]) -> int:
    """Configured FPS after profile + dry-run safety cap."""
    fps = int(config.get("capture_fps", 15))
    profile = resolve_profile_name(config)
    if profile == PROFILE_APEX_STYLE_LIVE_TRACE:
        return min(fps, 120)
    if profile in (PROFILE_APEX_STYLE_DRY_RUN, PROFILE_APEX_STYLE_LIVE_SAFE):
        return min(fps, 30)
    if not config.get("allow_live_mouse", False):
        if profile == PROFILE_APEX_STYLE_PERF_TEST:
            return min(fps, 120)
        return min(fps, 30)
    return fps


def effective_fov_radius(config: dict[str, Any], *, ads_active: bool) -> int:
    """Assist FOV radius (hip-fire vs ADS) for detection/capture/pull."""
    base = int(config.get("fov_radius_pixels", 140))
    if not ads_active:
        return max(80, base)
    ads = int(config.get("fov_radius_ads_pixels", 0))
    if ads <= 0:
        scale = float(config.get("fov_ads_scale", 1.30))
        ads = int(base * scale)
    return max(base, ads)


def effective_overlay_fov_radius(
    config: dict[str, Any], *, ads_active: bool = False
) -> int:
    """
    Visible HUD FOV ring only.

    When ``overlay_ring_fixed_hip`` is True (default), the drawn ring always
    uses hip-fire ``fov_radius_pixels`` — ADS does not spawn a second/larger
    cyan ring (the common double-FOV bug on RMB).
    """
    if bool(config.get("overlay_ring_fixed_hip", True)):
        return effective_fov_radius(config, ads_active=False)
    return effective_fov_radius(config, ads_active=ads_active)


def effective_detection_fov_radius(config: dict[str, Any], *, ads_active: bool) -> int:
    """
    Detection + pull FOV radius.

    When ``unified_fov`` is True (default), this equals ``effective_fov_radius``
    so the overlay ring, detector mask, pull clamp, and motion FOV all share
    one radius — no phantom second ring or split-brain tuning.

    Legacy split mode (``unified_fov=False``): adds ``detection_fov_margin_pixels``
    or ``detection_fov_margin_scale`` on top of the display radius.
    """
    core = effective_fov_radius(config, ads_active=ads_active)
    if bool(config.get("unified_fov", True)):
        return min(400, max(80, core))
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
