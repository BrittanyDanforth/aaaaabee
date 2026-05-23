"""Validate and normalize OverlayAssist config.json."""

from __future__ import annotations

from typing import Any

from profiles import normalize_profile_name

REQUIRED_KEYS = frozenset(
    {
        "fov_radius_pixels",
        "max_pull_speed_pixels_per_frame",
    }
)

SMOOTHING_CURVES = frozenset({"linear", "ease_out", "ease_in_out"})


class ConfigError(ValueError):
    """Raised when config.json is missing required fields or has invalid values."""


def _require_number(
    data: dict[str, Any],
    key: str,
    *,
    default: float | None = None,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if key not in data:
        if default is None:
            raise ConfigError(f"Missing required key: {key}")
        value = float(default)
    else:
        try:
            value = float(data[key])
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{key} must be a number") from exc
    if minimum is not None and value < minimum:
        raise ConfigError(f"{key} must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{key} must be <= {maximum}, got {value}")
    return value


def _validate_hsv_ranges(ranges: Any) -> list[dict[str, list[int]]]:
    if not isinstance(ranges, list) or not ranges:
        raise ConfigError("hsv_ranges must be a non-empty list")
    out: list[dict[str, list[int]]] = []
    for i, entry in enumerate(ranges):
        if not isinstance(entry, dict):
            raise ConfigError(f"hsv_ranges[{i}] must be an object")
        lower = entry.get("lower")
        upper = entry.get("upper")
        if not (isinstance(lower, list) and isinstance(upper, list) and len(lower) == 3 and len(upper) == 3):
            raise ConfigError(f"hsv_ranges[{i}] needs lower/upper arrays of length 3")
        lo = [int(v) for v in lower]
        hi = [int(v) for v in upper]
        if any(v < 0 or v > 255 for v in lo + hi):
            raise ConfigError(f"hsv_ranges[{i}] S/V channels must be 0–255")
        if lo[0] < 0 or hi[0] > 179:
            raise ConfigError(f"hsv_ranges[{i}] H channel must be 0–179 (OpenCV)")
        out.append({"lower": lo, "upper": hi})
    return out


def validate_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Return a normalized config dict; raises ConfigError on invalid input."""
    if not isinstance(raw, dict):
        raise ConfigError("Config root must be a JSON object")

    missing = REQUIRED_KEYS - raw.keys()
    if missing:
        raise ConfigError(f"Missing required keys: {', '.join(sorted(missing))}")

    cfg: dict[str, Any] = dict(raw)
    mode = str(cfg.get("detection_mode", "apex")).strip().lower()
    if mode not in ("shape", "hsv", "hybrid", "apex"):
        raise ConfigError("detection_mode must be apex, shape, hsv, or hybrid")
    cfg["detection_mode"] = mode
    if mode in ("hsv", "hybrid"):
        cfg["hsv_ranges"] = _validate_hsv_ranges(cfg.get("hsv_ranges"))
    else:
        cfg["hsv_ranges"] = cfg.get("hsv_ranges") or []

    cfg["fov_radius_pixels"] = int(_require_number(cfg, "fov_radius_pixels", minimum=1))
    cfg["max_pull_speed_pixels_per_frame"] = _require_number(
        cfg, "max_pull_speed_pixels_per_frame", minimum=0.1, maximum=80.0
    )
    cfg["pull_strength"] = _require_number(
        cfg, "pull_strength", default=0.35, minimum=0.0, maximum=2.0
    )
    cfg["deadzone_pixels"] = _require_number(cfg, "deadzone_pixels", default=5.0, minimum=0.0)
    cfg["velocity_smoothing"] = _require_number(
        cfg, "velocity_smoothing", default=0.18, minimum=0.01, maximum=1.0
    )

    curve = str(cfg.get("smoothing_curve", "ease_out")).lower()
    if curve not in SMOOTHING_CURVES:
        raise ConfigError(
            f"smoothing_curve must be one of {sorted(SMOOTHING_CURVES)}, got {curve!r}"
        )
    cfg["smoothing_curve"] = curve

    cfg["magnetism_radius_pixels"] = _require_number(
        cfg, "magnetism_radius_pixels", default=70.0, minimum=0.0
    )
    cfg["magnetism_min_pull_scale"] = _require_number(
        cfg, "magnetism_min_pull_scale", default=0.35, minimum=0.0, maximum=1.0
    )
    cfg["fov_edge_min_pull_scale"] = _require_number(
        cfg, "fov_edge_min_pull_scale", default=0.55, minimum=0.0, maximum=1.0
    )

    cfg["prediction_enabled"] = bool(cfg.get("prediction_enabled", True))
    cfg["prediction_lead_seconds"] = _require_number(
        cfg, "prediction_lead_seconds", default=0.04, minimum=0.0, maximum=0.5
    )
    cfg["prediction_max_pixels"] = _require_number(
        cfg, "prediction_max_pixels", default=24.0, minimum=0.0
    )
    cfg["prediction_vertical_cap_pixels"] = _require_number(
        cfg, "prediction_vertical_cap_pixels", default=4.0, minimum=0.0, maximum=24.0
    )

    cfg["humanize_enabled"] = bool(cfg.get("humanize_enabled", True))
    cfg["humanize_amplitude_pixels"] = _require_number(
        cfg, "humanize_amplitude_pixels", default=0.35, minimum=0.0, maximum=3.0
    )
    cfg["humanize_jerk_limit"] = _require_number(
        cfg, "humanize_jerk_limit", default=2.5, minimum=0.0
    )

    # Engagement-gated recoil compensator (only fires while LMB held + locked).
    # Defaults are OFF / zero so adding the keys does not change existing
    # tracking behaviour for users on older profiles.
    cfg["recoil_compensation_enabled"] = bool(cfg.get("recoil_compensation_enabled", False))
    cfg["recoil_pull_down_pixels_per_second"] = _require_number(
        cfg, "recoil_pull_down_pixels_per_second", default=0.0, minimum=0.0, maximum=180.0
    )
    cfg["jitter_enabled"] = bool(cfg.get("jitter_enabled", False))
    cfg["jitter_amplitude_pixels"] = _require_number(
        cfg, "jitter_amplitude_pixels", default=0.0, minimum=0.0, maximum=6.0
    )
    cfg["jitter_frequency_hz"] = _require_number(
        cfg, "jitter_frequency_hz", default=6.0, minimum=0.5, maximum=20.0
    )

    cfg["target_stickiness_pixels"] = _require_number(
        cfg, "target_stickiness_pixels", default=45.0, minimum=0.0
    )
    cfg["target_lost_frames_before_unlock"] = int(
        _require_number(cfg, "target_lost_frames_before_unlock", default=8.0, minimum=1)
    )
    cfg["distance_score_weight"] = _require_number(
        cfg, "distance_score_weight", default=2.0, minimum=0.0
    )
    cfg["area_score_weight"] = _require_number(cfg, "area_score_weight", default=0.02, minimum=0.0)
    cfg["min_target_area_pixels"] = _require_number(
        cfg, "min_target_area_pixels", default=60.0, minimum=1.0
    )
    cfg["capture_fps"] = int(
        _require_number(cfg, "capture_fps", default=60.0, minimum=1, maximum=240)
    )
    cfg["monitor_index"] = int(_require_number(cfg, "monitor_index", default=1.0, minimum=1))
    cfg["crosshair_offset_x"] = _require_number(cfg, "crosshair_offset_x", default=0.0)
    cfg["crosshair_offset_y"] = _require_number(cfg, "crosshair_offset_y", default=0.0)

    kill = cfg.get("kill_switch_key", "f8")
    if not isinstance(kill, str) or not kill.strip():
        raise ConfigError("kill_switch_key must be a non-empty string")
    kill_norm = kill.strip().lower()
    if kill_norm.startswith("key."):
        kill_norm = kill_norm[4:]
    if kill_norm.startswith("<") and kill_norm.endswith(">"):
        kill_norm = kill_norm[1:-1]
    allowed_kill = frozenset(
        {f"f{i}" for i in range(1, 13)}
        | {"esc", "pause", "delete", "insert", "home", "end", "page_up", "page_down"}
    )
    if kill_norm not in allowed_kill:
        raise ConfigError(
            f"kill_switch_key must be a function key (f1-f12) or esc/pause, got {kill!r}"
        )
    cfg["kill_switch_key"] = kill_norm

    cfg["show_debug_window"] = bool(cfg.get("show_debug_window", False))
    cfg["enable_overlay"] = bool(cfg.get("enable_overlay", False))

    cfg["capture_fov_crop"] = bool(cfg.get("capture_fov_crop", True))
    cfg["capture_crop_padding"] = _require_number(
        cfg, "capture_crop_padding", default=1.35, minimum=1.0, maximum=3.0
    )
    cfg["humanoid_min_height_pixels"] = _require_number(
        cfg, "humanoid_min_height_pixels", default=28.0, minimum=0.0
    )
    cfg["humanoid_min_aspect"] = _require_number(
        cfg, "humanoid_min_aspect", default=0.85, minimum=0.0
    )
    cfg["humanoid_max_aspect"] = _require_number(
        cfg, "humanoid_max_aspect", default=4.5, minimum=0.0
    )

    if cfg["magnetism_radius_pixels"] > cfg["fov_radius_pixels"]:
        raise ConfigError("magnetism_radius_pixels cannot exceed fov_radius_pixels")

    mouse_backend = str(cfg.get("mouse_backend", "auto")).lower()
    if mouse_backend not in ("auto", "pynput", "win32_sendinput", "recording"):
        raise ConfigError("mouse_backend must be auto, pynput, win32_sendinput, or recording")
    cfg["mouse_backend"] = mouse_backend

    ads_mode = str(cfg.get("ads_input_mode", "both")).lower()
    if ads_mode not in ("pynput", "win32_poll", "both", "disabled"):
        raise ConfigError("ads_input_mode must be pynput, win32_poll, both, or disabled")
    cfg["ads_input_mode"] = ads_mode

    cfg["humanoid_min_solidity"] = _require_number(
        cfg, "humanoid_min_solidity", default=0.25, minimum=0.0, maximum=1.0
    )
    cfg["torso_aim_fraction"] = _require_number(
        cfg, "torso_aim_fraction", default=0.38, minimum=0.2, maximum=0.55
    )
    cfg["aim_body_y_min_fraction"] = _require_number(
        cfg, "aim_body_y_min_fraction", default=0.28, minimum=0.15, maximum=0.45
    )
    cfg["aim_body_y_max_fraction"] = _require_number(
        cfg, "aim_body_y_max_fraction", default=0.52, minimum=0.35, maximum=0.65
    )
    if cfg["aim_body_y_min_fraction"] >= cfg["aim_body_y_max_fraction"]:
        raise ConfigError("aim_body_y_min_fraction must be < aim_body_y_max_fraction")

    cfg["smoothing_tau_still"] = _require_number(
        cfg, "smoothing_tau_still", default=0.042, minimum=0.01, maximum=0.25
    )
    cfg["smoothing_tau_moving"] = _require_number(
        cfg, "smoothing_tau_moving", default=0.018, minimum=0.005, maximum=0.15
    )
    cfg["detection_motion_assist"] = bool(cfg.get("detection_motion_assist", True))
    cfg["detection_motion_threshold"] = int(
        _require_number(
            cfg, "detection_motion_threshold", default=10.0, minimum=4.0, maximum=80.0
        )
    )
    cfg["body_shape_min_score"] = _require_number(
        cfg, "body_shape_min_score", default=0.40, minimum=0.2, maximum=0.85
    )
    cfg["head_score_weight"] = _require_number(
        cfg, "head_score_weight", default=0.26, minimum=0.0, maximum=1.0
    )
    cfg["torso_score_weight"] = _require_number(
        cfg, "torso_score_weight", default=0.26, minimum=0.0, maximum=1.0
    )
    cfg["limb_stack_score_weight"] = _require_number(
        cfg, "limb_stack_score_weight", default=0.22, minimum=0.0, maximum=1.0
    )
    cfg["mouse_gate_stale_grace_frames"] = int(
        _require_number(cfg, "mouse_gate_stale_grace_frames", default=12.0, minimum=0)
    )
    cfg["mouse_gate_pull_budget_scale"] = _require_number(
        cfg, "mouse_gate_pull_budget_scale", default=3.5, minimum=1.0, maximum=8.0
    )

    # PHASE-5 AUDIT FIX (D-LOW dead debug flags): the seven
    # ``debug_show_body_bbox`` / ``debug_show_anchor`` / etc. flags were
    # defined and persisted by validation/profiles but never read by
    # ``draw_debug``. Dead config is worse than no config — drop them.
    # ``debug_show_detect_ring`` IS read by draw_debug and runtime.py
    # to gate the optional second FOV ring, so it stays.
    for flag in (
        "debug_show_detect_ring",
    ):
        cfg[flag] = bool(cfg.get(flag, False))
    # Strip dead debug_show_* keys if a legacy profile leaked them in.
    for stale in (
        "debug_show_body_bbox",
        "debug_show_anchor",
        "debug_show_rejected",
        "debug_show_top_candidates",
        "debug_show_reject_reasons",
        "debug_show_mask_overlay",
        "debug_show_timing",
    ):
        cfg.pop(stale, None)

    cfg["stats_log_interval_frames"] = int(
        _require_number(cfg, "stats_log_interval_frames", default=60.0, minimum=1)
    )
    cfg["verbose_logging"] = bool(cfg.get("verbose_logging", False))
    log_file = cfg.get("log_file", "")
    cfg["log_file"] = str(log_file).strip() if log_file else ""

    title = cfg.get("target_window_title", "")
    cfg["target_window_title"] = str(title).strip() if title else ""
    cfg["pause_on_target_closed"] = bool(cfg.get("pause_on_target_closed", True))

    raw_profile = str(cfg.get("profile", "apex_style_dry_run")).lower()
    profile = "custom" if raw_profile == "custom" else normalize_profile_name(raw_profile)
    cfg["profile"] = profile
    cfg["dry_run_force_detect"] = bool(cfg.get("dry_run_force_detect", False))

    proc = cfg.get("target_process_name", "")
    cfg["target_process_name"] = str(proc).strip() if proc else ""
    cfg["target_process_required"] = bool(cfg.get("target_process_required", False))

    cfg["offline_dev_mode"] = bool(cfg.get("offline_dev_mode", True))
    cfg["allow_live_mouse"] = bool(cfg.get("allow_live_mouse", False))
    if cfg["allow_live_mouse"] and not cfg["offline_dev_mode"]:
        raise ConfigError("allow_live_mouse requires offline_dev_mode: true")

    debug_dir = cfg.get("debug_frames_dir", "artifacts/debug_frames")
    cfg["debug_frames_dir"] = str(debug_dir).strip() or "artifacts/debug_frames"

    if cfg["log_file"]:
        from path_utils import resolve_log_path

        cfg["log_file"] = resolve_log_path(cfg["log_file"])

    return cfg
