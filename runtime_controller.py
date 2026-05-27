"""Bridge ABA GUI to AssistRuntime (thread-safe start/stop + snapshots)."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

from aba_status import RuntimeSnapshot
from assist import load_config
from process_presence import ProcessPresenceDebouncer

logger = logging.getLogger("aba.controller")

# Patch keys that trigger YOLO / Apex hot-reload handling in apply_config_patch.
_YOLO_TOUCHED_KEYS = frozenset({
    "detection_mode",
    "yolo_weights_path",
    "yolo_yolov5_root",
    "yolo_inference_size",
    "yolo_confidence_min",
    "yolo_iou_thres",
    "yolo_max_det",
    "yolo_grab_width",
    "yolo_grab_height",
    "yolo_use_fp16",
    "yolo_aim_fraction",
    "yolo_device",
    "pull_mode",
    "apexaimbot_pid_x_p",
    "apexaimbot_pid_x_i",
    "apexaimbot_pid_x_d",
    "apexaimbot_pid_y_p",
    "apexaimbot_pid_y_i",
    "apexaimbot_pid_y_d",
    "apexaimbot_min_step",
    "apexaimbot_max_step",
    "apexaimbot_lock_range_x",
    "apexaimbot_lock_range_y",
    "apex_pid_subtick_hz",
    "apexaimbot_recoil_enabled",
    "apexaimbot_recoil_weapon",
    "apexaimbot_sens",
    "apexaimbot_ads_sens",
    "apexaimbot_auto_sens_modifier",
    "apexaimbot_recoil_modifier",
    "apexaimbot_scale_pid_by_modifier",
    "apexaimbot_mouse_modifier",
    "yolo_switch_reset_pixels",
})

_MODE_SUBSYSTEM_KEYS = frozenset({
    "detection_mode",
    "pull_mode",
    "detection_motion_assist",
    "detection_motion_threshold",
})

_RECOIL_ONLY_KEYS = frozenset({
    "apexaimbot_recoil_enabled",
    "apexaimbot_recoil_weapon",
    "apexaimbot_sens",
    "apexaimbot_ads_sens",
    "apexaimbot_auto_sens_modifier",
    "apexaimbot_recoil_modifier",
})

# Keys that require a fresh vendored engine (cache key in apexaimbot_bridge).
_YOLO_ENGINE_TUNE_KEYS = frozenset({
    "yolo_weights_path",
    "yolo_yolov5_root",
    "yolo_inference_size",
    "yolo_confidence_min",
    "yolo_iou_thres",
    "yolo_max_det",
    "yolo_grab_width",
    "yolo_grab_height",
    "yolo_use_fp16",
    "yolo_aim_fraction",
    "yolo_device",
    "apexaimbot_pid_x_p",
    "apexaimbot_pid_x_i",
    "apexaimbot_pid_x_d",
    "apexaimbot_pid_y_p",
    "apexaimbot_pid_y_i",
    "apexaimbot_pid_y_d",
    "apexaimbot_min_step",
    "apexaimbot_max_step",
    "apexaimbot_lock_range_x",
    "apexaimbot_lock_range_y",
    "apexaimbot_mouse_modifier",
    "apexaimbot_recoil_modifier",
    "apexaimbot_scale_pid_by_modifier",
})


def patch_touches_yolo(patch: dict[str, Any]) -> bool:
    return any(k in patch for k in _YOLO_TOUCHED_KEYS)


def should_reload_yolo_engine(
    patch: dict[str, Any],
    merged: dict[str, Any],
    *,
    full_replace: bool = False,
) -> bool:
    """True when live YOLO mode must reload the vendored engine from merged cfg."""
    from profiles import is_yolo_detection

    if not is_yolo_detection(merged):
        return False
    if full_replace:
        return True
    if any(k in patch for k in _YOLO_ENGINE_TUNE_KEYS):
        return True
    return False


class RuntimeController:
    def __init__(self, config: dict[str, Any], config_path: Path) -> None:
        self.config_path = Path(config_path)
        self._config = dict(config)
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._runtime: Any = None
        self._error = ""
        self._stop_warning = ""
        self._benchmark_summary = ""
        self.process_debounce = ProcessPresenceDebouncer()

    @property
    def is_running(self) -> bool:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return True
            if self._runtime is not None:
                return bool(getattr(self._runtime, "running", False))
            return False

    def reload_config(self) -> dict[str, Any]:
        cfg = load_config(self.config_path)
        with self._lock:
            self._config = cfg
        return cfg

    def _write_config_disk(self, cfg: dict[str, Any]) -> None:
        self.config_path.write_text(
            json.dumps(cfg, indent=2) + "\n",
            encoding="utf-8",
        )

    def save_config(self, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
        """Persist + hot-apply (same subsystem refresh as slider patches)."""
        data = dict(cfg if cfg is not None else self._config)
        return self.apply_config_patch(data, persist=True, full_replace=True)

    def apply_config_patch(
        self,
        patch: dict[str, Any],
        *,
        persist: bool = True,
        full_replace: bool = False,
    ) -> dict[str, Any]:
        with self._lock:
            if full_replace:
                merged = dict(patch)
            else:
                merged = dict(self._config)
                merged.update(patch)
        from config_pipeline import normalize_app_config

        merged = normalize_app_config(merged)
        with self._lock:
            self._config = merged
        if persist:
            self._write_config_disk(merged)
        live = self._runtime
        if live is not None and getattr(live, "running", False):
            live.config = merged
            if hasattr(live, "_aim_tracker"):
                live._aim_tracker.configure_prediction(
                    bool(merged.get("prediction_enabled", True)),
                    float(merged.get("prediction_lead_seconds", 0.03)),
                    float(merged.get("prediction_max_pixels", 20)),
                    vertical_cap_pixels=float(merged.get("prediction_vertical_cap_pixels", 4.0)),
                )
                live._aim_tracker.configure_body_clamp(
                    float(merged.get("aim_body_y_min_fraction", 0.28)),
                    float(merged.get("aim_body_y_max_fraction", 0.52)),
                )
                live._aim_tracker.configure_smoothing_tau(
                    float(merged.get("smoothing_tau_still", 0.062)),
                    float(merged.get("smoothing_tau_moving", 0.028)),
                )
                live._aim_tracker.configure_overlay_dot_alpha(
                    float(merged.get("overlay_dot_smooth_alpha", 0.52))
                )
            # R1 (audit): propagate detection-motion settings to the live
            # detection context. Without this, toggling motion_assist or
            # motion_threshold in the GUI required a full Stop -> Start to
            # take effect; users could not iterate on these settings live.
            if hasattr(live, "_detect_ctx") and live._detect_ctx is not None:
                live._detect_ctx.motion_assist = bool(
                    merged.get("detection_motion_assist", True)
                )
                live._detect_ctx.motion_threshold = int(
                    merged.get("detection_motion_threshold", 10)
                )
            if "mouse_backend" in patch and not live._dry:
                from mouse_io import create_mouse_backend

                live._mouse = create_mouse_backend(str(merged.get("mouse_backend", "auto")))
            yolo_touched = patch_touches_yolo(patch)
            if any(k in patch for k in _MODE_SUBSYSTEM_KEYS) and hasattr(
                live, "sync_config_subsystems"
            ):
                live.sync_config_subsystems(merged)
                if not live._dry and (
                    "detection_mode" in patch or "pull_mode" in patch
                ):
                    from mouse_io import create_mouse_backend

                    live._mouse = create_mouse_backend(
                        str(merged.get("mouse_backend", "auto"))
                    )
            if "capture_fps" in patch or full_replace:
                from profiles import effective_capture_fps

                fps = effective_capture_fps(merged)
                live._configured_fps = fps
                if getattr(live, "_stats", None) is not None:
                    live._stats.configured_fps = max(1, int(fps))
            if yolo_touched:
                from yolo_targeting import reload_yolo_engine

                if should_reload_yolo_engine(patch, merged, full_replace=full_replace):
                    live._yolo_engine = reload_yolo_engine(merged)
                if hasattr(live, "_reset_apex_aim_state"):
                    live._reset_apex_aim_state()
            if any(k in patch for k in _RECOIL_ONLY_KEYS) and hasattr(
                live, "_ensure_apex_recoil"
            ):
                from profiles import is_yolo_detection, uses_apex_pid_pull

                if is_yolo_detection(merged) and uses_apex_pid_pull(merged):
                    live._ensure_apex_recoil(merged)
            if hasattr(live, "_pull") and live._pull is not None:
                live._pull.update_tuning(
                    pull_strength=float(merged.get("pull_strength", 0.82)),
                    deadzone=float(merged.get("deadzone_pixels", 3)),
                    max_speed=float(merged.get("max_pull_speed_pixels_per_frame", 22)),
                    magnetism_radius=float(merged.get("magnetism_radius_pixels", 80)),
                    velocity_smoothing=float(merged.get("velocity_smoothing", 0.5)),
                    humanize_amplitude=float(merged.get("humanize_amplitude_pixels", 0.0)),
                    humanize_jerk_limit=float(merged.get("humanize_jerk_limit", 2.5)),
                    humanize_enabled=bool(merged.get("humanize_enabled", False)),
                    recoil_compensation_enabled=bool(
                        merged.get("recoil_compensation_enabled", False)
                    ),
                    recoil_pull_down_pixels_per_second=float(
                        merged.get("recoil_pull_down_pixels_per_second", 0.0)
                    ),
                    jitter_enabled=bool(merged.get("jitter_enabled", False)),
                    jitter_amplitude_pixels=float(merged.get("jitter_amplitude_pixels", 0.0)),
                    jitter_frequency_hz=float(merged.get("jitter_frequency_hz", 6.0)),
                    # PHASE-7 AUDIT FIX (MED9): three additional tuning
                    # fields that were previously only applied on a full
                    # Start/Stop cycle. magnetism_min_pull_scale and
                    # fov_edge_min_pull_scale set the floor for how
                    # gently the pull behaves near the screen edge /
                    # outside magnetism radius; smoothing_curve picks
                    # the easing function. Without hot-reload, GUI
                    # changes to these were silent until the user
                    # remembered to Stop->Start.
                    magnetism_min_scale=float(
                        merged.get("magnetism_min_pull_scale", 0.35)
                    ),
                    fov_edge_min_scale=float(
                        merged.get("fov_edge_min_pull_scale", 0.85)
                    ),
                    smoothing_curve=str(merged.get("smoothing_curve", "linear")),
                    stale_grace_frames=int(
                        merged.get("mouse_gate_stale_grace_frames", 12)
                    ),
                )
        # PHASE-7 AUDIT FIX (MED11): hot-apply verbose_logging changes
        # so the user can flip the toggle without restarting the
        # runtime. Without this, ``cfg["verbose_logging"] = True``
        # was stored but the root logger level never changed and the
        # detector debug lines stayed silenced.
        if "verbose_logging" in patch:
            level = logging.DEBUG if bool(merged.get("verbose_logging", False)) else logging.INFO
            logging.getLogger().setLevel(level)
            logging.getLogger("aba").setLevel(level)
        if live is not None and getattr(live, "running", False):
            if any(
                k in patch
                for k in (
                    "fov_radius_pixels",
                    "fov_radius_ads_pixels",
                    "unified_fov",
                    "detection_fov_margin_pixels",
                )
            ):
                from profiles import effective_overlay_fov_radius

                overlay_fov = int(effective_overlay_fov_radius(merged))
                if getattr(live, "_overlay", None) is not None:
                    try:
                        import mss

                        mon = mss.mss().monitors[
                            int(merged.get("monitor_index", 1))
                        ]
                        cx = mon["width"] / 2.0 + float(
                            merged.get("crosshair_offset_x", 0.0)
                        )
                        cy = mon["height"] / 2.0 + float(
                            merged.get("crosshair_offset_y", 0.0)
                        )
                        live._overlay.update_fov(
                            overlay_fov, False, cx, cy
                        )
                    except Exception:
                        logger.exception("hot-reload FOV failed")
                live._last_display_fov = overlay_fov
                live._last_fov_radius = -1
            if any(k in patch for k in ("crosshair_offset_x", "crosshair_offset_y")):
                if getattr(live, "_overlay", None) is not None:
                    try:
                        import mss

                        mon = mss.mss().monitors[
                            int(merged.get("monitor_index", 1))
                        ]
                        cx = mon["width"] / 2.0 + float(
                            merged.get("crosshair_offset_x", 0.0)
                        )
                        cy = mon["height"] / 2.0 + float(
                            merged.get("crosshair_offset_y", 0.0)
                        )
                        from profiles import effective_fov_radius

                        ads_active = False
                        if getattr(live, "_ads", None) is not None:
                            ads_active = bool(live._ads.is_ads_active())
                        fov_r = int(
                            effective_fov_radius(merged, ads_active=ads_active)
                        )
                        live._overlay.update_fov(fov_r, ads_active, cx, cy)
                        live._last_fov_radius = -1
                    except Exception:
                        logger.exception("hot-reload crosshair center failed")
            if "overlay_dot_smooth_alpha" in patch and getattr(live, "_overlay", None) is not None:
                live._overlay.set_dot_glide_alpha(
                    float(merged.get("overlay_dot_smooth_alpha", 0.52))
                )
        return merged

    def set_benchmark_summary(self, text: str) -> None:
        with self._lock:
            self._benchmark_summary = text
        rt = self._runtime
        if rt is not None:
            rt.set_benchmark_summary(text)

    def start(self) -> tuple[bool, str]:
        if self.is_running:
            return False, "Runtime already running."
        try:
            cfg = self.reload_config()
        except Exception as exc:
            self._error = str(exc)
            return False, f"Config load failed: {exc}"

        self._error = ""
        self._stop_warning = ""

        def _on_stop() -> None:
            self.stop()

        try:
            from runtime import AssistRuntime

            runtime = AssistRuntime(
                cfg,
                self.config_path,
                process_debounce=self.process_debounce,
                on_external_stop=_on_stop,
            )
        except Exception as exc:
            self._error = str(exc)
            logger.exception("AssistRuntime init failed")
            return False, f"Runtime init failed: {exc}"

        def _run() -> None:
            try:
                runtime.run()
            except Exception as exc:
                with self._lock:
                    self._error = str(exc)
                logger.exception("AssistRuntime crashed")

        with self._lock:
            self._runtime = runtime
            self._thread = threading.Thread(target=_run, name="ABA-Runtime", daemon=True)
            self._thread.start()
        return True, "Runtime started — hold RMB (ADS) when live; dry-run simulates pull."

    def stop(self) -> None:
        rt = None
        with self._lock:
            rt = self._runtime
        if rt is not None:
            rt.stop()
        th = None
        with self._lock:
            th = self._thread
        if th is not None and th.is_alive():
            th.join(timeout=8.0)
            if th.is_alive():
                with self._lock:
                    self._stop_warning = "Runtime thread did not exit within 8s."
        with self._lock:
            self._thread = None

    def request_debug_frame_save(self) -> None:
        rt = self._runtime
        if rt is not None and hasattr(rt, "request_debug_frame_save"):
            rt.request_debug_frame_save()

    def snapshot(self) -> RuntimeSnapshot:
        with self._lock:
            err = self._error
            warn = self._stop_warning
            bench = self._benchmark_summary
            rt = self._runtime
            th = self._thread
            cfg = dict(self._config)

        thread_alive = th is not None and th.is_alive()
        telem: dict[str, Any] = {}
        if rt is not None:
            try:
                telem = rt.get_telemetry()
            except Exception:
                telem = {}

        def _f(key: str, default: float = -1.0) -> float:
            v = telem.get(key, default)
            try:
                return float(v)
            except (TypeError, ValueError):
                return default

        def _i(key: str, default: int = -1) -> int:
            v = telem.get(key, default)
            try:
                return int(v)
            except (TypeError, ValueError):
                return default

        return RuntimeSnapshot(
            running=bool(telem.get("running", False)),
            stopping=bool(telem.get("stopping", False)),
            paused=bool(telem.get("paused", False)),
            ads=bool(telem.get("ads", False)),
            has_target=bool(telem.get("has_target", False)),
            frame_has_target=bool(telem.get("frame_has_target", False)),
            locked=bool(telem.get("locked", False)),
            stats_valid=bool(telem.get("stats_valid", False)),
            live_input_enabled=bool(telem.get("live_input_enabled", False)),
            dry_run=bool(telem.get("dry_run", True)),
            fps=_f("fps"),
            frame_ms=_f("frame_ms"),
            configured_capture_fps=_i("configured_capture_fps", int(cfg.get("capture_fps", -1))),
            dropped_frames=_i("dropped_frames"),
            stale_frames=_i("stale_frames"),
            confidence=_f("confidence"),
            pull_px=_f("pull_px"),
            simulated_pull_px=_f("simulated_pull_px"),
            pull_strength=_f("pull_strength"),
            distance_px=_f("distance_px"),
            candidates=_i("candidates"),
            mouse_backend=str(telem.get("mouse_backend", "")),
            capture_size=str(telem.get("capture_size", "")),
            hooks_enabled=bool(telem.get("hooks_enabled", False)),
            kill_switch_active=bool(telem.get("kill_switch_active", False)),
            ads_input_mode=str(telem.get("ads_input_mode", cfg.get("ads_input_mode", ""))),
            process_detection="presence-only (weak)",
            last_gate_block=str(telem.get("last_gate_block", "")),
            benchmark_summary=bench or str(telem.get("benchmark_summary", "")),
            thread_alive=thread_alive,
            mouse_armed=bool(telem.get("mouse_armed", False)),
            error_message=err,
            stop_warning=warn,
            body_shape_score=_f("body_shape_score"),
            head_score=_f("head_score"),
            torso_score=_f("torso_score"),
            limb_stack_score=_f("limb_stack_score"),
            reject_reason=str(telem.get("reject_reason", "")),
            anchor_x=_f("anchor_x"),
            anchor_y=_f("anchor_y"),
            bbox_x=_i("bbox_x"),
            bbox_y=_i("bbox_y"),
            bbox_w=_i("bbox_w"),
            bbox_h=_i("bbox_h"),
            capture_ms=_f("capture_ms"),
            detect_ms=_f("detect_ms"),
            motion_lag_ms=_f("motion_lag_ms"),
            pull_dx=_f("pull_dx"),
            pull_dy=_f("pull_dy"),
            mouse_gate_allowed=bool(telem.get("mouse_gate_allowed", False)),
            detection_fresh=bool(telem.get("detection_fresh", False)),
        )
