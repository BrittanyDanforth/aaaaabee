"""Main assist loop — testable, Windows-ready."""

from __future__ import annotations

import logging
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import mss

from ban_safety import (
    dry_run_mode,
    hooks_enabled,
    kill_switch_active,
    live_assist_enabled,
    validate_runtime_policy,
)
from capture import build_capture_region, grab_bgr, to_monitor_coords
from detector import DetectionResult, Target, draw_debug, find_best_target
from input_state import AdsInputState
from motion import TargetMotion, TargetTracker
from mouse_gate import MouseGateContext, MouseGateResult, evaluate_mouse_gate
from mouse_io import MouseBackend, create_mouse_backend
from platform_info import enable_dpi_awareness
from process_presence import ProcessPresenceDebouncer
from profiles import effective_capture_fps
from pull import PullController, PullTuning
from stats import RuntimeStats

logger = logging.getLogger("overlay_assist")


class AssistRuntime:
    def __init__(
        self,
        config: dict[str, Any],
        config_path: Path,
        *,
        mouse_backend: MouseBackend | None = None,
        ads_input: AdsInputState | None = None,
        on_external_stop: Any = None,
        process_debounce: ProcessPresenceDebouncer | None = None,
    ) -> None:
        self.config = config
        self.config_path = config_path
        self.running = True
        self._lock = threading.RLock()
        self._on_external_stop = on_external_stop
        self._live = live_assist_enabled(config)
        self._dry = dry_run_mode(config)
        self._force_detect = bool(config.get("dry_run_force_detect", False)) and not self._live
        self._configured_fps = effective_capture_fps(config)
        self._last_gate_block = ""
        self._stale_frames = 0
        self._benchmark_summary = ""

        # Body-column aim smoothing (bbox-aware — required for moving targets)
        self._aim_tracker = TargetTracker()
        self._last_motion: TargetMotion | None = None

        if mouse_backend is not None:
            self._mouse = mouse_backend
        elif self._dry:
            from mouse_io import RecordingMouseBackend

            self._mouse = RecordingMouseBackend()
        else:
            self._mouse = create_mouse_backend(str(config.get("mouse_backend", "auto")))

        if self._dry:
            self._ads = AdsInputState("disabled")
        elif ads_input is not None:
            self._ads = ads_input
        else:
            self._ads = AdsInputState(str(config.get("ads_input_mode", "both")))  # type: ignore[arg-type]

        self._overlay: OverlayWindow | None = None
        self._overlay_thread: threading.Thread | None = None
        self._locked_target: Target | None = None
        self._target_lost_frames = 0
        self._pull: PullController | None = None
        self._stats: RuntimeStats | None = None
        self._paused = False
        self._was_paused = False
        self._mouse_enabled = False
        self._stopping = False
        self._process_debounce = process_debounce or ProcessPresenceDebouncer()
        self._frame_has_target = False
        self._switch_candidate: Target | None = None
        self._switch_frames = 0
        self._trace_frame = 0
        self._trace_pull = False

    def _smooth_aim(
        self,
        target: Target | None,
        time_sec: float,
    ) -> TargetMotion | None:
        """
        Upper-chest body column via observe_target(bbox_*).
        Returns None when no target — overlay/pull must not use raw plate centroids.
        """
        if target is None:
            self._aim_tracker.reset()
            self._last_motion = None
            return None

        motion = self._aim_tracker.observe_target(
            target.centroid_x,
            target.centroid_y,
            time_sec,
            bbox_x=target.bbox_x,
            bbox_y=target.bbox_y,
            bbox_w=target.bbox_w,
            bbox_h=target.bbox_h,
        )
        self._last_motion = motion
        return motion

    @staticmethod
    def _target_for_pull(raw: Target, motion: TargetMotion) -> Target:
        """Pull + overlay use smoothed aim anchor, not hopping red-plate centroids."""
        return replace(raw, centroid_x=motion.x, centroid_y=motion.y)

    def stop(self) -> None:
        with self._lock:
            self.running = False
            self._mouse_enabled = False
            self._stopping = True
        self._ads.clear()
        self._release_ads()
        self._process_debounce.reset()
        with self._lock:
            self._frame_has_target = False
        self._clear_stats()

    def _clear_stats(self) -> None:
        with self._lock:
            if self._stats is not None:
                self._stats = RuntimeStats(self._configured_fps)
            self._stale_frames = 0

    def _should_run(self) -> bool:
        with self._lock:
            return self.running

    def set_benchmark_summary(self, text: str) -> None:
        with self._lock:
            self._benchmark_summary = text

    def get_telemetry(self) -> dict[str, float | bool | int | str]:
        """Thread-safe snapshot for ABA UI (no game memory)."""
        with self._lock:
            active = self.running and self._mouse_enabled and not self._stopping
            ads_live = self._ads.is_ads_active() if active else False
            paused = self._paused
            locked = self._locked_target is not None
            stats = self._stats.last if self._stats and self._stats.total_frames > 0 else None
            stats_valid = bool(
                stats is not None and self.running and not self._stopping and active
            )
            frame_has = self._frame_has_target and not paused and active
            has_target = frame_has
            if self._force_detect and active and not paused:
                ads_ui = has_target or ads_live
            else:
                ads_ui = ads_live and not paused and active
            return {
                "running": self.running,
                "stopping": self._stopping,
                "paused": paused,
                "ads": ads_ui,
                "has_target": has_target,
                "frame_has_target": frame_has,
                "locked": locked and not paused and active,
                "stats_valid": stats_valid,
                "live_input_enabled": self._live and active,
                "mouse_armed": self._live and active and not self._dry,
                "dry_run": self._dry,
                "fps": stats.fps if stats_valid else -1.0,
                "frame_ms": stats.frame_ms if stats_valid else -1.0,
                "configured_capture_fps": self._configured_fps,
                "dropped_frames": stats.dropped_frames if stats_valid else -1,
                "stale_frames": self._stale_frames,
                "confidence": stats.confidence if stats_valid else -1.0,
                "pull_px": stats.pull_px if stats_valid else -1.0,
                "simulated_pull_px": stats.pull_px if stats_valid and self._dry else -1.0,
                "pull_strength": stats.pull_strength if stats_valid else -1.0,
                "distance_px": stats.distance_px if stats_valid else -1.0,
                "candidates": stats.candidates if stats_valid else -1,
                "mouse_backend": stats.mouse_backend if stats_valid else self._mouse.name,
                "capture_size": (
                    f"{stats.capture_w}x{stats.capture_h}" if stats_valid else ""
                ),
                "hooks_enabled": hooks_enabled(self.config) and active,
                "kill_switch_active": kill_switch_active(self.config) and active,
                "ads_input_mode": str(self.config.get("ads_input_mode", "disabled")),
                "last_gate_block": self._last_gate_block,
                "benchmark_summary": self._benchmark_summary,
                "aim_smooth_x": self._last_motion.x if self._last_motion else -1.0,
                "aim_smooth_y": self._last_motion.y if self._last_motion else -1.0,
            }

    def _target_process_ok(self, cfg: dict[str, Any]) -> bool:
        proc = str(cfg.get("target_process_name", "")).strip()
        if not proc or not bool(cfg.get("pause_on_target_closed", True)):
            return True
        from process_presence import is_target_process_running

        return self._process_debounce.is_running(proc)

    def _safe_mouse_move(self, dx: int, dy: int) -> MouseGateResult:
        if dx == 0 and dy == 0:
            return MouseGateResult(True, "")
        cfg = self.config
        proc_ok = self._target_process_ok(cfg)
        with self._lock:
            running = self.running
            stopping = self._stopping
            paused = self._paused
            mouse_enabled = self._mouse_enabled
            has_target = self._locked_target is not None
            detection_fresh = self._frame_has_target
            stale_grace = int(cfg.get("mouse_gate_stale_grace_frames", 12))
            budget_scale = float(cfg.get("mouse_gate_pull_budget_scale", 3.5))
            ctx = MouseGateContext(
                running=running,
                stopping=stopping,
                paused=paused,
                mouse_enabled=mouse_enabled,
                ads_active=self._ads.is_ads_active(),
                has_target=has_target,
                detection_fresh=detection_fresh,
                target_lost_frames=self._target_lost_frames,
                stale_grace_frames=stale_grace,
                target_process_ok=proc_ok,
                dx=dx,
                dy=dy,
                max_pull_per_frame=float(cfg["max_pull_speed_pixels_per_frame"]),
                pull_budget_scale=budget_scale,
            )
            result = evaluate_mouse_gate(cfg, ctx)
            self._last_gate_block = result.reason
            if not result.allowed or not self.running or stopping:
                return result
        try:
            self._mouse.move_relative(dx, dy)
        except Exception:
            logger.debug("mouse move_relative failed", exc_info=True)
        return result

    def _sleep_interruptible(self, seconds: float) -> None:
        if seconds <= 0:
            return
        end = time.perf_counter() + seconds
        while True:
            remaining = end - time.perf_counter()
            if remaining <= 0:
                return
            if not self._should_run():
                return
            time.sleep(min(0.02, remaining))

    def _update_target_pause(self, cfg: dict[str, Any]) -> bool:
        """True when assist should idle because target process/window is gone."""
        if not bool(cfg.get("pause_on_target_closed", True)):
            with self._lock:
                self._paused = False
                self._was_paused = False
            return False
        proc = str(cfg.get("target_process_name", "")).strip()
        if not proc:
            with self._lock:
                self._paused = False
            return False
        running = self._process_debounce.is_running(proc)
        with self._lock:
            self._paused = not running
            if self._paused:
                self._mouse_enabled = False
                if not self._was_paused:
                    self._release_ads_inner()
            else:
                self._mouse_enabled = self.running and not self._stopping
            self._was_paused = self._paused
        return self._paused

    def _release_ads_inner(self) -> None:
        self._locked_target = None
        self._target_lost_frames = 0
        self._switch_candidate = None
        self._switch_frames = 0
        self._aim_tracker.reset()
        self._last_motion = None
        if self._pull is not None:
            self._pull.reset()

    def _on_click(self, _x: int, _y: int, button, pressed: bool) -> None:
        if getattr(button, "name", None) == "right" or str(button).endswith("right"):
            was = self._ads.is_ads_active()
            self._ads.set_pynput_ads(pressed)
            if was and not pressed:
                self._release_ads()

    @staticmethod
    def _key_label(key) -> str | None:
        name = getattr(key, "name", None)
        if name:
            return str(name).lower()
        char = getattr(key, "char", None)
        if char:
            return str(char).lower()
        label = str(key).lower()
        if label.startswith("key."):
            return label[4:]
        return label

    def _on_key(self, key) -> bool | None:
        kill = self.config["kill_switch_key"]
        name = self._key_label(key)
        if name and name == kill:
            logger.info("Kill switch %s — stopping runtime.", kill.upper())
            print(f"[ABA] Kill switch ({kill.upper()}) — stopping runtime.")
            with self._lock:
                self._stopping = True
                self.running = False
                self._mouse_enabled = False
            if self._on_external_stop is not None:
                self._on_external_stop()
            else:
                self.stop()
            return False
        return None

    def _release_ads(self) -> None:
        with self._lock:
            self._release_ads_inner()

    def _start_overlay(
        self,
        width: int,
        height: int,
        origin_x: int,
        origin_y: int,
        fov_center_x: float,
        fov_center_y: float,
    ) -> None:
        from overlay_window import OverlayWindow

        radius = int(self.config["fov_radius_pixels"])
        self._overlay = OverlayWindow(
            width,
            height,
            radius,
            origin_x,
            origin_y,
            fov_center_x=fov_center_x,
            fov_center_y=fov_center_y,
        )

        def run_overlay() -> None:
            assert self._overlay is not None
            self._overlay.run()

        self._overlay_thread = threading.Thread(target=run_overlay, daemon=True)
        self._overlay_thread.start()

    def _select_target(
        self,
        frame_bgr,
        hsv_ranges: list,
        fov_radius: int,
        min_area: float,
        center_x: float,
        center_y: float,
    ):
        cfg = self.config
        with self._lock:
            sticky = self._locked_target if self._target_lost_frames < int(
                cfg["target_lost_frames_before_unlock"]
            ) else None
        result = find_best_target(
            frame_bgr,
            hsv_ranges,
            fov_radius,
            min_area,
            center_x,
            center_y,
            sticky_target=sticky,
            stickiness_pixels=float(cfg["target_stickiness_pixels"]),
            distance_weight=float(cfg["distance_score_weight"]),
            area_weight=float(cfg["area_score_weight"]),
            min_height_px=float(cfg["humanoid_min_height_pixels"]),
            min_aspect=float(cfg["humanoid_min_aspect"]),
            max_aspect=float(cfg["humanoid_max_aspect"]),
            min_solidity=float(cfg.get("humanoid_min_solidity", 0.25)),
            torso_aim_fraction=float(cfg.get("torso_aim_fraction", 0.38)),
            debug=bool(cfg.get("verbose_logging", False)),
        )
        with self._lock:
            if result.target is not None:
                new_t = result.target
                if self._locked_target is not None:
                    import math as _m

                    drift = _m.hypot(
                        new_t.centroid_x - self._locked_target.centroid_x,
                        new_t.centroid_y - self._locked_target.centroid_y,
                    )
                    if drift < 25:
                        self._locked_target = new_t
                        self._target_lost_frames = 0
                        self._switch_candidate = None
                        self._switch_frames = 0
                        return result
                    if (
                        self._switch_candidate is not None
                        and _m.hypot(
                            new_t.centroid_x - self._switch_candidate.centroid_x,
                            new_t.centroid_y - self._switch_candidate.centroid_y,
                        )
                        < 30
                    ):
                        self._switch_frames += 1
                    else:
                        self._switch_candidate = new_t
                        self._switch_frames = 1
                    if self._switch_frames >= 3:
                        self._locked_target = new_t
                        self._target_lost_frames = 0
                        self._switch_candidate = None
                        self._switch_frames = 0
                        return result
                    self._target_lost_frames = 0
                    return DetectionResult(
                        self._locked_target,
                        result.candidates,
                        self._locked_target.confidence,
                    )
                self._locked_target = new_t
                self._target_lost_frames = 0
                self._switch_candidate = None
                self._switch_frames = 0
                return result

            self._target_lost_frames += 1
            self._switch_candidate = None
            self._switch_frames = 0
            lost_max = int(cfg["target_lost_frames_before_unlock"])
            if self._target_lost_frames >= lost_max:
                self._locked_target = None
                if self._pull is not None:
                    self._pull.reset()
                self._aim_tracker.reset()
            elif self._locked_target is not None:
                return DetectionResult(
                    self._locked_target,
                    result.candidates,
                    self._locked_target.confidence,
                )
            return DetectionResult(None, result.candidates, 0.0)

    def _teardown(
        self,
        mouse_listener: Any,
        keyboard_listener: Any,
    ) -> None:
        with self._lock:
            self._mouse_enabled = False
            self._stopping = True
        self._ads.clear()
        self._release_ads()
        self._process_debounce.reset()
        with self._lock:
            self._frame_has_target = False
        self._clear_stats()
        if mouse_listener is not None:
            try:
                mouse_listener.stop()
            except Exception:
                logger.exception("Failed to stop mouse listener")
        if keyboard_listener is not None:
            try:
                keyboard_listener.stop()
            except Exception:
                logger.exception("Failed to stop keyboard listener")
        if self._overlay is not None:
            self._overlay.close()
            if self._overlay_thread is not None:
                self._overlay_thread.join(timeout=2.0)
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    def run(self) -> int:
        cfg = self.config
        ok, msg = validate_runtime_policy(cfg)
        if not ok:
            raise RuntimeError(msg)

        dpi = enable_dpi_awareness()
        logger.info("DPI awareness: %s", dpi)

        fov_radius = int(cfg["fov_radius_pixels"])
        fps = self._configured_fps
        if self._dry:
            print(
                f"[ABA] DRY-RUN @ {fps} FPS (capped): mask/detection sanity — NOT flick/reacquire/combat proof."
            )
        self._stats = RuntimeStats(fps)
        self._trace_pull = bool(cfg.get("trace_pull", False))
        self._trace_frame = 0
        log_interval = max(1, int(cfg.get("stats_log_interval_frames", 60)))

        self._aim_tracker.configure_prediction(
            bool(cfg["prediction_enabled"]),
            float(cfg["prediction_lead_seconds"]),
            float(cfg["prediction_max_pixels"]),
        )

        self._pull = PullController(
            PullTuning(
                max_speed=float(cfg["max_pull_speed_pixels_per_frame"]),
                pull_strength=float(cfg["pull_strength"]),
                deadzone=float(cfg["deadzone_pixels"]),
                velocity_smoothing=float(cfg["velocity_smoothing"]),
                smoothing_curve=str(cfg["smoothing_curve"]),
                magnetism_radius=float(cfg["magnetism_radius_pixels"]),
                magnetism_min_scale=float(cfg["magnetism_min_pull_scale"]),
                fov_radius=float(fov_radius),
                fov_edge_min_scale=float(cfg["fov_edge_min_pull_scale"]),
                prediction_enabled=bool(cfg["prediction_enabled"]),
                prediction_lead_seconds=float(cfg["prediction_lead_seconds"]),
                prediction_max_pixels=float(cfg["prediction_max_pixels"]),
                humanize_enabled=bool(cfg["humanize_enabled"]),
                humanize_amplitude=float(cfg["humanize_amplitude_pixels"]),
                humanize_jerk_limit=float(cfg["humanize_jerk_limit"]),
                aim_pre_smoothed=True,
            )
        )

        mouse_listener = None
        keyboard_listener = None
        frame_interval = 1.0 / fps
        if self._live:
            from pynput import keyboard, mouse

            mouse_listener = mouse.Listener(on_click=self._on_click)
            keyboard_listener = keyboard.Listener(on_press=self._on_key)
            mouse_listener.start()
            keyboard_listener.start()

        kill_key = str(cfg["kill_switch_key"]).upper()
        profile = cfg.get("profile", "apex_style_dry_run")
        if self._live:
            print(f"[ABA] LIVE INPUT ENABLED — ban risk. Hold RMB. {kill_key} to stop.")
        else:
            print(
                f"[ABA] MODE: DRY RUN — no hooks, no OS mouse. Stop via ABA UI. "
                f"{kill_key} inactive."
            )
        print(f"[ABA] Profile: {profile} | capture_fps={fps} | Mouse: {self._mouse.name}")
        print(f"[ABA] Target process: {cfg.get('target_process_name', '') or '(none)'}")
        print(
            "[ABA] Aim path: detector body-shape -> motion.observe_target(bbox) -> pull/overlay"
        )
        if cfg.get("pause_on_target_closed", True) and cfg.get("target_process_name"):
            print("[ABA] Assist pauses when target process is not running.")
        if self._live and cfg.get("ads_input_mode") in ("both", "win32_poll"):
            print("[ABA] WARN: win32_poll sees RMB globally — WILL BAN if used online with anti-cheat.")

        with mss.mss() as sct:
            monitor_index = int(cfg["monitor_index"])
            monitors = sct.monitors
            if monitor_index < 1 or monitor_index >= len(monitors):
                print(
                    f"[OverlayAssist] monitor_index {monitor_index} invalid; using 1. "
                    f"Available: 1..{len(monitors) - 1}",
                    file=sys.stderr,
                )
                monitor_index = 1
            mon = monitors[monitor_index]
            logger.info(
                "Monitor %d: %dx%d @ (%d,%d)",
                monitor_index,
                mon["width"],
                mon["height"],
                mon["left"],
                mon["top"],
            )

            center_x = mon["width"] / 2.0 + float(cfg["crosshair_offset_x"])
            center_y = mon["height"] / 2.0 + float(cfg["crosshair_offset_y"])
            cap_region = build_capture_region(
                mon,
                center_x,
                center_y,
                fov_radius,
                use_crop=bool(cfg["capture_fov_crop"]),
                crop_padding=float(cfg["capture_crop_padding"]),
            )
            frame_cx = center_x - cap_region.offset_x
            frame_cy = center_y - cap_region.offset_y

            if bool(cfg["enable_overlay"]):
                self._start_overlay(
                    mon["width"],
                    mon["height"],
                    int(mon["left"]),
                    int(mon["top"]),
                    center_x,
                    center_y,
                )

            hsv_ranges = cfg["hsv_ranges"]
            show_debug = bool(cfg["show_debug_window"])
            frame_i = 0
            with self._lock:
                self._mouse_enabled = True
                self._stopping = False

            try:
                while True:
                    t0 = time.perf_counter()
                    if not self._should_run():
                        break

                    paused = self._update_target_pause(cfg)
                    if paused:
                        with self._lock:
                            self._frame_has_target = False
                        if self._pull is not None:
                            self._pull.reset()
                        self._aim_tracker.reset()
                        sleep_time = frame_interval - (time.perf_counter() - t0)
                        self._sleep_interruptible(sleep_time)
                        continue

                    ads_live = self._ads.is_ads_active()
                    ads_for_assist = ads_live if self._live else (self._force_detect or ads_live)

                    frame_bgr = grab_bgr(sct, cap_region)

                    detection_fresh = False
                    if ads_for_assist and not paused:
                        det = self._select_target(
                            frame_bgr,
                            hsv_ranges,
                            fov_radius,
                            float(cfg["min_target_area_pixels"]),
                            frame_cx,
                            frame_cy,
                        )
                        detection_fresh = det.target is not None and self._target_lost_frames == 0
                    else:
                        det = DetectionResult(None, 0, 0.0)
                        self._aim_tracker.reset()
                    target = det.target
                    stale_det = target is not None and self._target_lost_frames > 0
                    with self._lock:
                        self._frame_has_target = detection_fresh

                    motion = self._smooth_aim(target, t0)
                    pull_target = (
                        self._target_for_pull(target, motion)
                        if target is not None and motion is not None
                        else None
                    )

                    pull_px = 0.0
                    pull_strength = 0.0
                    if (
                        not paused
                        and self._should_run()
                        and ads_for_assist
                        and pull_target is not None
                        and self._pull is not None
                    ):
                        pr = self._pull.compute_delta(
                            pull_target,
                            frame_cx,
                            frame_cy,
                            time_sec=t0,
                            stale_detection=stale_det,
                        )
                        pull_px = pr.magnitude
                        pull_strength = pr.effective_strength
                        gate_result = MouseGateResult(True, "")
                        moved = (0, 0)
                        if (pr.dx != 0 or pr.dy != 0) and self._should_run():
                            gate_result = self._safe_mouse_move(pr.dx, pr.dy)
                            if gate_result.allowed:
                                moved = (pr.dx, pr.dy)
                        if self._trace_pull and target is not None and motion is not None:
                            from pull_trace import PullTraceFrame, log_trace_frame

                            self._trace_frame += 1
                            log_trace_frame(
                                PullTraceFrame(
                                    frame=self._trace_frame,
                                    raw_target=(target.centroid_x, target.centroid_y),
                                    motion_target=(motion.x, motion.y),
                                    center=(frame_cx, frame_cy),
                                    error=(motion.x - frame_cx, motion.y - frame_cy),
                                    pull_dxdy=(pr.dx, pr.dy),
                                    pull_mag=pr.magnitude,
                                    pull_vel=(pr.vel_x, pr.vel_y),
                                    pull_desired=(pr.desired_x, pr.desired_y),
                                    gate_allowed=gate_result.allowed,
                                    gate_reason=gate_result.reason,
                                    mouse_move_called=moved,
                                    detection_fresh=detection_fresh,
                                    target_lost_frames=self._target_lost_frames,
                                    stale_detection=stale_det,
                                )
                            )

                    elif (not ads_for_assist or paused or target is None) and self._pull is not None:
                        self._pull.reset()

                    with self._lock:
                        elapsed_ms = (time.perf_counter() - t0) * 1000.0
                        frame_interval_ms = frame_interval * 1000.0
                        if (
                            elapsed_ms > frame_interval_ms * 2.0
                            and self.running
                            and not self._stopping
                        ):
                            self._stale_frames += 1
                        record_stats = self.running and not self._stopping
                        if record_stats and self._stats is not None:
                            self._stats.begin_frame(t0)
                            self._stats.update(
                                frame_ms=elapsed_ms,
                                ads=ads_for_assist,
                                target_distance=pull_target.distance_to_center
                                if pull_target
                                else None,
                                confidence=pull_target.confidence if pull_target else 0.0,
                                pull_px=pull_px,
                                pull_strength=pull_strength,
                                locked=self._locked_target is not None,
                                has_target=detection_fresh,
                                mouse_backend=self._mouse.name,
                                candidates=det.candidates,
                                capture_size=(cap_region.width, cap_region.height),
                            )

                    if self._overlay is not None and self._should_run():
                        if motion is not None:
                            ox, oy = to_monitor_coords(motion.x, motion.y, cap_region)
                            overlay_pt = (ox, oy)
                        else:
                            overlay_pt = None
                        self._overlay.set_state(ads_for_assist, overlay_pt)

                    frame_i += 1
                    if frame_i % log_interval == 0 and self._stats is not None:
                        for line in self._stats.format_lines():
                            logger.info(line)

                    if show_debug and self._should_run():
                        dbg_target = pull_target if pull_target is not None else target
                        dbg = draw_debug(
                            frame_bgr,
                            dbg_target,
                            fov_radius,
                            frame_cx,
                            frame_cy,
                            magnetism_radius=int(cfg["magnetism_radius_pixels"]),
                            hsv_ranges=hsv_ranges,
                            stats_lines=self._stats.format_lines() if self._stats else [],
                            detection_debug=det.debug_lines if hasattr(det, "debug_lines") else None,
                        )
                        if motion is not None:
                            cv2.drawMarker(
                                dbg,
                                (int(motion.x), int(motion.y)),
                                (255, 255, 0),
                                cv2.MARKER_DIAMOND,
                                12,
                                2,
                            )
                        cv2.imshow("OverlayAssist debug", dbg)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            self.stop()

                    sleep_time = frame_interval - (time.perf_counter() - t0)
                    self._sleep_interruptible(sleep_time)
            finally:
                self._teardown(mouse_listener, keyboard_listener)
        return 0
