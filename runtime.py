"""Main assist loop — testable, Windows-ready."""

from __future__ import annotations

import logging
import math
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
from detector import (
    DetectionContext,
    DetectionResult,
    Target,
    draw_debug,
    find_best_target,
)
from target_lock import (
    TargetLockState,
    apply_target_lock,
    detection_sticky_context,
    may_assist_pull_target,
    overlay_may_show_target,
    viewmodel_exclude_bottom,
)
from input_state import AdsInputState
from motion import TargetMotion, TargetTracker
from mouse_gate import MouseGateContext, MouseGateResult, evaluate_mouse_gate
from mouse_io import MouseBackend, create_mouse_backend
from platform_info import enable_dpi_awareness
from process_presence import ProcessPresenceDebouncer
from profiles import (
    effective_capture_fps,
    effective_capture_fov_radius,
    effective_detection_fov_radius,
    effective_fov_radius,
    effective_overlay_fps,
)
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
        # Stateful detection context — supplies prev-frame gray buffer for the
        # motion-difference channel that boosts recall on low-contrast targets.
        self._detect_ctx = DetectionContext(
            motion_assist=bool(config.get("detection_motion_assist", True)),
            motion_threshold=int(config.get("detection_motion_threshold", 10)),
        )

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
        self._target_lock = TargetLockState()
        self._pull: PullController | None = None
        self._stats: RuntimeStats | None = None
        self._paused = False
        self._was_paused = False
        self._mouse_enabled = False
        self._stopping = False
        self._process_debounce = process_debounce or ProcessPresenceDebouncer()
        self._frame_has_target = False
        self._prev_ads_for_assist = False
        self._ads_hold_frames = 0
        self._trace_frame = 0
        self._trace_pull = False
        self._last_debug: dict[str, float | int | str | bool] = {}
        self._last_pull_dx = 0
        self._last_pull_dy = 0
        self._last_gate_allowed = True
        self._pending_debug_save = False
        self._last_frame_bgr = None
        # LMB-held flag for recoil compensator engagement gating. Updated by
        # the pynput mouse listener under _lock; read by the runtime loop.
        self._is_firing = False

    @property
    def _locked_target(self) -> Target | None:
        return self._target_lock.locked_target

    @_locked_target.setter
    def _locked_target(self, value: Target | None) -> None:
        self._target_lock.locked_target = value

    @property
    def _target_lost_frames(self) -> int:
        return self._target_lock.target_lost_frames

    @_target_lost_frames.setter
    def _target_lost_frames(self, value: int) -> None:
        self._target_lock.target_lost_frames = value

    @property
    def _switch_candidate(self) -> Target | None:
        return self._target_lock.switch_candidate

    @_switch_candidate.setter
    def _switch_candidate(self, value: Target | None) -> None:
        self._target_lock.switch_candidate = value

    @property
    def _switch_frames(self) -> int:
        return self._target_lock.switch_frames

    @_switch_frames.setter
    def _switch_frames(self, value: int) -> None:
        self._target_lock.switch_frames = value

    def _smooth_aim(
        self,
        target: Target | None,
        time_sec: float,
        *,
        stale: bool = False,
        keep_motion_anchor: bool = False,
    ) -> TargetMotion | None:
        """
        Upper-chest body column via observe_target(bbox_*).
        Returns None when no target — overlay/pull must not use raw plate centroids.

        M1 (audit): when ``stale`` is True (the runtime is returning a
        frozen lock during the grace window because no fresh detection
        was made this frame) we DO NOT call ``observe_target`` — that
        would keep feeding the smoother with the stale centroid every
        frame and accumulate motion the user perceives as glitchy chase.
        Instead we hold ``_last_motion`` so the overlay/pull see a frozen
        anchor until detection refreshes or the lock expires.

        PHASE-7 AUDIT FIX (CRIT2): when ``target`` is None we used to
        immediately ``self._aim_tracker.reset()`` (HARD reset that
        clears ``_smooth_x/y`` and ``_last_meas_x/y``). The runtime's
        ``_select_target`` calls ``self._aim_tracker.soft_reset()``
        right before returning None at the lock-expiry boundary —
        carefully preserving the smoothed anchor — and then the very
        next line of the main loop calls ``_smooth_aim(target=None)``
        which would HARD reset and undo the soft_reset. The next
        acquired target would teleport.

        Fix: when we have a ``_last_motion`` cached (lock just
        expired but the tracker still carries an anchor), return that
        frozen anchor for one frame instead of hard-resetting. The
        overlay logic already hides the dot after ``>=2`` stale frames
        (``hide_overlay_stale``), and the next observed target will
        step-cap from this anchor rather than teleport.
        """
        if target is None:
            # CRIT2 pull anchor only — never feed a ghost dot when detection
            # is empty (firing-range crates/HUD were showing _last_motion).
            if keep_motion_anchor and self._last_motion is not None:
                return self._last_motion
            self._aim_tracker.reset()
            return None

        if stale:
            return self._last_motion

        fov_r = float(self.config.get("_runtime_detect_fov", 0) or 0)
        if fov_r > 0 and hasattr(self, "_frame_cx"):
            self._aim_tracker.configure_fov_clamp(self._frame_cx, self._frame_cy, fov_r)
        motion = self._aim_tracker.observe_target(
            target.centroid_x,
            target.centroid_y,
            time_sec,
            bbox_x=target.bbox_x,
            bbox_y=target.bbox_y,
            bbox_w=target.bbox_w,
            bbox_h=target.bbox_h,
            aim_is_body_anchor=True,
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
            self._is_firing = False
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
            locked = (
                self._locked_target is not None
                and ads_live
                and not paused
            )
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
                "locked": locked and active,
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
                **self._last_debug,
                "pull_dx": float(self._last_pull_dx),
                "pull_dy": float(self._last_pull_dy),
                "mouse_gate_allowed": bool(self._last_gate_allowed),
            }

    def _target_process_ok(self, cfg: dict[str, Any]) -> bool:
        proc = str(cfg.get("target_process_name", "")).strip()
        if not proc or not bool(cfg.get("pause_on_target_closed", True)):
            return True
        from process_presence import is_target_process_running

        return self._process_debounce.is_running(proc)


    def _emit_pull_trace(
        self,
        cfg: dict[str, Any],
        *,
        frame_cx: float,
        frame_cy: float,
        target: Target | None,
        motion: TargetMotion | None,
        pr_dx: int,
        pr_dy: int,
        pr_mag: float,
        pr_vel: tuple[float, float],
        pr_desired: tuple[float, float],
        gate_allowed: bool,
        gate_reason: str,
        mouse_move: tuple[int, int],
        detection_fresh: bool,
        stale_det: bool,
        ads_active: bool,
        overlay_dot: tuple[float, float] | None = None,
        capture_ms: float = -1.0,
        detect_ms: float = -1.0,
        total_loop_ms: float = -1.0,
        achieved_fps: float = -1.0,
    ) -> None:
        if not self._trace_pull:
            return
        from pull_trace import PullTraceFrame, log_trace_frame

        self._trace_frame += 1
        if motion is not None:
            mx, my = motion.x, motion.y
        elif target is not None:
            mx, my = target.centroid_x, target.centroid_y
        else:
            mx, my = frame_cx, frame_cy
        if target is not None:
            rx, ry = target.centroid_x, target.centroid_y
        else:
            rx, ry = mx, my
        raw_bbox = None
        head_s = torso_s = limb_s = -1.0
        selected_reason = ""
        if target is not None:
            raw_bbox = (target.bbox_x, target.bbox_y, target.bbox_w, target.bbox_h)
            head_s = float(getattr(target, "head_score", -1.0))
            torso_s = float(getattr(target, "torso_score", -1.0))
            limb_s = float(getattr(target, "limb_stack_score", -1.0))
            selected_reason = str(getattr(target, "reject_reason", "") or "body_lock")
        pred_off = getattr(self._aim_tracker, "last_prediction_offset", (0.0, 0.0))
        pre_pp = getattr(self._aim_tracker, "last_pre_predict_point", None)
        body_clamp = pre_pp if pre_pp is not None else (rx, ry)
        pull_in = (mx, my) if motion is not None else (rx, ry)
        log_trace_frame(
            PullTraceFrame(
                frame=self._trace_frame,
                raw_target=(rx, ry),
                motion_target=(mx, my),
                raw_detector_anchor=(rx, ry),
                body_anchor_after_clamp=body_clamp,
                prediction_offset=pred_off,
                pull_input=pull_in,
                center=(frame_cx, frame_cy),
                error=(mx - frame_cx, my - frame_cy),
                pull_dxdy=(pr_dx, pr_dy),
                pull_mag=pr_mag,
                pull_vel=pr_vel,
                pull_desired=pr_desired,
                gate_allowed=gate_allowed,
                gate_reason=gate_reason,
                mouse_move_called=mouse_move,
                detection_fresh=detection_fresh,
                target_lost_frames=self._target_lost_frames,
                stale_detection=stale_det,
                has_target=target is not None,
                ads_active=ads_active,
                raw_bbox=raw_bbox,
                raw_anchor=(rx, ry),
                motion_anchor=(mx, my),
                overlay_dot=overlay_dot,
                center_error=(mx - frame_cx, my - frame_cy),
                pull_output=(pr_dx, pr_dy),
                head_score=head_s,
                torso_score=torso_s,
                limb_score=limb_s,
                selected_reason=selected_reason,
                capture_ms=capture_ms,
                detect_ms=detect_ms,
                total_loop_ms=total_loop_ms,
                achieved_fps=achieved_fps,
            ),
            cfg,
        )


    def update_live_debug(
        self,
        *,
        target: Target | None,
        capture_ms: float,
        detect_ms: float,
        detection_fresh: bool,
        motion_lag_ms: float = -1.0,
    ) -> None:
        if target is None:
            snap = {
                "body_shape_score": -1.0,
                "head_score": -1.0,
                "torso_score": -1.0,
                "limb_stack_score": -1.0,
                "reject_reason": "",
                "anchor_x": -1.0,
                "anchor_y": -1.0,
                "bbox_x": -1,
                "bbox_y": -1,
                "bbox_w": -1,
                "bbox_h": -1,
                "capture_ms": capture_ms,
                "detect_ms": detect_ms,
                "motion_lag_ms": motion_lag_ms,
                "detection_fresh": detection_fresh,
            }
        else:
            snap = {
                "body_shape_score": float(target.body_shape_score),
                "head_score": float(target.head_score),
                "torso_score": float(target.torso_score),
                "limb_stack_score": float(target.limb_stack_score),
                "reject_reason": str(getattr(target, "reject_reason", "") or ""),
                "anchor_x": float(target.centroid_x),
                "anchor_y": float(target.centroid_y),
                "bbox_x": int(target.bbox_x),
                "bbox_y": int(target.bbox_y),
                "bbox_w": int(target.bbox_w),
                "bbox_h": int(target.bbox_h),
                "capture_ms": capture_ms,
                "detect_ms": detect_ms,
                "motion_lag_ms": motion_lag_ms,
                "detection_fresh": detection_fresh,
            }
        with self._lock:
            self._last_debug = snap

    def request_debug_frame_save(self) -> None:
        with self._lock:
            self._pending_debug_save = True

    def _maybe_save_debug_frame(self, frame_bgr, det, cfg, cx, cy, fov) -> None:
        pending = False
        with self._lock:
            pending = self._pending_debug_save
            self._pending_debug_save = False
        if not pending:
            return
        try:
            import json
            from datetime import datetime

            import detector

            out_root = Path(str(cfg.get("debug_frames_dir", "artifacts/debug_frames")))
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = out_root / stamp
            out_dir.mkdir(parents=True, exist_ok=True)
            hsv = cfg["hsv_ranges"]
            min_area = float(cfg["min_target_area_pixels"])
            candidates, mask, _parts = detector.enumerate_candidates(
                frame_bgr, hsv, int(fov), min_area, cx, cy,
                torso_aim_fraction=float(cfg.get("torso_aim_fraction", 0.38)),
                body_shape_min_score=float(cfg.get("body_shape_min_score", 0.40)),
                detection_mode=str(cfg.get("detection_mode", "apex")),
            )
            import cv2

            cv2.imwrite(str(out_dir / "01_original.png"), frame_bgr)
            cv2.imwrite(str(out_dir / "02_hsv_mask.png"), mask)
            overlay = detector.render_debug_artifacts(
                frame_bgr,
                candidates,
                det.target if det.target is not None else None,
                int(fov),
                cx,
                cy,
                mask=mask,
            )
            cv2.imwrite(str(out_dir / "03_overlay.png"), overlay)
            from dataclasses import asdict

            meta = {
                "target": asdict(det.target) if det.target is not None else None,
                "candidates": [
                    {
                        "idx": c.idx,
                        "body": c.body_shape_score,
                        "reject": c.reject_reason,
                    }
                    for c in candidates[:12]
                ],
            }
            (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            logger.info("Saved debug frame artifacts to %s", out_dir)
        except Exception:
            logger.exception("debug frame save failed")


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
            self._last_gate_allowed = result.allowed
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
        self._target_lock.reset()
        # PHASE-7 AUDIT FIX (HIGH4): RMB releases used to ``reset()`` the
        # tracker, wiping ``_smooth_x/y`` and ``_last_meas_x/y``. Every
        # re-ADS observation would then have no step-cap anchor, so the
        # first observed target teleported into place. ``soft_reset()``
        # preserves the geometric anchor; the lock state above is
        # already cleared so there's no risk of re-using stale lock
        # data on the next acquisition. Hard ``reset()`` is reserved
        # for ``stop()`` / ``_teardown`` (full session end).
        self._aim_tracker.soft_reset()
        self._last_motion = None
        self._detect_ctx.reset()
        # Releasing ADS implicitly ends an engagement — drop the firing edge
        # so the recoil compensator phase resets cleanly. The LMB listener
        # will re-arm on the next LMB press.
        self._is_firing = False
        if self._pull is not None:
            self._pull.reset()

    def _on_click(self, _x: int, _y: int, button, pressed: bool) -> None:
        name = getattr(button, "name", None)
        label = str(button)
        if name == "right" or label.endswith("right"):
            was = self._ads.is_ads_active()
            self._ads.set_pynput_ads(pressed)
            if was and not pressed:
                self._release_ads()
        elif name == "left" or label.endswith("left"):
            # Recoil compensator engagement signal — `_is_firing` is read under
            # `_lock` from the main runtime loop. Polling pynput state would be
            # race-prone; tracking edges here is the only thread-safe path.
            with self._lock:
                self._is_firing = bool(pressed)

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

    def _sync_ads_assist_state(self, ads_for_assist: bool) -> None:
        """Clear lock/pull when ADS ends; reset motion memory on ADS start / long hold."""
        if ads_for_assist and not self._prev_ads_for_assist:
            self._ads_hold_frames = 0
            self._detect_ctx._validated_bbox = None
            self._detect_ctx._validated_credit = 0
            with self._lock:
                self._target_lock.overlay_confirm_frames = 0
        if self._prev_ads_for_assist and not ads_for_assist:
            self._release_ads_inner()
            self._ads_hold_frames = 0
        elif ads_for_assist:
            self._ads_hold_frames += 1
            # Long ADS: decay stale motion-validation so ranking cannot drift to sky.
            if self._ads_hold_frames in (1, 90, 180, 270):
                self._detect_ctx._validated_bbox = None
                self._detect_ctx._validated_credit = 0
        self._prev_ads_for_assist = ads_for_assist

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

        radius = effective_fov_radius(self.config, ads_active=False)
        cfg = self.config
        self._overlay = OverlayWindow(
            width,
            height,
            radius,
            origin_x,
            origin_y,
            fov_center_x=fov_center_x,
            fov_center_y=fov_center_y,
            overlay_fps=effective_overlay_fps(cfg),
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
            sticky, currently_locked, _lost_max = detection_sticky_context(
                self._target_lock, cfg
            )
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
            body_shape_min_score=float(cfg.get("body_shape_min_score", 0.40)),
            head_score_weight=float(cfg.get("head_score_weight", 0.26)),
            torso_score_weight=float(cfg.get("torso_score_weight", 0.26)),
            limb_stack_score_weight=float(cfg.get("limb_stack_score_weight", 0.22)),
            aim_y_min_fraction=float(cfg.get("aim_body_y_min_fraction", 0.28)),
            aim_y_max_fraction=float(cfg.get("aim_body_y_max_fraction", 0.52)),
            debug=bool(cfg.get("verbose_logging", False)),
            detection_mode=str(cfg.get("detection_mode", "apex")),
            context=self._detect_ctx,
            currently_locked=currently_locked,
            exclude_bottom_frac=viewmodel_exclude_bottom(cfg),
        )
        with self._lock:

            def _on_lock_expired() -> None:
                if self._pull is not None:
                    self._pull.reset()
                self._aim_tracker.soft_reset()

            fh, fw = frame_bgr.shape[0], frame_bgr.shape[1]
            result, _is_stale = apply_target_lock(
                self._target_lock,
                result,
                center_y=center_y,
                cfg=cfg,
                on_lock_expired=_on_lock_expired,
                fov_cx=center_x,
                fov_cy=center_y,
                frame_size=(fw, fh),
            )
            return result

    def _teardown(
        self,
        mouse_listener: Any,
        keyboard_listener: Any,
    ) -> None:
        with self._lock:
            self._mouse_enabled = False
            self._stopping = True
            self._is_firing = False
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

        fps = self._configured_fps
        self._cap_region = None
        self._frame_cx = 0.0
        self._frame_cy = 0.0
        self._last_fov_radius = -1
        self._last_display_fov = -1
        self._last_overlay_fps = -1
        if self._dry:
            print(
                f"[ABA] DRY-RUN @ {fps} FPS (capped): mask/detection sanity — NOT flick/reacquire/combat proof."
            )
        self._stats = RuntimeStats(fps)
        self._trace_pull = bool(cfg.get("trace_pull", False))
        self._trace_frame = 0
        if self._trace_pull:
            from pull_trace import setup_trace_logging

            setup_trace_logging(cfg, app_root=self.config_path.parent)
            print(
                f"[ABA] Pull trace ON -> {cfg.get('trace_pull_log_file', 'logs/pull_trace.log')} "
                f"(interval={cfg.get('trace_pull_interval_frames', 1)} "
                f"max={cfg.get('trace_pull_max_frames', 0)})"
            )
        log_interval = max(1, int(cfg.get("stats_log_interval_frames", 60)))

        self._aim_tracker.configure_prediction(
            bool(cfg["prediction_enabled"]),
            float(cfg["prediction_lead_seconds"]),
            float(cfg["prediction_max_pixels"]),
            vertical_cap_pixels=float(cfg.get("prediction_vertical_cap_pixels", 4.0)),
        )
        self._aim_tracker.configure_body_clamp(
            float(cfg.get("aim_body_y_min_fraction", 0.28)),
            float(cfg.get("aim_body_y_max_fraction", 0.52)),
        )
        self._aim_tracker.configure_smoothing_tau(
            float(cfg.get("smoothing_tau_still", 0.062)),
            float(cfg.get("smoothing_tau_moving", 0.028)),
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
                fov_radius=float(
                    effective_detection_fov_radius(cfg, ads_active=False)
                ),
                fov_edge_min_scale=float(cfg["fov_edge_min_pull_scale"]),
                prediction_enabled=bool(cfg["prediction_enabled"]),
                prediction_lead_seconds=float(cfg["prediction_lead_seconds"]),
                prediction_max_pixels=float(cfg["prediction_max_pixels"]),
                humanize_enabled=bool(cfg["humanize_enabled"]),
                humanize_amplitude=float(cfg["humanize_amplitude_pixels"]),
                humanize_jerk_limit=float(cfg["humanize_jerk_limit"]),
                aim_pre_smoothed=True,
                recoil_compensation_enabled=bool(
                    cfg.get("recoil_compensation_enabled", False)
                ),
                recoil_pull_down_pixels_per_second=float(
                    cfg.get("recoil_pull_down_pixels_per_second", 0.0)
                ),
                jitter_enabled=bool(cfg.get("jitter_enabled", False)),
                jitter_amplitude_pixels=float(cfg.get("jitter_amplitude_pixels", 0.0)),
                jitter_frequency_hz=float(cfg.get("jitter_frequency_hz", 6.0)),
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
            "[ABA] Aim path: shape-only detect -> motion.observe_target(bbox+FOV) -> pull/overlay"
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
            cap_region = None
            frame_cx = center_x
            frame_cy = center_y

            if bool(cfg["enable_overlay"]):
                self._start_overlay(
                    mon["width"],
                    mon["height"],
                    int(mon["left"]),
                    int(mon["top"]),
                    center_x,
                    center_y,
                )

            hsv_ranges = cfg.get("hsv_ranges", [])
            show_debug = bool(cfg.get("show_debug_window", False))
            frame_i = 0
            with self._lock:
                self._mouse_enabled = True
                self._stopping = False

            try:
                while True:
                    t0 = time.perf_counter()
                    if not self._should_run():
                        break

                    cfg = self.config
                    hsv_ranges = cfg.get("hsv_ranges", [])
                    show_debug = bool(cfg.get("show_debug_window", False))

                    # PHASE-5 AUDIT FIX (D-LOW hot-reload): re-read
                    # ``enable_overlay`` and ``trace_pull`` each frame so
                    # the user's GUI toggle takes effect without
                    # Stop → Start. ``enable_overlay`` starts or closes
                    # the overlay thread on the flip; ``trace_pull``
                    # lazily wires the pull-trace logger on first
                    # enable.
                    want_overlay = bool(cfg.get("enable_overlay", False))
                    if want_overlay and self._overlay is None:
                        self._start_overlay(
                            mon["width"],
                            mon["height"],
                            int(mon["left"]),
                            int(mon["top"]),
                            center_x,
                            center_y,
                        )
                    elif (
                        not want_overlay
                        and self._overlay is not None
                    ):
                        try:
                            self._overlay.close()
                        except Exception:
                            logger.exception("hot-reload overlay close failed")
                        if self._overlay_thread is not None:
                            self._overlay_thread.join(timeout=1.5)
                        self._overlay = None
                        self._overlay_thread = None

                    want_trace = bool(cfg.get("trace_pull", False))
                    if want_trace and not self._trace_pull:
                        from pull_trace import setup_trace_logging

                        try:
                            setup_trace_logging(
                                cfg, app_root=self.config_path.parent
                            )
                            self._trace_pull = True
                            print(
                                f"[ABA] Pull trace hot-enabled -> "
                                f"{cfg.get('trace_pull_log_file', 'logs/pull_trace.log')}"
                            )
                        except Exception:
                            logger.exception("hot-reload trace_pull setup failed")
                    elif not want_trace and self._trace_pull:
                        self._trace_pull = False
                        print("[ABA] Pull trace hot-disabled")
                    paused = self._update_target_pause(cfg)
                    if paused:
                        with self._lock:
                            self._frame_has_target = False
                        if self._pull is not None:
                            self._pull.reset()
                        self._aim_tracker.reset()
                        self._detect_ctx.reset()
                        sleep_time = frame_interval - (time.perf_counter() - t0)
                        self._sleep_interruptible(sleep_time)
                        continue

                    ads_live = self._ads.is_ads_active()
                    ads_for_assist = ads_live if self._live else (self._force_detect or ads_live)
                    self._sync_ads_assist_state(ads_for_assist)

                    display_fov = effective_fov_radius(cfg, ads_active=False)
                    detect_fov = effective_detection_fov_radius(cfg, ads_active=False)
                    capture_fov = effective_capture_fov_radius(cfg, ads_active=False)
                    if cap_region is None or detect_fov != self._last_fov_radius:
                        cap_region = build_capture_region(
                            mon,
                            center_x,
                            center_y,
                            capture_fov,
                            use_crop=bool(cfg["capture_fov_crop"]),
                            crop_padding=float(cfg["capture_crop_padding"]),
                        )
                        self._frame_cx = center_x - cap_region.offset_x
                        self._frame_cy = center_y - cap_region.offset_y
                        self._last_fov_radius = detect_fov
                        cfg["_runtime_detect_fov"] = detect_fov
                        if self._pull is not None:
                            self._pull._tuning.fov_radius = float(detect_fov)
                    if self._overlay is not None and display_fov != self._last_display_fov:
                        # One ring only — resize when hip-fire display FOV changes
                        # (GUI slider), not only when detection crop rebuilds.
                        self._overlay.set_fov_radius(display_fov)
                        self._last_display_fov = display_fov
                    frame_cx = self._frame_cx
                    frame_cy = self._frame_cy

                    t_cap0 = time.perf_counter()
                    frame_bgr = grab_bgr(sct, cap_region)
                    capture_ms = (time.perf_counter() - t_cap0) * 1000.0
                    detect_ms = 0.0

                    detection_fresh = False
                    if ads_for_assist and not paused:
                        t_det0 = time.perf_counter()
                        det = self._select_target(
                            frame_bgr,
                            hsv_ranges,
                            detect_fov,
                            float(cfg["min_target_area_pixels"]),
                            frame_cx,
                            frame_cy,
                        )
                        detect_ms = (time.perf_counter() - t_det0) * 1000.0
                        detection_fresh = det.target is not None and self._target_lost_frames == 0
                        self.update_live_debug(
                            target=det.target,
                            capture_ms=capture_ms,
                            detect_ms=detect_ms,
                            detection_fresh=det.target is not None and self._target_lost_frames == 0,
                            motion_lag_ms=detect_ms,
                        )
                        self._maybe_save_debug_frame(frame_bgr, det, cfg, frame_cx, frame_cy, detect_fov)
                    else:
                        det = DetectionResult(None, 0, 0.0)
                        self._aim_tracker.reset()
                        self._detect_ctx.reset()
                    target = det.target
                    stale_det = target is not None and self._target_lost_frames > 0
                    with self._lock:
                        self._frame_has_target = detection_fresh

                    motion = self._smooth_aim(
                        target,
                        t0,
                        stale=stale_det,
                        keep_motion_anchor=False,
                    )
                    pull_target = (
                        self._target_for_pull(target, motion)
                        if target is not None and motion is not None
                        else None
                    )

                    pull_px = 0.0
                    pull_strength = 0.0
                    stale_grace = int(cfg.get("mouse_gate_stale_grace_frames", 12))
                    may_pull = (
                        pull_target is not None
                        and target is not None
                        and may_assist_pull_target(
                            target,
                            detection_fresh=detection_fresh,
                            center_y=frame_cy,
                            target_lost_frames=self._target_lost_frames,
                            stale_grace_frames=stale_grace,
                        )
                    )
                    if (
                        not paused
                        and self._should_run()
                        and ads_for_assist
                        and may_pull
                        and self._pull is not None
                    ):
                        with self._lock:
                            firing_now = self._is_firing
                        pr = self._pull.compute_delta(
                            pull_target,
                            frame_cx,
                            frame_cy,
                            time_sec=t0,
                            stale_detection=stale_det,
                            is_firing=firing_now,
                        )
                        pull_px = pr.magnitude
                        pull_strength = pr.effective_strength
                        with self._lock:
                            self._last_pull_dx = pr.dx
                            self._last_pull_dy = pr.dy
                        gate_result = MouseGateResult(True, "")
                        moved = (0, 0)
                        if (pr.dx != 0 or pr.dy != 0) and self._should_run():
                            gate_result = self._safe_mouse_move(pr.dx, pr.dy)
                            if gate_result.allowed:
                                moved = (pr.dx, pr.dy)
                        overlay_mon = None
                        if motion is not None and cap_region is not None:
                            ox, oy = to_monitor_coords(motion.x, motion.y, cap_region)
                            overlay_mon = (ox, oy)
                        loop_ms = (time.perf_counter() - t0) * 1000.0
                        ach_fps = 1000.0 / loop_ms if loop_ms > 0.1 else 0.0
                        if self._trace_pull:
                            self._emit_pull_trace(
                                cfg,
                                frame_cx=frame_cx,
                                frame_cy=frame_cy,
                                target=target,
                                motion=motion,
                                pr_dx=pr.dx,
                                pr_dy=pr.dy,
                                pr_mag=pr.magnitude,
                                pr_vel=(pr.vel_x, pr.vel_y),
                                pr_desired=(pr.desired_x, pr.desired_y),
                                gate_allowed=gate_result.allowed,
                                gate_reason=gate_result.reason,
                                mouse_move=moved,
                                detection_fresh=detection_fresh,
                                stale_det=stale_det,
                                ads_active=ads_for_assist,
                                overlay_dot=overlay_mon,
                                capture_ms=capture_ms,
                                detect_ms=detect_ms,
                                total_loop_ms=loop_ms,
                                achieved_fps=ach_fps,
                            )

                    elif ads_for_assist and self._trace_pull and pull_target is None:
                        gate_result = MouseGateResult(True, "")
                        loop_ms = (time.perf_counter() - t0) * 1000.0
                        ach_fps = 1000.0 / loop_ms if loop_ms > 0.1 else 0.0
                        self._emit_pull_trace(
                            cfg,
                            frame_cx=frame_cx,
                            frame_cy=frame_cy,
                            target=target,
                            motion=motion,
                            pr_dx=0,
                            pr_dy=0,
                            pr_mag=0.0,
                            pr_vel=(0.0, 0.0),
                            pr_desired=(0.0, 0.0),
                            gate_allowed=True,
                            gate_reason="no pull_target",
                            mouse_move=(0, 0),
                            detection_fresh=detection_fresh,
                            stale_det=stale_det,
                            ads_active=ads_for_assist,
                            capture_ms=capture_ms,
                            detect_ms=detect_ms,
                            total_loop_ms=loop_ms,
                            achieved_fps=ach_fps,
                        )

                    elif (not ads_for_assist or paused or target is None) and self._pull is not None:
                        self._pull.reset()
                        with self._lock:
                            self._last_pull_dx = 0
                            self._last_pull_dy = 0

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
                                locked=(
                                    self._locked_target is not None
                                    and ads_for_assist
                                ),
                                has_target=detection_fresh,
                                mouse_backend=self._mouse.name,
                                candidates=det.candidates,
                                capture_size=(cap_region.width, cap_region.height),
                            )

                    if self._overlay is not None:
                        want_fps = effective_overlay_fps(cfg)
                        if want_fps != getattr(self, "_last_overlay_fps", -1):
                            self._overlay.set_overlay_fps(want_fps)
                            self._last_overlay_fps = want_fps

                    if self._overlay is not None and self._should_run():
                        overlay_pt = None
                        # M1 (audit): hide the overlay dot after >=2 stale
                        # frames so the user does not see it parked on the
                        # last-known position when the target has moved.
                        show_overlay_dot = overlay_may_show_target(
                            target,
                            detection_fresh=detection_fresh,
                            center_y=frame_cy,
                            lock_state=self._target_lock,
                        )
                        overlay_motion = (
                            motion
                            if show_overlay_dot
                            else None
                        )
                        if (
                            overlay_motion is not None
                            and math.isfinite(overlay_motion.x)
                            and math.isfinite(overlay_motion.y)
                        ):
                            ox, oy = to_monitor_coords(
                                overlay_motion.x, overlay_motion.y, cap_region
                            )
                            fov_cx_mon = float(center_x)
                            fov_cy_mon = float(center_y)
                            odx = ox - fov_cx_mon
                            ody = oy - fov_cy_mon
                            odist = math.hypot(odx, ody)
                            # O2 (audit): clamp the overlay dot to the
                            # SMALLER of (display_fov, detect_fov) so the
                            # dot always stays inside the GREEN ring the
                            # user sees on screen. The previous clamp used
                            # detect_fov alone which is wider than the
                            # display ring when detection_fov_margin_pixels
                            # is non-zero — that's why the user saw the
                            # dot pop outside the visible ring.
                            fov_limit = max(
                                1.0,
                                min(float(detect_fov), float(display_fov)),
                            ) * 0.96
                            if math.isfinite(odist) and odist > fov_limit and odist > 0.0:
                                s = fov_limit / odist
                                ox = fov_cx_mon + odx * s
                                oy = fov_cy_mon + ody * s
                            if math.isfinite(ox) and math.isfinite(oy):
                                # Mild EMA on the overlay-clamped point so the
                                # dot does not flicker in/out when the centroid
                                # oscillates across the FOV ring boundary.
                                dot_alpha = float(
                                    cfg.get("overlay_dot_smooth_alpha", 0.62)
                                )
                                sx, sy = self._aim_tracker.smooth_overlay_point(
                                    ox, oy, alpha=dot_alpha
                                )
                                # O3 (audit): re-clamp AFTER the EMA so
                                # boundary-motion drift can't drag the
                                # dot outside the ring on a smoothed
                                # frame.
                                sodx = sx - fov_cx_mon
                                sody = sy - fov_cy_mon
                                sodist = math.hypot(sodx, sody)
                                if math.isfinite(sodist) and sodist > fov_limit and sodist > 0.0:
                                    scale_r = fov_limit / sodist
                                    sx = fov_cx_mon + sodx * scale_r
                                    sy = fov_cy_mon + sody * scale_r
                                overlay_pt = (sx, sy)
                        else:
                            self._aim_tracker.reset_overlay_smoothing()
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
                            detect_fov,
                            frame_cx,
                            frame_cy,
                            magnetism_radius=int(cfg["magnetism_radius_pixels"]),
                            hsv_ranges=hsv_ranges,
                            stats_lines=self._stats.format_lines() if self._stats else [],
                            detection_debug=det.debug_lines if hasattr(det, "debug_lines") else None,
                            # O1 (audit): draw the green ring at the
                            # SAME radius as the live overlay so the
                            # debug window can't show a phantom second
                            # ring at the detection FOV. The second
                            # detect-FOV ring is opt-in via cfg flag.
                            display_fov_radius=display_fov,
                            debug_show_detect_ring=bool(
                                cfg.get("debug_show_detect_ring", False)
                            ),
                            # PHASE-7 AUDIT FIX (MED10): pass the LIVE
                            # detection mode so the debug viewer's tint
                            # mask matches what the runtime sees.
                            detection_mode=str(cfg.get("detection_mode", "apex")),
                            exclude_bottom_frac=viewmodel_exclude_bottom(cfg),
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
