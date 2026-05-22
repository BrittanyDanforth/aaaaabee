"""ABA desktop UI — simplified Basic/Advanced layout with presets."""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Any, Callable

from aba_status import AbaStatus, RuntimeSnapshot, resolve_aba_status
from ban_safety import (
    BAN_ACK_MESSAGE,
    DRY_RUN_FPS_NOTE,
    block_extra_capture_reason,
    dry_run_mode,
    live_assist_enabled,
)
from debug_hud import launch_debug_hud
from path_utils import APP_ROOT, LOGS_DIR, SELFCHECK_LOG, SETUP_LOG, open_logs_folder
from perf_benchmark import run_perf_benchmark
from process_presence import ProcessPresenceDebouncer
from profiles import APEX_PROCESS_NAME, effective_capture_fps, normalize_profile_name
from runtime_controller import RuntimeController
from self_check import run_self_check_detailed
from setup_doctor import format_report, run_setup_doctor

logger = logging.getLogger("aba.gui")

POLL_MS = 150
UI_BG = "#141414"
UI_PANEL = "#1c1c1c"
UI_SIDEBAR = "#101010"
UI_ACCENT = "#2a2a2a"
UI_TEXT = "#e8e8e8"
UI_MUTED = "#9a9a9a"
UI_SLIDER = "#ffffff"
UI_ACTIVE_TAB = "#2e2e2e"
UI_PRESET_BG = "#1a3a2a"
UI_PRESET_FG = "#7dffb0"

STATUS_COLORS = {
    AbaStatus.GAME_CLOSED: ("#2e2e2e", "#b0b0b0"),
    AbaStatus.TARGET_DETECTED: ("#1a4d32", "#7dffb0"),
    AbaStatus.IDLE: ("#4a4020", "#ffd966"),
    AbaStatus.ACTIVE_SIMULATED: ("#2a3550", "#9ec8ff"),
    AbaStatus.ACTIVE_LIVE: ("#4a1818", "#ff9999"),
    AbaStatus.STOPPING: ("#4a3518", "#ffcc66"),
    AbaStatus.ERROR: ("#4a1818", "#ff8888"),
}

TUNING_PRESETS: dict[str, dict[str, Any]] = {
    "Stable": {
        "pull_strength": 0.55,
        "smoothing_tau_still": 0.060,
        "smoothing_tau_moving": 0.030,
        "velocity_smoothing": 0.55,
        "max_pull_speed_pixels_per_frame": 20.0,
        "torso_aim_fraction": 0.38,
        "target_stickiness_pixels": 100,
        "body_shape_min_score": 0.50,
        "deadzone_pixels": 4,
        "detection_motion_assist": True,
        "detection_motion_threshold": 12,
    },
    "Responsive": {
        "pull_strength": 0.88,
        "smoothing_tau_still": 0.034,
        "smoothing_tau_moving": 0.014,
        "velocity_smoothing": 0.42,
        "max_pull_speed_pixels_per_frame": 30.0,
        "torso_aim_fraction": 0.40,
        "target_stickiness_pixels": 60,
        "body_shape_min_score": 0.40,
        "deadzone_pixels": 2,
        "detection_motion_assist": True,
        "detection_motion_threshold": 9,
    },
    "Strong": {
        "pull_strength": 1.10,
        "smoothing_tau_still": 0.026,
        "smoothing_tau_moving": 0.010,
        "velocity_smoothing": 0.32,
        "max_pull_speed_pixels_per_frame": 38.0,
        "torso_aim_fraction": 0.42,
        "target_stickiness_pixels": 45,
        "body_shape_min_score": 0.35,
        "deadzone_pixels": 1,
        "detection_motion_assist": True,
        "detection_motion_threshold": 8,
    },
    "Debug": {
        "pull_strength": 0.88,
        "smoothing_tau_still": 0.034,
        "smoothing_tau_moving": 0.014,
        "velocity_smoothing": 0.42,
        "max_pull_speed_pixels_per_frame": 30.0,
        "torso_aim_fraction": 0.40,
        "target_stickiness_pixels": 60,
        "body_shape_min_score": 0.40,
        "deadzone_pixels": 2,
        "detection_motion_assist": True,
        "detection_motion_threshold": 9,
        "enable_overlay": True,
        "trace_pull": True,
        "verbose_logging": True,
    },
}


def _fmt_num(value: float, *, suffix: str = "", precision: int = 1) -> str:
    if value < 0:
        return "—"
    return f"{value:.{precision}f}{suffix}"


def _fmt_int(value: int) -> str:
    if value < 0:
        return "—"
    return str(value)


class _ConfigControl:
    def __init__(
        self,
        parent: tk.Widget,
        label: str,
        key: str,
        *,
        minimum: float,
        maximum: float,
        resolution: float,
        default: float,
        is_int: bool = False,
        on_change: Callable[[str, float], None],
        tooltip: str = "",
    ) -> None:
        self.key = key
        self.frame = tk.Frame(parent, bg=UI_PANEL)
        self.frame.pack(fill=tk.X, pady=3)
        lbl_text = label
        if tooltip:
            lbl_text = f"{label}  ({tooltip})"
        tk.Label(self.frame, text=lbl_text, bg=UI_PANEL, fg=UI_MUTED, font=("Segoe UI", 9)).pack(
            anchor="w"
        )
        inner = tk.Frame(self.frame, bg=UI_PANEL)
        inner.pack(fill=tk.X)
        self._var = tk.DoubleVar(value=default)
        scale = tk.Scale(
            inner,
            from_=minimum,
            to=maximum,
            resolution=resolution,
            orient=tk.HORIZONTAL,
            variable=self._var,
            bg=UI_PANEL,
            fg=UI_TEXT,
            troughcolor=UI_ACCENT,
            activebackground=UI_SLIDER,
            highlightthickness=0,
            sliderrelief=tk.FLAT,
            showvalue=False,
            command=lambda _v: on_change(key, self.value()),
        )
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._val_label = tk.Label(
            inner,
            text=str(default),
            bg=UI_PANEL,
            fg=UI_TEXT,
            font=("Consolas", 9),
            width=8,
        )
        self._val_label.pack(side=tk.RIGHT, padx=(8, 0))
        self._is_int = is_int

    def value(self) -> float:
        v = float(self._var.get())
        return int(round(v)) if self._is_int else round(v, 4)

    def set(self, value: float) -> None:
        self._var.set(value)
        self._val_label.config(text=str(self.value()))


class AbaApplication:
    def __init__(self, config: dict, config_path: Path) -> None:
        self.config = config
        self.config_path = config_path
        self._controller = RuntimeController(config, config_path)
        self._process_name = str(config.get("target_process_name", APEX_PROCESS_NAME))
        self._process_required = bool(config.get("target_process_required", False))
        self._profile = normalize_profile_name(str(config.get("profile", "apex_style_dry_run")))
        self._configured_fps = effective_capture_fps(config)
        self._ads_mode = str(config.get("ads_input_mode", "both"))
        self._monitor_index = int(config.get("monitor_index", 1))
        self._kill_key = str(config.get("kill_switch_key", "f8")).upper()
        self._debug_proc: subprocess.Popen | None = None
        self._self_check_result = "Not run yet."
        self._setup_status = self._detect_setup_status()
        self._detail_override = ""
        self._proc_debounce = self._controller.process_debounce
        self._benchmark_running = False
        self._closing = False
        self._sliders: dict[str, _ConfigControl] = {}
        self._bool_vars: dict[str, tk.BooleanVar] = {}
        self._active_tab = "basic"
        self._advanced_mode = False

        self._root = tk.Tk()
        self._root.title("ABA")
        self._root.geometry("920x720")
        self._root.minsize(860, 640)
        self._root.configure(bg=UI_BG)
        try:
            self._root.attributes("-topmost", False)
        except tk.TclError:
            pass

        header = tk.Frame(self._root, bg=UI_BG, padx=16, pady=12)
        header.pack(fill=tk.X)
        tk.Label(header, text="ABA", bg=UI_BG, fg=UI_TEXT, font=("Segoe UI", 20, "bold")).pack(
            side=tk.LEFT
        )
        tk.Label(
            header,
            text="Apex reference · external overlay · hold RMB to ADS",
            bg=UI_BG,
            fg=UI_MUTED,
            font=("Segoe UI", 9),
        ).pack(side=tk.LEFT, padx=(12, 0))

        self._adv_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            header,
            text="Advanced",
            variable=self._adv_var,
            bg=UI_BG,
            fg=UI_MUTED,
            selectcolor=UI_ACCENT,
            activebackground=UI_BG,
            activeforeground=UI_TEXT,
            font=("Segoe UI", 9),
            command=self._toggle_advanced,
        ).pack(side=tk.RIGHT, padx=(0, 8))

        ban_frame = tk.Frame(self._root, bg="#4a1515", padx=12, pady=6)
        ban_frame.pack(fill=tk.X, padx=12, pady=(0, 4))
        tk.Label(
            ban_frame,
            text="BAN RISK: capture + hooks + synthetic mouse on live EAC. Offline/private only.",
            font=("Segoe UI", 8, "bold"),
            fg="#ffcccc",
            bg="#4a1515",
            wraplength=860,
            justify=tk.LEFT,
        ).pack(anchor="w")

        body = tk.Frame(self._root, bg=UI_BG)
        body.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)

        sidebar = tk.Frame(body, bg=UI_SIDEBAR, width=130)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)
        self._tab_buttons: dict[str, tk.Button] = {}
        self._tab_defs = [
            ("basic", "Basic"),
            ("body", "Body Targeting"),
            ("motion", "Motion"),
            ("debug", "Debug"),
            ("setup", "Setup"),
        ]
        self._advanced_tabs = {
            ("aim_adv", "Aim Advanced"),
            ("detector_adv", "Detector Adv"),
            ("overlay_adv", "Overlay Adv"),
        }
        for tab_id, title in self._tab_defs:
            btn = tk.Button(
                sidebar,
                text=title,
                anchor="w",
                padx=10,
                pady=8,
                bg=UI_SIDEBAR,
                fg=UI_MUTED,
                activebackground=UI_ACTIVE_TAB,
                activeforeground=UI_TEXT,
                relief=tk.FLAT,
                borderwidth=0,
                font=("Segoe UI", 9),
                command=lambda t=tab_id: self._show_tab(t),
            )
            btn.pack(fill=tk.X)
            self._tab_buttons[tab_id] = btn

        self._adv_tab_btns: dict[str, tk.Button] = {}
        for tab_id, title in self._advanced_tabs:
            btn = tk.Button(
                sidebar,
                text=title,
                anchor="w",
                padx=10,
                pady=8,
                bg=UI_SIDEBAR,
                fg=UI_MUTED,
                activebackground=UI_ACTIVE_TAB,
                activeforeground=UI_TEXT,
                relief=tk.FLAT,
                borderwidth=0,
                font=("Segoe UI", 8),
                command=lambda t=tab_id: self._show_tab(t),
            )
            self._adv_tab_btns[tab_id] = btn

        right = tk.Frame(body, bg=UI_BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0))

        status_row = tk.Frame(right, bg=UI_BG)
        status_row.pack(fill=tk.X)
        self._status_badge = tk.Frame(status_row, bg=STATUS_COLORS[AbaStatus.IDLE][0], padx=12, pady=8)
        self._status_badge.pack(fill=tk.X)
        self._status_var = tk.StringVar(value="STATUS: IDLE")
        self._status_label = tk.Label(
            self._status_badge,
            textvariable=self._status_var,
            font=("Segoe UI", 14, "bold"),
            bg=STATUS_COLORS[AbaStatus.IDLE][0],
            fg=STATUS_COLORS[AbaStatus.IDLE][1],
        )
        self._status_label.pack(anchor="w")
        self._detail_var = tk.StringVar(value="Press Start ABA to run overlay runtime.")
        tk.Label(
            right,
            textvariable=self._detail_var,
            bg=UI_BG,
            fg=UI_MUTED,
            wraplength=680,
            justify=tk.LEFT,
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(4, 6))

        self._panels: dict[str, tk.Frame] = {}
        scroll_host = tk.Frame(right, bg=UI_BG)
        scroll_host.pack(fill=tk.BOTH, expand=True)
        all_tabs = [t[0] for t in self._tab_defs] + [t[0] for t in self._advanced_tabs]
        for tab_id in all_tabs:
            panel = tk.Frame(scroll_host, bg=UI_PANEL, padx=14, pady=10)
            self._panels[tab_id] = panel

        self._build_basic_panel(self._panels["basic"])
        self._build_body_panel(self._panels["body"])
        self._build_motion_panel(self._panels["motion"])
        self._build_debug_panel(self._panels["debug"])
        self._build_setup_panel(self._panels["setup"])
        self._build_aim_adv_panel(self._panels["aim_adv"])
        self._build_detector_adv_panel(self._panels["detector_adv"])
        self._build_overlay_adv_panel(self._panels["overlay_adv"])
        self._show_tab("basic")

        live = tk.LabelFrame(
            right,
            text=" Live telemetry ",
            bg=UI_PANEL,
            fg=UI_MUTED,
            font=("Segoe UI", 9),
            padx=8,
            pady=6,
            labelanchor="nw",
        )
        live.pack(fill=tk.X, pady=(6, 0))
        self._live_vars: dict[str, tk.StringVar] = {}
        live_grid = tk.Frame(live, bg=UI_PANEL)
        live_grid.pack(fill=tk.X)
        live_rows = [
            ("Target", "has_target"),
            ("Confidence", "conf"),
            ("Body score", "body"),
            ("Head / Torso / Limb", "htl"),
            ("Reject", "reject"),
            ("Anchor", "anchor"),
            ("BBox", "bbox"),
            ("FPS / detect ms", "timing"),
            ("Pull dx/dy", "pull"),
            ("Gate", "gate"),
        ]
        for i, (label, key) in enumerate(live_rows):
            r, c = divmod(i, 2)
            tk.Label(live_grid, text=f"{label}:", bg=UI_PANEL, fg=UI_MUTED, font=("Segoe UI", 8)).grid(
                row=r, column=c * 2, sticky="w", padx=(0, 4), pady=1
            )
            var = tk.StringVar(value="—")
            self._live_vars[key] = var
            tk.Label(live_grid, textvariable=var, bg=UI_PANEL, fg=UI_TEXT, font=("Consolas", 8)).grid(
                row=r, column=c * 2 + 1, sticky="w", padx=(0, 14)
            )

        btn = tk.Frame(right, bg=UI_BG)
        btn.pack(fill=tk.X, pady=8)
        self._start_btn = tk.Button(
            btn,
            text="Start ABA",
            command=self._on_start,
            bg="#2d5a3d",
            fg=UI_TEXT,
            relief=tk.FLAT,
            padx=14,
            pady=6,
        )
        self._start_btn.pack(side=tk.LEFT, padx=(0, 8))
        self._stop_btn = tk.Button(
            btn,
            text="Stop ABA",
            command=self._on_stop,
            state=tk.DISABLED,
            bg="#5a2d2d",
            fg=UI_TEXT,
            relief=tk.FLAT,
            padx=14,
            pady=6,
        )
        self._stop_btn.pack(side=tk.LEFT, padx=(0, 8))
        tk.Button(
            btn,
            text="Save Settings",
            command=self._on_save_settings,
            bg=UI_ACCENT,
            fg=UI_TEXT,
            relief=tk.FLAT,
            padx=10,
            pady=6,
        ).pack(side=tk.LEFT, padx=(0, 8))
        tk.Button(
            btn,
            text="Debug HUD",
            command=self._on_debug,
            bg=UI_ACCENT,
            fg=UI_TEXT,
            relief=tk.FLAT,
            padx=10,
            pady=6,
        ).pack(side=tk.LEFT, padx=(0, 8))
        tk.Button(
            btn,
            text="Close",
            command=self._on_close,
            bg=UI_ACCENT,
            fg=UI_TEXT,
            relief=tk.FLAT,
            padx=10,
            pady=6,
        ).pack(side=tk.RIGHT)

        self._error_var = tk.StringVar(value="")
        tk.Label(right, textvariable=self._error_var, bg=UI_BG, fg="#cc6666", wraplength=680).pack(
            anchor="w"
        )

        self._root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._sync_controls_from_config()
        self._poll()

    def _slider(
        self,
        parent: tk.Widget,
        label: str,
        key: str,
        *,
        minimum: float,
        maximum: float,
        resolution: float = 0.01,
        is_int: bool = False,
        tooltip: str = "",
    ) -> _ConfigControl:
        default = float(self.config.get(key, minimum))
        ctrl = _ConfigControl(
            parent,
            label,
            key,
            minimum=minimum,
            maximum=maximum,
            resolution=resolution,
            default=default,
            is_int=is_int,
            on_change=self._on_slider_change,
            tooltip=tooltip,
        )
        self._sliders[key] = ctrl
        return ctrl

    def _toggle(self, parent: tk.Widget, label: str, key: str) -> None:
        var = tk.BooleanVar(value=bool(self.config.get(key, False)))
        self._bool_vars[key] = var
        row = tk.Frame(parent, bg=UI_PANEL)
        row.pack(fill=tk.X, pady=2)
        tk.Checkbutton(
            row,
            text=label,
            variable=var,
            bg=UI_PANEL,
            fg=UI_TEXT,
            selectcolor=UI_ACCENT,
            activebackground=UI_PANEL,
            activeforeground=UI_TEXT,
            command=lambda k=key: self._on_bool_change(k),
        ).pack(anchor="w")

    def _on_slider_change(self, key: str, _value: float) -> None:
        if key in self._sliders:
            self._sliders[key]._val_label.config(text=str(self._sliders[key].value()))
        patch = {key: self._sliders[key].value() for key in (key,)}
        try:
            self.config = self._controller.apply_config_patch(patch, persist=False)
        except Exception as exc:
            self._error_var.set(f"Config update: {exc}")

    def _on_bool_change(self, key: str) -> None:
        patch = {key: bool(self._bool_vars[key].get())}
        try:
            self.config = self._controller.apply_config_patch(patch, persist=False)
        except Exception as exc:
            self._error_var.set(f"Config update: {exc}")

    def _sync_controls_from_config(self) -> None:
        for key, ctrl in self._sliders.items():
            if key in self.config:
                ctrl.set(float(self.config[key]))
        for key, var in self._bool_vars.items():
            var.set(bool(self.config.get(key, False)))

    def _show_tab(self, tab_id: str) -> None:
        self._active_tab = tab_id
        for tid, panel in self._panels.items():
            panel.pack_forget()
        self._panels[tab_id].pack(fill=tk.BOTH, expand=True)
        for tid, btn in self._tab_buttons.items():
            if tid == tab_id:
                btn.config(bg=UI_ACTIVE_TAB, fg=UI_TEXT)
            else:
                btn.config(bg=UI_SIDEBAR, fg=UI_MUTED)
        for tid, btn in self._adv_tab_btns.items():
            if tid == tab_id:
                btn.config(bg=UI_ACTIVE_TAB, fg=UI_TEXT)
            else:
                btn.config(bg=UI_SIDEBAR, fg=UI_MUTED)

    def _toggle_advanced(self) -> None:
        self._advanced_mode = bool(self._adv_var.get())
        if self._advanced_mode:
            for tab_id, btn in self._adv_tab_btns.items():
                btn.pack(fill=tk.X)
        else:
            for tab_id, btn in self._adv_tab_btns.items():
                btn.pack_forget()
            if self._active_tab in [t[0] for t in self._advanced_tabs]:
                self._show_tab("basic")

    def _section(self, parent: tk.Widget, title: str) -> tk.Frame:
        tk.Label(parent, text=title, bg=UI_PANEL, fg=UI_TEXT, font=("Segoe UI", 11, "bold")).pack(
            anchor="w", pady=(0, 4)
        )
        return parent

    def _apply_preset(self, name: str) -> None:
        if name not in TUNING_PRESETS:
            return
        preset = TUNING_PRESETS[name]
        try:
            self.config = self._controller.apply_config_patch(preset, persist=False)
            self._sync_controls_from_config()
            self._detail_var.set(f"Preset '{name}' applied. Save Settings to keep.")
        except Exception as exc:
            self._error_var.set(f"Preset failed: {exc}")

    # === BASIC TAB ===
    def _build_basic_panel(self, parent: tk.Frame) -> None:
        self._section(parent, "Quick controls")

        preset_row = tk.Frame(parent, bg=UI_PANEL)
        preset_row.pack(fill=tk.X, pady=(0, 8))
        tk.Label(preset_row, text="Presets:", bg=UI_PANEL, fg=UI_MUTED, font=("Segoe UI", 9)).pack(
            side=tk.LEFT, padx=(0, 8)
        )
        for name in TUNING_PRESETS:
            tk.Button(
                preset_row,
                text=name,
                command=lambda n=name: self._apply_preset(n),
                bg=UI_PRESET_BG,
                fg=UI_PRESET_FG,
                relief=tk.FLAT,
                padx=10,
                pady=3,
                font=("Segoe UI", 9),
            ).pack(side=tk.LEFT, padx=2)

        self._slider(
            parent, "Tracking Strength", "pull_strength",
            minimum=0.2, maximum=1.2,
            tooltip="how hard the aim pulls toward target",
        )
        self._slider(
            parent, "Smoothness", "smoothing_tau_still",
            minimum=0.02, maximum=0.15, resolution=0.002,
            tooltip="higher = smoother but laggier",
        )
        self._slider(
            parent, "Moving Target Response", "smoothing_tau_moving",
            minimum=0.01, maximum=0.10, resolution=0.002,
            tooltip="how fast aim catches a strafing target",
        )
        self._slider(
            parent, "Aim Height", "torso_aim_fraction",
            minimum=0.32, maximum=0.52,
            tooltip="where on body: 0.35=upper chest, 0.50=belly",
        )
        self._slider(
            parent, "Stickiness", "target_stickiness_pixels",
            minimum=20.0, maximum=140.0, resolution=1.0, is_int=True,
            tooltip="px hysteresis before switching targets",
        )
        self._toggle(parent, "Show overlay (red dot + FOV ring)", "enable_overlay")

    # === BODY TARGETING TAB ===
    def _build_body_panel(self, parent: tk.Frame) -> None:
        self._section(parent, "Body targeting")
        self._slider(
            parent, "Body Shape Strictness", "body_shape_min_score",
            minimum=0.28, maximum=0.65,
            tooltip="higher = fewer false positives, may miss small targets",
        )
        self._slider(
            parent, "Aim band top", "aim_body_y_min_fraction",
            minimum=0.20, maximum=0.45,
            tooltip="upper limit of aim clamp on body (lower = higher aim)",
        )
        self._slider(
            parent, "Aim band bottom", "aim_body_y_max_fraction",
            minimum=0.40, maximum=0.60,
            tooltip="lower limit of aim clamp on body",
        )
        self._slider(
            parent, "Min target area (px)", "min_target_area_pixels",
            minimum=10, maximum=120, resolution=1, is_int=True,
            tooltip="reject blobs smaller than this",
        )
        self._slider(
            parent, "Motion detection sensitivity", "detection_motion_threshold",
            minimum=4, maximum=40, resolution=1, is_int=True,
            tooltip="lower = picks up subtler movement (Apex strafing). raise to ignore noise.",
        )
        self._toggle(parent, "Use inter-frame motion to find low-contrast enemies", "detection_motion_assist")
        self._slider(
            parent, "Min humanoid height (px)", "humanoid_min_height_pixels",
            minimum=8, maximum=60, resolution=1,
            tooltip="reject short non-body shapes",
        )

    # === MOTION TAB ===
    def _build_motion_panel(self, parent: tk.Frame) -> None:
        self._section(parent, "Motion & pull")
        self._slider(
            parent, "Pull speed smoothing", "velocity_smoothing",
            minimum=0.05, maximum=0.95,
            tooltip="EMA on pull velocity (lower = snappier)",
        )
        self._slider(
            parent, "Max pull speed / frame", "max_pull_speed_pixels_per_frame",
            minimum=4.0, maximum=40.0,
            tooltip="cap on mouse movement per frame",
        )
        self._slider(
            parent, "Deadzone", "deadzone_pixels",
            minimum=0.0, maximum=20.0,
            tooltip="no pull inside this radius of crosshair",
        )
        self._slider(
            parent, "Magnetism radius", "magnetism_radius_pixels",
            minimum=30, maximum=120, resolution=1, is_int=True,
            tooltip="distance-based pull scaling radius",
        )
        self._slider(
            parent, "Lost frames before unlock", "target_lost_frames_before_unlock",
            minimum=3, maximum=40, resolution=1, is_int=True,
            tooltip="frames without detection before dropping lock",
        )
        self._slider(
            parent, "Stale grace frames", "mouse_gate_stale_grace_frames",
            minimum=0, maximum=30, resolution=1, is_int=True,
            tooltip="frames mouse can still move after losing detection",
        )

    # === DEBUG TAB ===
    def _build_debug_panel(self, parent: tk.Frame) -> None:
        self._section(parent, "Debug & diagnostics")
        tk.Label(
            parent,
            text="Live telemetry shows real values from the running pipeline below.",
            bg=UI_PANEL, fg=UI_MUTED, font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(0, 6))
        self._toggle(parent, "Enable overlay (click-through)", "enable_overlay")
        self._toggle(parent, "Verbose logging", "verbose_logging")
        self._toggle(parent, "Pull trace log", "trace_pull")
        tk.Button(
            parent,
            text="Save debug frame (next capture)",
            command=self._on_save_debug_frame,
            bg=UI_ACCENT,
            fg=UI_TEXT,
            relief=tk.FLAT,
            padx=10,
            pady=5,
        ).pack(anchor="w", pady=8)
        tk.Label(
            parent,
            text=f"Debug frames: {self.config.get('debug_frames_dir', 'artifacts/debug_frames')}",
            bg=UI_PANEL, fg=UI_MUTED, font=("Consolas", 8),
        ).pack(anchor="w")
        tk.Label(
            parent,
            text=f"Pull trace: {self.config.get('trace_pull_log_file', 'logs/pull_trace.log')}",
            bg=UI_PANEL, fg=UI_MUTED, font=("Consolas", 8),
        ).pack(anchor="w")

    # === ADVANCED: Aim ===
    def _build_aim_adv_panel(self, parent: tk.Frame) -> None:
        self._section(parent, "Advanced aim (prediction & lock)")
        tk.Label(
            parent,
            text="Prediction is auto-disabled with body-anchor mode (default). "
            "These only apply if aim_is_body_anchor is False.",
            bg=UI_PANEL, fg=UI_MUTED, font=("Segoe UI", 8), wraplength=600,
        ).pack(anchor="w", pady=(0, 6))
        self._toggle(parent, "Prediction enabled", "prediction_enabled")
        self._slider(
            parent, "Lead seconds", "prediction_lead_seconds",
            minimum=0.0, maximum=0.12, resolution=0.001,
        )
        self._slider(
            parent, "Max lead pixels", "prediction_max_pixels",
            minimum=0.0, maximum=40.0,
        )
        self._slider(
            parent, "Vertical prediction cap (px)", "prediction_vertical_cap_pixels",
            minimum=0.0, maximum=16.0, resolution=0.5,
        )

    # === ADVANCED: Detector ===
    def _build_detector_adv_panel(self, parent: tk.Frame) -> None:
        self._section(parent, "Advanced body scoring weights")
        self._slider(parent, "Head score weight", "head_score_weight", minimum=0.0, maximum=0.5)
        self._slider(parent, "Torso score weight", "torso_score_weight", minimum=0.0, maximum=0.5)
        self._slider(
            parent, "Limb stack weight", "limb_stack_score_weight",
            minimum=0.0, maximum=0.5,
        )

    # === ADVANCED: Overlay ===
    def _build_overlay_adv_panel(self, parent: tk.Frame) -> None:
        self._section(parent, "Advanced overlay & debug flags")
        self._toggle(parent, "OpenCV debug window (CLI only)", "show_debug_window")
        tk.Label(
            parent,
            text="ADS input mode (config): " + str(self.config.get("ads_input_mode", "both")),
            bg=UI_PANEL, fg=UI_TEXT, font=("Consolas", 9),
        ).pack(anchor="w", pady=4)
        tk.Label(
            parent,
            text="Trigger = hold RIGHT MOUSE BUTTON (ADS). No extra keybinds.",
            bg=UI_PANEL, fg=UI_MUTED, font=("Segoe UI", 9), wraplength=600,
        ).pack(anchor="w", pady=4)

    # === SETUP TAB ===
    def _build_setup_panel(self, parent: tk.Frame) -> None:
        self._setup_var = tk.StringVar(value=self._setup_status)
        self._mode_var = tk.StringVar(
            value="MODE: DRY RUN" if dry_run_mode(self.config) else "MODE: LIVE INPUT ENABLED"
        )
        self._proc_detect_var = tk.StringVar(value="checking…")
        self._bench_var = tk.StringVar(value="Benchmark: not run")
        self._selfcheck_var = tk.StringVar(value=self._self_check_result)

        tk.Label(parent, textvariable=self._setup_var, bg=UI_PANEL, fg=UI_MUTED, wraplength=640).pack(
            anchor="w"
        )
        tk.Label(
            parent,
            textvariable=self._mode_var,
            bg=UI_PANEL,
            fg=UI_TEXT,
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w", pady=4)
        if dry_run_mode(self.config):
            tk.Label(
                parent,
                text=DRY_RUN_FPS_NOTE,
                bg=UI_PANEL,
                fg=UI_MUTED,
                wraplength=640,
                font=("Segoe UI", 8),
            ).pack(anchor="w")
        self._live_cfg_var = tk.StringVar()
        live_on = bool(self.config.get("allow_live_mouse", False))
        ack = bool(self.config.get("offline_dev_mode", True))
        self._live_cfg_var.set(
            f"Live mouse: {'ON' if live_on else 'OFF (dry-run)'} · "
            f"offline_dev_mode={ack} (private-build safety ack — NOT dry-run)"
        )
        tk.Label(
            parent,
            textvariable=self._live_cfg_var,
            bg=UI_PANEL,
            fg="#7dffb0" if live_on else UI_MUTED,
            font=("Segoe UI", 9, "bold"),
            wraplength=640,
        ).pack(anchor="w", pady=(2, 4))
        tk.Label(
            parent,
            text=f"Profile: {self._profile} · FPS cap: {self._configured_fps} · Monitor: {self._monitor_index}",
            bg=UI_PANEL,
            fg=UI_TEXT,
            font=("Consolas", 9),
        ).pack(anchor="w", pady=4)
        tk.Label(
            parent,
            text=f"FOV radius (read-only): {self.config.get('fov_radius_pixels', '?')} hip / "
            f"{self.config.get('fov_radius_ads_pixels', '?')} ADS",
            bg=UI_PANEL,
            fg=UI_MUTED,
            font=("Consolas", 8),
        ).pack(anchor="w")
        tk.Label(
            parent,
            textvariable=self._proc_detect_var,
            bg=UI_PANEL,
            fg=UI_MUTED,
            font=("Consolas", 8),
        ).pack(anchor="w", pady=4)
        tk.Button(
            parent,
            text="Run Setup Doctor",
            command=self._on_setup_doctor,
            bg=UI_ACCENT,
            fg=UI_TEXT,
            relief=tk.FLAT,
        ).pack(anchor="w", pady=4)
        tk.Button(
            parent,
            text="Run Perf Benchmark",
            command=self._on_benchmark,
            bg=UI_ACCENT,
            fg=UI_TEXT,
            relief=tk.FLAT,
        ).pack(anchor="w", pady=2)
        tk.Label(parent, textvariable=self._bench_var, bg=UI_PANEL, fg=UI_MUTED, font=("Consolas", 8)).pack(
            anchor="w"
        )
        tk.Button(
            parent,
            text="Run Self-Check",
            command=self._on_self_check,
            bg=UI_ACCENT,
            fg=UI_TEXT,
            relief=tk.FLAT,
        ).pack(anchor="w", pady=8)
        tk.Label(parent, textvariable=self._selfcheck_var, bg=UI_PANEL, fg=UI_MUTED, wraplength=640).pack(
            anchor="w"
        )
        tk.Button(
            parent,
            text="Open Logs Folder",
            command=self._on_open_logs,
            bg=UI_ACCENT,
            fg=UI_TEXT,
            relief=tk.FLAT,
        ).pack(anchor="w", pady=4)
        tk.Label(
            parent,
            text=f"setup: {SETUP_LOG.name} · self-check: {SELFCHECK_LOG.name}",
            bg=UI_PANEL,
            fg=UI_MUTED,
            font=("Consolas", 8),
        ).pack(anchor="w")

    @staticmethod
    def _detect_setup_status() -> str:
        venv_py = APP_ROOT / ".venv" / "Scripts" / "python.exe"
        if venv_py.is_file():
            return f"OK: venv found ({venv_py.name})"
        return "WARN: .venv missing — run run_windows.bat first"

    def _set_status_badge(self, status: AbaStatus) -> None:
        bg, fg = STATUS_COLORS[status]
        self._status_badge.config(bg=bg)
        self._status_label.config(bg=bg, fg=fg)
        self._status_var.set(f"STATUS: {status.value}")

    def _on_save_settings(self) -> None:
        try:
            self._controller.save_config(self.config)
            self._detail_var.set(f"Saved {self.config_path.name}")
        except Exception as exc:
            self._error_var.set(f"Save failed: {exc}")

    def _on_save_debug_frame(self) -> None:
        if not self._controller.is_running:
            self._error_var.set("Start ABA first, then save debug frame on next capture.")
            return
        self._controller.request_debug_frame_save()
        self._detail_var.set("Debug frame queued — will save on next runtime frame.")

    def _on_setup_doctor(self) -> None:
        require = sys.platform == "win32"
        report = run_setup_doctor(require_venv=require)
        self._setup_var.set("Doctor: PASSED" if report.passed else "Doctor: FAILED — see logs")
        self._selfcheck_var.set(format_report(report)[:500])

    def _on_open_logs(self) -> None:
        ok, msg = open_logs_folder()
        if ok:
            self._detail_var.set(f"Opened logs: {msg}")
        else:
            self._error_var.set(f"Could not open logs: {msg}")

    def _on_start(self) -> None:
        venv_py = APP_ROOT / ".venv" / "Scripts" / "python.exe"
        if not venv_py.is_file() and sys.platform == "win32":
            self._error_var.set("No .venv — run run_windows.bat before Start ABA.")
            return
        if live_assist_enabled(self.config):
            from tkinter import messagebox

            if not messagebox.askokcancel(
                "ABA — ban risk acknowledgment",
                BAN_ACK_MESSAGE,
                icon="warning",
                parent=self._root,
            ):
                self._detail_var.set("Start cancelled — ban acknowledgment required.")
                return
        self._stop_debug_hud()
        try:
            self._controller.save_config(self.config)
            self.config = self._controller.reload_config()
            self.config["show_debug_window"] = False
            self.config["enable_overlay"] = bool(self.config.get("enable_overlay", False))
            self._configured_fps = effective_capture_fps(self.config)
            self._process_name = str(self.config.get("target_process_name", APEX_PROCESS_NAME))
            self._process_required = bool(self.config.get("target_process_required", False))
            self._ads_mode = str(self.config.get("ads_input_mode", "both"))
        except Exception as exc:
            self._error_var.set(f"Config reload failed: {exc}")
            return
        self._detail_override = ""
        ok, msg = self._controller.start()
        if ok:
            self._start_btn.config(state=tk.DISABLED)
            self._stop_btn.config(state=tk.NORMAL)
        self._detail_var.set(msg)

    def _on_benchmark(self) -> None:
        if self._controller.is_running:
            self._bench_var.set("Benchmark blocked — stop ABA first.")
            return
        if self._benchmark_running:
            return
        self._benchmark_running = True
        self._bench_var.set("Benchmark running…")

        def work() -> None:
            try:
                cfg = self._controller.reload_config()
                report = run_perf_benchmark(cfg, duration_sec=2.5)
                summary = report.summary()
                self._controller.set_benchmark_summary(summary)
            except Exception as exc:
                summary = f"FAILED: {exc}"

            def done() -> None:
                self._benchmark_running = False
                self._bench_var.set(f"Benchmark: {summary}")

            if not self._closing:
                try:
                    self._root.after(0, done)
                except tk.TclError:
                    pass

        threading.Thread(target=work, name="ABA-Benchmark", daemon=True).start()

    def _stop_debug_hud(self) -> None:
        if self._debug_proc is None or self._debug_proc.poll() is not None:
            self._debug_proc = None
            return
        self._debug_proc.terminate()
        try:
            self._debug_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self._debug_proc.kill()
        self._debug_proc = None

    def _on_stop(self) -> None:
        self._detail_override = "Stopping runtime…"
        self._controller.stop()
        self._apply_poll_state()

    def _on_debug(self) -> None:
        if self._debug_proc is not None and self._debug_proc.poll() is None:
            self._detail_var.set(f"Debug HUD running (PID {self._debug_proc.pid}).")
            return
        block = block_extra_capture_reason(self.config)
        if block:
            self._error_var.set(f"Debug HUD blocked — {block}")
            return
        ok, msg, proc = launch_debug_hud(self.config_path)
        if ok and proc:
            self._debug_proc = proc
            self._detail_var.set(msg)
        else:
            self._error_var.set(f"Debug HUD failed: {msg}")

    def _on_self_check(self) -> None:
        try:
            self.config = self._controller.reload_config()
        except Exception:
            pass
        self._selfcheck_var.set("Running self-check…")

        def work() -> None:
            result = run_self_check_detailed(self.config)
            summary = "PASSED" if result.passed else "FAILED"
            if result.skipped:
                summary += f" ({len(result.skipped)} skipped)"
            self._self_check_result = f"Self-check {summary}"
            if result.errors:
                self._self_check_result += " — " + result.errors[0][:60]

            def done() -> None:
                self._selfcheck_var.set(self._self_check_result)
                proc_running = self._proc_debounce.is_running(self._process_name)
                self._refresh_start_button(proc_running)

            if not self._closing:
                try:
                    self._root.after(0, done)
                except tk.TclError:
                    pass

        threading.Thread(target=work, name="ABA-SelfCheck", daemon=True).start()

    def _refresh_start_button(self, proc_running: bool) -> None:
        if self._controller.is_running:
            return
        if self._benchmark_running:
            self._start_btn.config(state=tk.DISABLED)
            return
        allow = proc_running or not self._process_required or not self._process_name.strip()
        self._start_btn.config(state=tk.NORMAL if allow else tk.DISABLED)

    def _update_live_cfg_label(self) -> None:
        if not hasattr(self, "_live_cfg_var"):
            return
        live_on = bool(self.config.get("allow_live_mouse", False))
        ack = bool(self.config.get("offline_dev_mode", True))
        self._live_cfg_var.set(
            f"Live mouse: {'ON' if live_on else 'OFF (dry-run)'} · "
            f"offline_dev_mode={ack} (private-build ack — required for live)"
        )

    def _update_live(self, snap: RuntimeSnapshot) -> None:
        self._live_vars["has_target"].set("yes" if snap.frame_has_target else "no")
        self._live_vars["conf"].set(_fmt_num(snap.confidence, precision=2))
        self._live_vars["body"].set(_fmt_num(snap.body_shape_score, precision=2))
        self._live_vars["htl"].set(
            f"{_fmt_num(snap.head_score, precision=2)} / "
            f"{_fmt_num(snap.torso_score, precision=2)} / "
            f"{_fmt_num(snap.limb_stack_score, precision=2)}"
        )
        self._live_vars["reject"].set(snap.reject_reason or "—")
        if snap.anchor_x >= 0:
            self._live_vars["anchor"].set(
                f"{snap.anchor_x:.0f}, {snap.anchor_y:.0f}"
            )
        else:
            self._live_vars["anchor"].set("—")
        if snap.bbox_w >= 0:
            self._live_vars["bbox"].set(
                f"{snap.bbox_x},{snap.bbox_y} {snap.bbox_w}x{snap.bbox_h}"
            )
        else:
            self._live_vars["bbox"].set("—")
        self._live_vars["timing"].set(
            f"fps {_fmt_num(snap.fps, precision=0)} · cap {_fmt_int(snap.detect_ms)}ms · "
            f"capture {_fmt_num(snap.capture_ms, precision=1)}ms"
        )
        self._live_vars["pull"].set(
            f"{_fmt_num(snap.pull_dx, precision=0)}, {_fmt_num(snap.pull_dy, precision=0)} · "
            f"pull {_fmt_num(snap.pull_px, precision=1)}px"
        )
        gate = "allowed" if snap.mouse_gate_allowed else "blocked"
        reason = snap.last_gate_block or "—"
        self._live_vars["gate"].set(f"{gate} ({reason})")

    def _apply_poll_state(self) -> None:
        proc_running = True
        if self._process_name.strip():
            proc_running = self._proc_debounce.is_running(self._process_name)
            if hasattr(self, "_proc_detect_var"):
                self._proc_detect_var.set(
                    f"{'yes' if proc_running else 'no'} — {self._process_name}"
                )
        snap = self._controller.snapshot()
        status = resolve_aba_status(
            process_name=self._process_name,
            process_running=proc_running,
            process_required=self._process_required,
            runtime=snap,
        )
        self._set_status_badge(status)
        self._update_live(snap)
        self._update_live_cfg_label()
        if hasattr(self, "_mode_var"):
            self._mode_var.set(
                "MODE: DRY RUN" if snap.dry_run else "MODE: LIVE INPUT ENABLED"
            )
        if snap.error_message:
            self._error_var.set(snap.error_message)
        elif snap.stop_warning:
            self._error_var.set(snap.stop_warning)
        else:
            self._error_var.set("")
        if self._detail_override and (status == AbaStatus.STOPPING or not snap.thread_alive):
            self._detail_var.set(self._detail_override)
        elif snap.stop_warning:
            self._detail_var.set(snap.stop_warning)
        else:
            self._detail_var.set(self._format_detail(status, snap))
        if status == AbaStatus.ERROR and not self._controller.is_running:
            self._start_btn.config(state=tk.NORMAL)
            self._stop_btn.config(state=tk.DISABLED)
            self._detail_override = ""
        elif snap.thread_alive:
            self._start_btn.config(state=tk.DISABLED)
            self._stop_btn.config(state=tk.NORMAL)
        else:
            self._refresh_start_button(proc_running)
            self._stop_btn.config(state=tk.DISABLED)
        if self._debug_proc is not None and self._debug_proc.poll() is not None:
            self._debug_proc = None

    def _on_close(self) -> None:
        self._closing = True
        if self._controller.is_running:
            self._controller.stop()
        self._stop_debug_hud()
        self._root.destroy()

    def _poll(self) -> None:
        if self._closing:
            return
        self._apply_poll_state()
        self._root.after(POLL_MS, self._poll)

    @staticmethod
    def _format_detail(status: AbaStatus, snap: RuntimeSnapshot) -> str:
        if status == AbaStatus.ERROR:
            return snap.error_message or "Unknown error"
        if status == AbaStatus.STOPPING:
            return "Stopping capture loop and input listeners…"
        if status == AbaStatus.GAME_CLOSED:
            if snap.running and snap.paused:
                return "Target process absent — assist paused."
            return "Launch game for TARGET DETECTED."
        if status == AbaStatus.TARGET_DETECTED:
            return "Process detected. Start ABA, then hold RMB in range."
        if status == AbaStatus.ACTIVE_SIMULATED:
            return "ACTIVE SIMULATED — pull math only, no OS mouse."
        if status == AbaStatus.ACTIVE_LIVE:
            return "ACTIVE LIVE — hold RMB + target. Ban risk."
        if status == AbaStatus.IDLE and snap.running:
            return "Runtime on — hold RMB for ADS + target."
        return "Press Start ABA to run overlay runtime."

    def run(self) -> None:
        self._root.mainloop()


def run_gui(config: dict, config_path: Path) -> None:
    AbaApplication(config, config_path).run()
