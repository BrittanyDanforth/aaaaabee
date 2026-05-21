"""ABA UI status — derived from process presence + runtime telemetry only."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AbaStatus(str, Enum):
    GAME_CLOSED = "GAME CLOSED"
    TARGET_DETECTED = "TARGET DETECTED"
    IDLE = "IDLE"
    ACTIVE_SIMULATED = "ACTIVE SIMULATED"
    ACTIVE_LIVE = "ACTIVE LIVE"
    STOPPING = "STOPPING"
    ERROR = "ERROR"

    # Back-compat for tests referencing old name
    ACTIVE = "ACTIVE SIMULATED"


@dataclass(frozen=True)
class RuntimeSnapshot:
    running: bool = False
    stopping: bool = False
    paused: bool = False
    ads: bool = False
    has_target: bool = False
    frame_has_target: bool = False
    locked: bool = False
    stats_valid: bool = False
    live_input_enabled: bool = False
    dry_run: bool = True
    fps: float = -1.0
    frame_ms: float = -1.0
    configured_capture_fps: int = -1
    dropped_frames: int = -1
    stale_frames: int = -1
    confidence: float = -1.0
    pull_px: float = -1.0
    simulated_pull_px: float = -1.0
    pull_strength: float = -1.0
    distance_px: float = -1.0
    candidates: int = -1
    mouse_backend: str = ""
    capture_size: str = ""
    hooks_enabled: bool = False
    kill_switch_active: bool = False
    ads_input_mode: str = ""
    process_detection: str = ""
    last_gate_block: str = ""
    benchmark_summary: str = ""
    thread_alive: bool = False
    mouse_armed: bool = False
    error_message: str = ""
    stop_warning: str = ""
    # Extended live debug (optional; -1 / empty when unavailable)
    body_shape_score: float = -1.0
    head_score: float = -1.0
    torso_score: float = -1.0
    limb_stack_score: float = -1.0
    reject_reason: str = ""
    anchor_x: float = -1.0
    anchor_y: float = -1.0
    bbox_x: int = -1
    bbox_y: int = -1
    bbox_w: int = -1
    bbox_h: int = -1
    capture_ms: float = -1.0
    detect_ms: float = -1.0
    motion_lag_ms: float = -1.0
    pull_dx: float = -1.0
    pull_dy: float = -1.0
    mouse_gate_allowed: bool = False
    detection_fresh: bool = False


def resolve_aba_status(
    *,
    process_name: str,
    process_running: bool,
    process_required: bool,
    runtime: RuntimeSnapshot,
) -> AbaStatus:
    if runtime.error_message:
        return AbaStatus.ERROR

    if runtime.stopping and runtime.thread_alive:
        return AbaStatus.STOPPING

    proc_configured = bool(process_name.strip())

    if not runtime.running and not runtime.thread_alive:
        if proc_configured and not process_running:
            return AbaStatus.GAME_CLOSED
        if proc_configured and process_running:
            return AbaStatus.TARGET_DETECTED
        return AbaStatus.TARGET_DETECTED

    if runtime.running and runtime.paused and proc_configured:
        return AbaStatus.GAME_CLOSED

    assist_active = runtime.frame_has_target
    if assist_active and (runtime.ads or runtime.dry_run):
        if runtime.dry_run:
            return AbaStatus.ACTIVE_SIMULATED
        return AbaStatus.ACTIVE_LIVE
    return AbaStatus.IDLE
