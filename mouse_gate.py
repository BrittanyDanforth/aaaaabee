"""Final-line mouse movement gate — last check before any OS cursor move."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ban_safety import dry_run_mode, live_assist_enabled

# Pull scales max_speed by dt (up to ~3.2x at 30 FPS) — gate must not block those moves.
_DEFAULT_PULL_BUDGET_SCALE = 3.5


@dataclass(frozen=True)
class MouseGateContext:
    running: bool
    stopping: bool
    paused: bool
    mouse_enabled: bool
    ads_active: bool
    has_target: bool
    target_process_ok: bool
    dx: int
    dy: int
    max_pull_per_frame: float
    detection_fresh: bool = True
    target_lost_frames: int = 0
    stale_grace_frames: int = 12
    pull_budget_scale: float = _DEFAULT_PULL_BUDGET_SCALE
    # ApexAimBot preset: LMB hip-fire assist without ADS
    assist_without_ads: bool = False
    # Vendored recoil pattern — not gated on target lock / stale detect
    recoil_only: bool = False


@dataclass(frozen=True)
class MouseGateResult:
    allowed: bool
    reason: str = ""


def _target_lock_ok(ctx: MouseGateContext) -> bool:
    if not ctx.has_target:
        return False
    if ctx.detection_fresh:
        return True
    if ctx.stale_grace_frames <= 0:
        return True
    return ctx.target_lost_frames <= ctx.stale_grace_frames


def pull_budget_px(ctx: MouseGateContext, config: dict[str, Any]) -> float:
    cap = max(1.0, float(ctx.max_pull_per_frame))
    scale = float(config.get("mouse_gate_pull_budget_scale", ctx.pull_budget_scale))
    return cap * max(1.0, scale)


def evaluate_mouse_gate(config: dict[str, Any], ctx: MouseGateContext) -> MouseGateResult:
    """Return whether move_relative may run. Empty reason when allowed."""
    if ctx.dx == 0 and ctx.dy == 0:
        return MouseGateResult(True, "")

    if not ctx.running:
        return MouseGateResult(False, "gate[pull]: runtime not running")
    if ctx.stopping:
        return MouseGateResult(False, "gate[pull]: stopping")
    if ctx.paused:
        return MouseGateResult(False, "gate[pull]: target paused")
    if not ctx.mouse_enabled:
        return MouseGateResult(False, "gate[pull]: mouse disabled")
    if dry_run_mode(config):
        return MouseGateResult(False, "gate[pull]: dry_run / allow_live_mouse=false")
    if not live_assist_enabled(config):
        return MouseGateResult(False, "gate[pull]: allow_live_mouse=false")
    if not ctx.ads_active and not ctx.assist_without_ads:
        return MouseGateResult(False, "gate[pull]: ADS not active (enable ADS or apexaimbot_pid+LMB)")
    if ctx.recoil_only:
        if not ctx.target_process_ok:
            return MouseGateResult(False, "gate[recoil]: target process not present")
        return MouseGateResult(True, "")
    if not _target_lock_ok(ctx):
        if ctx.has_target and not ctx.detection_fresh:
            return MouseGateResult(
                False,
                f"gate[pull]: stale detection ({ctx.target_lost_frames} lost frames)",
            )
        return MouseGateResult(False, "gate[pull]: no target lock")
    if not ctx.target_process_ok:
        return MouseGateResult(False, "gate[pull]: target process not present")

    mag = (ctx.dx * ctx.dx + ctx.dy * ctx.dy) ** 0.5
    budget = pull_budget_px(ctx, config)
    if mag > budget:
        return MouseGateResult(
            False,
            f"gate[pull]: move {mag:.1f}px exceeds budget {budget:.1f}",
        )

    return MouseGateResult(True, "")
