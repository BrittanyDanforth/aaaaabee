"""ApexAimBot PID / subtick / recoil — single implementation (no duplicate runtime paths)."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from apexaimbot_bridge import (
    ApexAimBotRuntime,
    apex_pid_errors,
    in_lock_box,
    pid_mouse_delta,
    reset_apexaimbot_pid,
)
from detector import Target
from pull import PullResult


class MouseMoveFn(Protocol):
    def __call__(self, dx: int, dy: int, *, recoil_only: bool = False) -> Any: ...


class RecoilTickFn(Protocol):
    def __call__(self, cfg: dict[str, Any], *, skip_x: bool) -> bool: ...


@dataclass(frozen=True)
class ApexAimSettings:
    """Resolved per-frame Apex aim flags from config + loop state."""

    active: bool
    subtick_hz: int
    recoil_enabled: bool
    hip_fire: bool

    @classmethod
    def from_cfg(
        cls,
        cfg: dict[str, Any],
        *,
        firing: bool,
        ads_live: bool,
    ) -> ApexAimSettings:
        det = str(cfg.get("detection_mode", "apex")).strip().lower()
        pull = str(cfg.get("pull_mode", "aba")).strip().lower()
        active = pull == "apexaimbot_pid" and det == "yolo"
        return cls(
            active=active,
            subtick_hz=int(cfg.get("apex_pid_subtick_hz", 0) or 0) if active else 0,
            recoil_enabled=active and bool(cfg.get("apexaimbot_recoil_enabled", False)),
            hip_fire=bool(firing and not ads_live),
        )


def aba_recoil_active(pull: Any | None) -> bool:
    return pull is not None and pull.recoil_pull_down_active()


def compute_apex_pid_pull(
    engine: ApexAimBotRuntime,
    *,
    target: Target,
    frame_cx: float,
    frame_cy: float,
    box_wh: tuple[float, float],
    hip_fire: bool,
    use_subticks: bool,
) -> PullResult:
    """
    Main detect-frame PID step.

    When subticks own movement, do not advance PID here (avoids double integrator).
    """
    err_x, err_y, _pos_x, pos_y = apex_pid_errors(
        target,
        frame_cx=frame_cx,
        frame_cy=frame_cy,
        aim_offset_fraction=float(engine.config.aim_offset_fraction),
    )
    bw, bh = box_wh
    if not in_lock_box(
        engine,
        error_x=err_x,
        error_y=err_y,
        box_width=bw,
        box_height=bh,
        hip_fire=hip_fire,
        raw_offset_y=pos_y,
    ):
        reset_apexaimbot_pid(engine)
        return PullResult(0, 0, 0.0, 0.0, 0.0)
    if use_subticks:
        return PullResult(
            0,
            0,
            0.0,
            1.0,
            float(math.hypot(err_x, err_y)),
        )
    pdx, pdy = pid_mouse_delta(
        engine, error_x=err_x, error_y=err_y, hip_fire=hip_fire
    )
    return PullResult(
        pdx,
        pdy,
        float(math.hypot(pdx, pdy)),
        1.0,
        float(math.hypot(err_x, err_y)),
    )


def run_apex_subtick_window(
    engine: ApexAimBotRuntime,
    cfg: dict[str, Any],
    *,
    target: Target,
    frame_cx: float,
    frame_cy: float,
    box_wh: tuple[float, float],
    hip_fire: bool,
    deadline: float,
    subtick_hz: int,
    pid_enabled: bool,
    recoil_enabled: bool,
    is_firing: Callable[[], bool],
    mouse_move: MouseMoveFn,
    recoil_tick: RecoilTickFn,
    on_pid_moved: Callable[[], None],
    sleep: Callable[[float], None],
) -> None:
    """PID + recoil between detect frames; stops when LMB released or deadline hit."""
    if subtick_hz <= 0:
        return
    sub_dt = 1.0 / float(subtick_hz)
    bw, bh = box_wh
    next_tick = time.perf_counter()
    while next_tick < deadline:
        if not is_firing():
            break
        wait = next_tick - time.perf_counter()
        if wait > 0.0005:
            sleep(min(wait, max(0.0, deadline - time.perf_counter())))
        if time.perf_counter() >= deadline:
            break
        moved_x = False
        if pid_enabled:
            err_x, err_y, _pos_x, pos_y = apex_pid_errors(
                target,
                frame_cx=frame_cx,
                frame_cy=frame_cy,
                aim_offset_fraction=float(engine.config.aim_offset_fraction),
            )
            if in_lock_box(
                engine,
                error_x=err_x,
                error_y=err_y,
                box_width=bw,
                box_height=bh,
                hip_fire=hip_fire,
                raw_offset_y=pos_y,
            ):
                pdx, pdy = pid_mouse_delta(
                    engine, error_x=err_x, error_y=err_y, hip_fire=hip_fire
                )
                if pdx or pdy:
                    gate = mouse_move(pdx, pdy, recoil_only=False)
                    if getattr(gate, "allowed", True):
                        on_pid_moved()
                        moved_x = bool(pdx)
            else:
                reset_apexaimbot_pid(engine)
        if recoil_enabled and is_firing():
            recoil_tick(cfg, skip_x=moved_x)
        next_tick += sub_dt


def resolve_apex_aim_point(
    target: Target,
    pull_target: Target,
    *,
    detection_fresh: bool,
) -> tuple[float, float]:
    pt = target if detection_fresh else pull_target
    return float(pt.centroid_x), float(pt.centroid_y)
