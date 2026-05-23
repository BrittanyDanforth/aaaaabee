"""Target lock state machine — single source of truth for runtime + tests.

There are **two cooperating layers** (not duplicates):

1. **Detection sticky** (``detector.find_best_target``): per-frame candidate
   ranking with ``sticky_target`` / ``currently_locked``. Biases scoring,
   confidence floor, sticky pool, sky/clutter rejects. Does **not** own
   grace counters or switch hysteresis frames.

2. **Frame lock** (this module — ``apply_target_lock``): owns
   ``target_lost_frames``, instant-adopt, 3-frame switch hysteresis, and
   new-lock gates. ``AssistRuntime._select_target`` calls
   ``find_best_target`` then ``apply_target_lock``.

``TargetingRuntime`` (artifact/tests) must use the same frame lock — not a
third copy. Simple ``_sticky = t`` without ``apply_target_lock`` is legacy.

Use :func:`detection_sticky_context` everywhere you call ``find_best_target``
so sticky/currently_locked cannot drift between runtime, audits, and tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

from detector import (
    DetectionResult,
    Target,
    _bbox_iou,
    bbox_mid_in_sky_band,
    target_is_background_clutter,
    target_is_viewmodel_column_fp,
)

# IoU gates (documented in tests/test_detection_hardening.py).
INSTANT_ADOPT_MIN_IOU = 0.32
HIGH_OVERLAP_REFINE_IOU = 0.45
HIGH_OVERLAP_MAX_DRIFT_PX = 36.0
SOFT_REFINE_MIN_IOU = 0.26
SWITCH_MIN_IOU = 0.28
CLUTTER_REJECT_MAX_IOU = 0.18
NEW_LOCK_CONFIRM_FRAMES = 2
NEW_LOCK_MIN_BODY = 0.58
NEW_LOCK_MIN_RED = 0.04
NEW_LOCK_MIN_PARTS = 2
OVERLAY_CONFIRM_FRAMES = 2


def viewmodel_exclude_bottom(cfg: dict[str, Any]) -> float:
    """Bottom-of-frame mask fraction — must match production runtime."""
    return float(cfg.get("viewmodel_exclude_bottom_frac", 0.28))


def _same_lock_identity(a: Target, b: Target) -> bool:
    """True when two detections are likely the same blob (confirm new lock)."""
    iou = _bbox_iou(
        a.bbox_x, a.bbox_y, a.bbox_w, a.bbox_h,
        b.bbox_x, b.bbox_y, b.bbox_w, b.bbox_h,
    )
    if iou >= 0.22:
        return True
    drift = math.hypot(a.centroid_x - b.centroid_x, a.centroid_y - b.centroid_y)
    return drift < 32.0


def _passes_new_lock_gates(target: Target, *, center_y: float) -> bool:
    """Reject crates, panels, and scope/HUD blobs before a lock can confirm."""
    if target.body_shape_score < NEW_LOCK_MIN_BODY:
        return False
    if bbox_mid_in_sky_band(target.bbox_y, target.bbox_h, center_y):
        return False
    if target.red_coverage < NEW_LOCK_MIN_RED:
        return False
    if int(target.part_count) < NEW_LOCK_MIN_PARTS:
        return False
    if target_is_background_clutter(target):
        return False
    # Solid wide props (loot crates, floor panels) often score body>0.55 with
    # one blob — require torso structure or a real head+torso stack.
    if not target.has_classified_torso:
        if target.torso_score < 0.32 or target.head_score < 0.22:
            return False
    if target.fill_ratio > 0.82 and int(target.part_count) <= 2:
        return False
    if target.max_circularity > 0.62 and int(target.part_count) <= 2:
        return False
    bw = max(1, int(target.bbox_w))
    bh = max(1, int(target.bbox_h))
    if bh < bw * 1.08:
        return False
    return True


def _passes_instant_refine_gates(target: Target) -> bool:
    """Stricter than stale refine — blocks img7 gun-column instant swaps."""
    if target.red_coverage < NEW_LOCK_MIN_RED:
        return False
    if target_is_background_clutter(target):
        return False
    if not target.has_classified_torso and target.torso_score < 0.28:
        return False
    if int(target.part_count) < 2:
        return False
    return True


def _locked_is_environment_fp(
    target: Target,
    *,
    center_y: float,
    fov_cx: float | None = None,
    fov_cy: float | None = None,
    frame_w: int = 0,
    frame_h: int = 0,
) -> bool:
    """True when an existing lock is clearly a crate/HUD/sky/viewmodel FP."""
    if target_is_background_clutter(target):
        return True
    if bbox_mid_in_sky_band(target.bbox_y, target.bbox_h, center_y):
        return True
    if (
        fov_cx is not None
        and fov_cy is not None
        and frame_w > 0
        and frame_h > 0
        and target_is_viewmodel_column_fp(
            target,
            frame_w=frame_w,
            frame_h=frame_h,
            fov_cx=float(fov_cx),
            fov_cy=float(fov_cy),
        )
    ):
        return True
    if (
        target.red_coverage < NEW_LOCK_MIN_RED
        and not target.has_classified_torso
        and target.torso_score < 0.28
    ):
        return True
    if target.fill_ratio > 0.82 and int(target.part_count) <= 2:
        return True
    if target.max_circularity > 0.62 and int(target.part_count) <= 2:
        return True
    bw = max(1, int(target.bbox_w))
    bh = max(1, int(target.bbox_h))
    if bh < bw * 1.08 and target.fill_ratio > 0.5:
        return True
    return False


def may_assist_pull_target(
    target: Target | None,
    *,
    detection_fresh: bool,
    center_y: float,
    target_lost_frames: int = 0,
    stale_grace_frames: int = 12,
) -> bool:
    """Whether mouse pull may run — separate from overlay dot confirm frames.

    Pull should stay engaged during brief detection gaps (grace) and while
    the aim anchor tracks small motion. Overlay uses stricter 2-frame confirm.
    """
    if target is None:
        return False
    if not _passes_new_lock_gates(target, center_y=center_y):
        return False
    if detection_fresh:
        return True
    if stale_grace_frames > 0 and target_lost_frames <= stale_grace_frames:
        return True
    return False


def overlay_may_show_target(
    target: Target | None,
    *,
    detection_fresh: bool,
    center_y: float,
    lock_state: TargetLockState | None = None,
) -> bool:
    """Single gate for whether the red overlay dot may render.

  When ``lock_state`` is provided (production runtime), the dot only
  appears after ``OVERLAY_CONFIRM_FRAMES`` consecutive frames pass the
  humanoid gates — kills the 1-frame ADS flash onto crates/gun column.
    """
    if lock_state is not None:
        if not detection_fresh or target is None:
            lock_state.overlay_confirm_frames = 0
            lock_state._overlay_last_cy = None
            return False
        if not _passes_new_lock_gates(target, center_y=center_y):
            lock_state.overlay_confirm_frames = 0
            lock_state._overlay_last_cy = None
            return False
        last_cy = getattr(lock_state, "_overlay_last_cy", None)
        if last_cy is not None and target.centroid_y < float(last_cy) - 22.0:
            lock_state.overlay_confirm_frames = 0
        lock_state._overlay_last_cy = float(target.centroid_y)
        lock_state.overlay_confirm_frames += 1
        return lock_state.overlay_confirm_frames >= OVERLAY_CONFIRM_FRAMES
    if not detection_fresh or target is None:
        return False
    return _passes_new_lock_gates(target, center_y=center_y)


@dataclass
class TargetLockState:
    locked_target: Target | None = None
    target_lost_frames: int = 0
    switch_candidate: Target | None = None
    switch_frames: int = 0
    new_lock_candidate: Target | None = None
    new_lock_frames: int = 0
    overlay_confirm_frames: int = 0
    _overlay_last_cy: float | None = None

    def reset(self) -> None:
        self.locked_target = None
        self.target_lost_frames = 0
        self.switch_candidate = None
        self.switch_frames = 0
        self.new_lock_candidate = None
        self.new_lock_frames = 0
        self.overlay_confirm_frames = 0
        self._overlay_last_cy = None


def detection_sticky_context(
    state: TargetLockState,
    cfg: dict[str, Any],
) -> tuple[Target | None, bool, int]:
    """Inputs for ``find_best_target`` derived from frame lock state.

    Returns ``(sticky_target, currently_locked, lost_max)``.
    """
    lost_max = int(cfg["target_lost_frames_before_unlock"])
    in_grace = state.target_lost_frames < lost_max
    sticky = state.locked_target if in_grace else None
    currently_locked = state.locked_target is not None and in_grace
    return sticky, currently_locked, lost_max


def apply_target_lock(
    state: TargetLockState,
    result: DetectionResult,
    *,
    center_y: float,
    cfg: dict[str, Any],
    on_lock_expired: Callable[[], None] | None = None,
    fov_cx: float | None = None,
    fov_cy: float | None = None,
    frame_size: tuple[int, int] | None = None,
) -> tuple[DetectionResult, bool]:
    """Merge a detector result with sticky lock state.

    Returns ``(effective_result, is_stale)``. ``is_stale`` is True when the
    effective target is the frozen lock during grace (``target_lost_frames > 0``)
    without a fresh detection on that frame.
    """
    lost_max = int(cfg["target_lost_frames_before_unlock"])
    is_stale = False
    fw, fh = (0, 0)
    if frame_size is not None and len(frame_size) >= 2:
        fw, fh = int(frame_size[0]), int(frame_size[1])

    def _env_fp(t: Target) -> bool:
        return _locked_is_environment_fp(
            t,
            center_y=center_y,
            fov_cx=fov_cx,
            fov_cy=fov_cy,
            frame_w=fw,
            frame_h=fh,
        )

    if result.target is not None:
        new_t = result.target
        new_is_env = _env_fp(new_t)
        locked = state.locked_target
        if locked is not None and _env_fp(locked):
            state.reset()
            if on_lock_expired is not None:
                on_lock_expired()
            locked = None
        if locked is not None:
            drift = math.hypot(
                new_t.centroid_x - locked.centroid_x,
                new_t.centroid_y - locked.centroid_y,
            )
            fov_lim = float(cfg.get("_runtime_detect_fov", 200) or 200) * 0.55
            bs_ratio_ok = new_t.body_shape_score >= locked.body_shape_score * 0.85
            bs_abs_ok = new_t.body_shape_score >= 0.60
            upward_fragment = (
                new_t.bbox_y < locked.bbox_y - locked.bbox_h * 0.12
                and new_t.bbox_h < locked.bbox_h * 0.92
            )
            aim_jump_up = new_t.centroid_y < locked.centroid_y - 12.0
            weak_red_adopt = aim_jump_up and new_t.red_coverage < 0.06
            upward_sky_steal = (
                aim_jump_up
                and new_t.red_coverage < max(NEW_LOCK_MIN_RED, locked.red_coverage * 0.72)
            )
            adopt_iou = _bbox_iou(
                locked.bbox_x,
                locked.bbox_y,
                locked.bbox_w,
                locked.bbox_h,
                new_t.bbox_x,
                new_t.bbox_y,
                new_t.bbox_w,
                new_t.bbox_h,
            )
            sky_band = bbox_mid_in_sky_band(new_t.bbox_y, new_t.bbox_h, center_y)
            clutter_fp = (
                target_is_background_clutter(new_t) or new_is_env or upward_sky_steal
            )
            instant_adopt_ok = (
                bs_ratio_ok
                and bs_abs_ok
                and not upward_fragment
                and not weak_red_adopt
                and not clutter_fp
                and adopt_iou >= INSTANT_ADOPT_MIN_IOU
                and not sky_band
                and _passes_instant_refine_gates(new_t)
            )
            soft_refine_ok = (
                adopt_iou >= SOFT_REFINE_MIN_IOU
                and bs_ratio_ok
                and new_t.body_shape_score >= 0.55
                and not upward_fragment
                and not weak_red_adopt
                and not clutter_fp
                and not sky_band
                and drift < 28
                and drift < fov_lim
            )
            high_overlap_refine = (
                adopt_iou >= HIGH_OVERLAP_REFINE_IOU
                and instant_adopt_ok
                and drift <= HIGH_OVERLAP_MAX_DRIFT_PX
            )
            if instant_adopt_ok and (
                high_overlap_refine
                or soft_refine_ok
                or (drift < 25 and drift < fov_lim)
            ):
                state.locked_target = new_t
                state.target_lost_frames = 0
                state.switch_candidate = None
                state.switch_frames = 0
                return result, False

            if state.target_lost_frames == 0:
                state.switch_candidate = None
                state.switch_frames = 0
                if (
                    adopt_iou >= SOFT_REFINE_MIN_IOU
                    and not clutter_fp
                    and not new_is_env
                    and _passes_instant_refine_gates(new_t)
                ):
                    state.locked_target = new_t
                    return result, False
                return (
                    DetectionResult(
                        locked,
                        result.candidates,
                        locked.confidence,
                    ),
                    False,
                )

            if clutter_fp or adopt_iou < CLUTTER_REJECT_MAX_IOU:
                state.switch_candidate = None
                state.switch_frames = 0
                state.target_lost_frames = max(1, state.target_lost_frames)
                is_stale = True
                return (
                    DetectionResult(locked, result.candidates, locked.confidence),
                    is_stale,
                )

            if (
                state.switch_candidate is not None
                and math.hypot(
                    new_t.centroid_x - state.switch_candidate.centroid_x,
                    new_t.centroid_y - state.switch_candidate.centroid_y,
                )
                < 30
            ):
                state.switch_frames += 1
            else:
                state.switch_candidate = new_t
                state.switch_frames = 1

            switch_score_ok = (
                new_t.body_shape_score >= locked.body_shape_score + 0.10
                and new_t.confidence >= locked.confidence * 0.90
                and new_t.red_coverage >= 0.05
                and not weak_red_adopt
                and not upward_sky_steal
                and not target_is_background_clutter(new_t)
                and not new_is_env
                and adopt_iou >= SWITCH_MIN_IOU
            )
            if state.switch_frames >= 3 and switch_score_ok:
                state.locked_target = new_t
                state.target_lost_frames = 0
                state.switch_candidate = None
                state.switch_frames = 0
                return result, False

            state.target_lost_frames = max(1, state.target_lost_frames)
            is_stale = True
            return (
                DetectionResult(locked, result.candidates, locked.confidence),
                is_stale,
            )

        if not _passes_new_lock_gates(new_t, center_y=center_y) or new_is_env:
            state.new_lock_candidate = None
            state.new_lock_frames = 0
            if locked is not None and state.target_lost_frames == 0:
                return (
                    DetectionResult(locked, result.candidates, locked.confidence),
                    False,
                )
            return DetectionResult(None, result.candidates, 0.0), False

        confirm_frames = int(cfg.get("new_lock_confirm_frames", NEW_LOCK_CONFIRM_FRAMES))
        confirm_frames = max(1, min(4, confirm_frames))
        if (
            state.new_lock_candidate is not None
            and _same_lock_identity(state.new_lock_candidate, new_t)
        ):
            state.new_lock_frames += 1
        else:
            state.new_lock_candidate = new_t
            state.new_lock_frames = 1

        if state.new_lock_frames >= confirm_frames:
            state.locked_target = new_t
            state.target_lost_frames = 0
            state.switch_candidate = None
            state.switch_frames = 0
            state.new_lock_candidate = None
            state.new_lock_frames = 0
            return result, False

        # Building lock — no dot until confirmed (stops ADS FP flash).
        return DetectionResult(None, result.candidates, 0.0), False

    state.target_lost_frames += 1
    state.new_lock_candidate = None
    state.new_lock_frames = 0
    state.switch_candidate = None
    state.switch_frames = 0
    if state.target_lost_frames >= lost_max:
        state.reset()
        if on_lock_expired is not None:
            on_lock_expired()
        return DetectionResult(None, result.candidates, 0.0), False

    locked = state.locked_target
    if locked is not None:
        if _env_fp(locked):
            state.reset()
            if on_lock_expired is not None:
                on_lock_expired()
            return DetectionResult(None, result.candidates, 0.0), False
        is_stale = True
        return (
            DetectionResult(locked, result.candidates, locked.confidence),
            is_stale,
        )
    return DetectionResult(None, result.candidates, 0.0), False


class TargetLockMachine:
    """Thin test/audit wrapper around :class:`TargetLockState`."""

    def __init__(self, cfg: dict[str, Any], *, center_y: float = 360.0) -> None:
        self.cfg = cfg
        self.center_y = center_y
        self.state = TargetLockState()
        self.last_instant_adopted = False
        self.last_switch_adopted = False

    @property
    def locked_target(self) -> Target | None:
        return self.state.locked_target

    @locked_target.setter
    def locked_target(self, value: Target | None) -> None:
        self.state.locked_target = value

    @property
    def target_lost_frames(self) -> int:
        return self.state.target_lost_frames

    @target_lost_frames.setter
    def target_lost_frames(self, value: int) -> None:
        self.state.target_lost_frames = value

    @property
    def locked(self) -> Target | None:
        return self.state.locked_target

    @locked.setter
    def locked(self, value: Target | None) -> None:
        self.state.locked_target = value

    @property
    def lost(self) -> int:
        return self.state.target_lost_frames

    @lost.setter
    def lost(self, value: int) -> None:
        self.state.target_lost_frames = value

    @property
    def switch_cand(self) -> Target | None:
        return self.state.switch_candidate

    @property
    def switch_candidate(self) -> Target | None:
        return self.state.switch_candidate

    @property
    def switch_frames(self) -> int:
        return self.state.switch_frames

    def reset(self) -> None:
        self.state.reset()

    def merge_detection(
        self,
        result: DetectionResult,
        *,
        on_lock_expired: Callable[[], None] | None = None,
    ) -> tuple[DetectionResult, bool]:
        prev_locked = self.state.locked_target
        prev_switch_frames = self.state.switch_frames
        self.last_instant_adopted = False
        self.last_switch_adopted = False
        effective, is_stale = apply_target_lock(
            self.state,
            result,
            center_y=self.center_y,
            cfg=self.cfg,
            on_lock_expired=on_lock_expired,
        )
        if (
            effective.target is not None
            and prev_locked is not None
            and effective.target is not prev_locked
            and self.state.target_lost_frames == 0
            and result.target is effective.target
        ):
            if prev_switch_frames >= 3:
                self.last_switch_adopted = True
            else:
                self.last_instant_adopted = True
        return effective, is_stale

    def step_target(
        self,
        raw_target: Target | None,
        *,
        candidates: int = 0,
    ) -> tuple[Target | None, bool]:
        """Convenience for unit tests that only pass a bare ``Target``."""
        if raw_target is not None:
            conf = float(getattr(raw_target, "confidence", 0.0))
            result = DetectionResult(raw_target, candidates, conf)
        else:
            result = DetectionResult(None, candidates, 0.0)
        effective, is_stale = self.merge_detection(result)
        return effective.target, is_stale

    def step(
        self,
        raw_target: Target | None,
        **_: object,
    ) -> tuple[Target | None, bool]:
        """Regression-test entry: bare ``Target`` or ``None`` per frame."""
        return self.step_target(raw_target)

    def step_detection(
        self,
        result: DetectionResult,
        *,
        on_lock_expired: Callable[[], None] | None = None,
    ) -> tuple[DetectionResult, bool]:
        """Audit/runtime entry: full ``DetectionResult``."""
        return self.merge_detection(result, on_lock_expired=on_lock_expired)
