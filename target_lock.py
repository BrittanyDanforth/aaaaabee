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

from detector import DetectionResult, Target, _bbox_iou, target_is_background_clutter

# IoU gates (documented in tests/test_detection_hardening.py).
INSTANT_ADOPT_MIN_IOU = 0.35
HIGH_OVERLAP_REFINE_IOU = 0.45
SWITCH_MIN_IOU = 0.28
CLUTTER_REJECT_MAX_IOU = 0.18


def viewmodel_exclude_bottom(cfg: dict[str, Any]) -> float:
    """Bottom-of-frame mask fraction — must match production runtime."""
    return float(cfg.get("viewmodel_exclude_bottom_frac", 0.28))


@dataclass
class TargetLockState:
    locked_target: Target | None = None
    target_lost_frames: int = 0
    switch_candidate: Target | None = None
    switch_frames: int = 0

    def reset(self) -> None:
        self.locked_target = None
        self.target_lost_frames = 0
        self.switch_candidate = None
        self.switch_frames = 0


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
) -> tuple[DetectionResult, bool]:
    """Merge a detector result with sticky lock state.

    Returns ``(effective_result, is_stale)``. ``is_stale`` is True when the
    effective target is the frozen lock during grace (``target_lost_frames > 0``)
    without a fresh detection on that frame.
    """
    lost_max = int(cfg["target_lost_frames_before_unlock"])
    is_stale = False

    if result.target is not None:
        new_t = result.target
        locked = state.locked_target
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
            aim_jump_up = new_t.centroid_y < locked.centroid_y - 16.0
            weak_red_adopt = aim_jump_up and new_t.red_coverage < 0.06
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
            sky_band = new_t.bbox_y + new_t.bbox_h * 0.5 < center_y * 0.40
            clutter_fp = target_is_background_clutter(new_t)
            instant_adopt_ok = (
                bs_ratio_ok
                and bs_abs_ok
                and not upward_fragment
                and not weak_red_adopt
                and not clutter_fp
                and adopt_iou >= INSTANT_ADOPT_MIN_IOU
                and not sky_band
            )
            high_overlap_refine = adopt_iou >= HIGH_OVERLAP_REFINE_IOU and instant_adopt_ok
            if instant_adopt_ok and (
                high_overlap_refine or (drift < 25 and drift < fov_lim)
            ):
                state.locked_target = new_t
                state.target_lost_frames = 0
                state.switch_candidate = None
                state.switch_frames = 0
                return result, False

            if state.target_lost_frames == 0:
                state.switch_candidate = None
                state.switch_frames = 0
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
                and not target_is_background_clutter(new_t)
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

        new_sky = new_t.bbox_y + new_t.bbox_h * 0.5 < center_y * 0.40
        low_red_lock = new_t.red_coverage < 0.04
        if (
            new_t.body_shape_score < 0.55
            or new_sky
            or low_red_lock
            or target_is_background_clutter(new_t)
        ):
            return DetectionResult(None, result.candidates, 0.0), False

        state.locked_target = new_t
        state.target_lost_frames = 0
        state.switch_candidate = None
        state.switch_frames = 0
        return result, False

    state.target_lost_frames += 1
    state.switch_candidate = None
    state.switch_frames = 0
    if state.target_lost_frames >= lost_max:
        state.reset()
        if on_lock_expired is not None:
            on_lock_expired()
        return DetectionResult(None, result.candidates, 0.0), False

    locked = state.locked_target
    if locked is not None:
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
