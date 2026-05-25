"""Runtime frame statistics for overlay assist loop."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class FrameStats:
    fps: float = 0.0
    frame_ms: float = 0.0
    dropped_frames: int = 0
    confidence: float = 0.0
    pull_px: float = 0.0
    pull_strength: float = 0.0
    distance_px: float = 0.0
    locked: bool = False
    has_target: bool = False
    mouse_backend: str = ""
    candidates: int = 0
    capture_w: int = 0
    capture_h: int = 0


class RuntimeStats:
    def __init__(self, configured_fps: int) -> None:
        self.configured_fps = max(1, int(configured_fps))
        self.total_frames = 0
        self.dropped_frames = 0
        self._window_start = time.perf_counter()
        self._last_frame_t = self._window_start
        self.last = FrameStats()
        self._lines: list[str] = []

    def begin_frame(self, t: float) -> None:
        self._last_frame_t = t

    def update(
        self,
        *,
        frame_ms: float,
        ads: bool,
        target_distance: float | None,
        confidence: float,
        pull_px: float,
        pull_strength: float,
        locked: bool,
        has_target: bool,
        mouse_backend: str,
        candidates: int,
        capture_size: tuple[int, int],
    ) -> None:
        self.total_frames += 1
        budget_ms = 1000.0 / self.configured_fps
        if frame_ms > budget_ms * 1.35:
            self.dropped_frames += 1
        now = time.perf_counter()
        elapsed = max(0.001, now - self._window_start)
        fps = self.total_frames / elapsed
        self.last = FrameStats(
            fps=fps,
            frame_ms=frame_ms,
            dropped_frames=self.dropped_frames,
            confidence=confidence,
            pull_px=pull_px,
            pull_strength=pull_strength,
            distance_px=target_distance or 0.0,
            locked=locked,
            has_target=has_target,
            mouse_backend=mouse_backend,
            candidates=candidates,
            capture_w=int(capture_size[0]),
            capture_h=int(capture_size[1]),
        )
        self._lines = [
            f"fps={fps:.1f} frame_ms={frame_ms:.1f} dropped={self.dropped_frames}",
            f"conf={confidence:.2f} pull={pull_px:.1f} str={pull_strength:.2f}",
            f"cands={candidates} ads={ads} lock={locked}",
        ]

    def format_lines(self) -> list[str]:
        return list(self._lines)
