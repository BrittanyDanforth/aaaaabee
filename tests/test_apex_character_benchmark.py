"""Bug-3 regression: ``apex`` mode must NOT regress detection on non-red
characters. The fused mask (shape edges + chroma spread + motion diff +
red enemy) must pick up humanoid silhouettes regardless of base colour.

A simple seven-character benchmark covers the palette range the user
flagged: red helmet/armour (Crypto), red robot (Revenant), yellow gas
trooper (Caustic), white/teal scientist (Horizon), red/orange medic
(Lifeline), green stim runner (Octane), and khaki/military (Bangalore).

Frames are synthesised with a humanoid silhouette (head + chest + legs)
drawn in the character's base colours against a dim, slightly textured
background. A second motion frame is fed first so the
:class:`detector.DetectionContext` populates its inter-frame motion mask
before the active detection frame is queried — this matches the live
runtime path where each frame becomes ``prev_gray`` for the next.

We assert ≥ 6/7 characters detect (``active=True``); the bar is loose
enough to leave one slot for edge cases (e.g. a low-saturation skin
that fully blends with the desaturated terrain) while still failing
loudly if the apex fusion is broken.
"""

from __future__ import annotations

import unittest

import cv2
import numpy as np

import detector

H, W = 720, 1280
CX, CY = W // 2, H // 2
FOV = 220
MIN_AREA = 40.0


def _draw_character(
    frame: np.ndarray,
    cx: int,
    cy: int,
    *,
    base: tuple[int, int, int],
    accent: tuple[int, int, int],
    helmet: tuple[int, int, int] | None = None,
) -> None:
    """Render a humanoid silhouette with head (helmet), torso (base),
    arms (accent), and legs (base)."""
    # Narrow humanoid (~ 50 wide, 180 tall) to mimic Apex enemy proportions
    # at typical engagement distance. Wider models trigger the solid_wall
    # reject when motion-fused mask saturates the bbox.
    h_color = helmet if helmet is not None else base
    cv2.ellipse(frame, (cx, cy - 60), (14, 18), 0, 0, 360, h_color, -1)
    cv2.rectangle(frame, (cx - 22, cy - 35), (cx + 22, cy + 35), base, -1)
    cv2.rectangle(frame, (cx - 32, cy - 25), (cx - 22, cy + 20), accent, -1)
    cv2.rectangle(frame, (cx + 22, cy - 25), (cx + 32, cy + 20), accent, -1)
    cv2.rectangle(frame, (cx - 16, cy + 35), (cx - 2, cy + 110), base, -1)
    cv2.rectangle(frame, (cx + 2, cy + 35), (cx + 16, cy + 110), base, -1)


# (name, helmet_bgr, base_bgr, accent_bgr)
# Colours approximate the in-game palettes the user listed; values are
# chosen to be vivid enough to fire chroma_spread + shape edges without
# matching any non-character HSV gate (no near-grey skins).
CHARACTERS: list[tuple[str, tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]] = [
    ("crypto_red", (40, 40, 220), (40, 40, 220), (40, 40, 200)),
    ("revenant_red_robot", (60, 60, 180), (50, 50, 160), (40, 40, 130)),
    ("caustic_yellow", (40, 180, 200), (40, 170, 200), (30, 130, 160)),
    ("horizon_teal_white", (200, 200, 200), (180, 180, 80), (200, 200, 200)),
    ("lifeline_red_orange", (40, 80, 220), (50, 100, 240), (60, 140, 240)),
    ("octane_green", (60, 200, 80), (40, 180, 60), (40, 160, 60)),
    ("bangalore_khaki", (60, 110, 130), (60, 100, 120), (50, 90, 110)),
]


def _make_motion_pair(
    name: str,
    helmet: tuple[int, int, int],
    base: tuple[int, int, int],
    accent: tuple[int, int, int],
    *,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Two frames: the same character drawn at a shifted position so the
    inter-frame motion-diff mask fires on the silhouette. Background is
    a deterministic low-frequency textured dim terrain — fixed seed so
    the benchmark hit rate does not vary between Python invocations
    (PYTHONHASHSEED randomisation)."""
    rng = np.random.default_rng(seed)
    bg = (rng.integers(35, 75, (H, W, 3))).astype(np.uint8)
    prev = bg.copy()
    curr = bg.copy()
    _draw_character(prev, CX - 25, CY, base=base, accent=accent, helmet=helmet)
    _draw_character(curr, CX, CY, base=base, accent=accent, helmet=helmet)
    return prev, curr


class ApexFusedDetectionBenchmarkTests(unittest.TestCase):
    def test_six_of_seven_characters_detect_with_motion_context(self) -> None:
        ctx = detector.DetectionContext(motion_assist=True)
        results: list[tuple[str, bool, str]] = []
        for name, helmet, base, accent in CHARACTERS:
            ctx.reset()
            prev, curr = _make_motion_pair(name, helmet, base, accent)
            # First frame primes prev_gray inside the context.
            detector.find_best_target(
                prev, [], FOV, MIN_AREA, float(CX), float(CY),
                debug=False, detection_mode="apex", context=ctx,
            )
            # Active frame: now the motion-diff mask fires on the
            # character silhouette.
            r = detector.find_best_target(
                curr, [], FOV, MIN_AREA, float(CX), float(CY),
                debug=True, detection_mode="apex", context=ctx,
            )
            tail = r.debug_lines[-1] if r.debug_lines else "(no debug)"
            results.append((name, r.active, tail))

        hits = sum(1 for _, ok, _ in results if ok)
        summary = "\n".join(f"  {n}: {'OK' if ok else 'MISS'} :: {tail}" for n, ok, tail in results)
        self.assertGreaterEqual(
            hits, 6,
            f"apex mode detected only {hits}/7 character palettes:\n{summary}",
        )

    def test_apex_mode_at_least_matches_shape_only_on_red_character(self) -> None:
        """Sanity: on the red character, the apex fusion must do at least
        as well as plain shape mode (it should be strictly better)."""
        helmet, base, accent = (40, 40, 220), (40, 40, 220), (40, 40, 200)
        frame = np.full((H, W, 3), 30, dtype=np.uint8)
        _draw_character(frame, CX, CY, base=base, accent=accent, helmet=helmet)

        r_shape = detector.find_best_target(
            frame, [], FOV, MIN_AREA, float(CX), float(CY),
            debug=False, detection_mode="shape",
        )
        r_apex = detector.find_best_target(
            frame, [], FOV, MIN_AREA, float(CX), float(CY),
            debug=False, detection_mode="apex",
        )
        # Apex mode must detect what shape mode detects, plus the red
        # character that shape alone misses (covered separately).
        if r_shape.active:
            self.assertTrue(r_apex.active)


if __name__ == "__main__":
    unittest.main()
