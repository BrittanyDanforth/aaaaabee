#!/usr/bin/env python3
"""Extract every frame from source.gif into _gif_frames_all/frame_NNNN.png."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[1]
GIF = REPO / "artifacts" / "real_apex_test" / "_gif_frames" / "source.gif"
OUT = REPO / "artifacts" / "real_apex_test" / "_gif_frames_all"


def main() -> int:
    if not GIF.exists():
        print(f"Missing {GIF}", file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(GIF))
    if not cap.isOpened():
        print("Could not open GIF", file=sys.stderr)
        return 1
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        path = OUT / f"frame_{n:04d}.png"
        cv2.imwrite(str(path), frame)
        n += 1
    cap.release()
    print(f"Wrote {n} frames to {OUT}")
    return 0 if n > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
