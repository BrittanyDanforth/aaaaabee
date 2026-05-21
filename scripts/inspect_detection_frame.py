#!/usr/bin/env python3
"""Print mask/contour/cluster inspection for a PNG or synthetic scenario."""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import detector
from tests.synthetic_frame_generators import HSV_RED, CX, CY, FOV_RADIUS, SYNTHETIC_FRAME_GENERATORS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=list(SYNTHETIC_FRAME_GENERATORS.keys()))
    parser.add_argument("--image", help="Path to screenshot PNG/JPG")
    parser.add_argument("--fov", type=int, default=FOV_RADIUS)
    args = parser.parse_args()

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Could not read {args.image}")
            return 1
    elif args.scenario:
        frame, spec = SYNTHETIC_FRAME_GENERATORS[args.scenario]()
        print(f"scenario={spec.name} expect_active={spec.expect_active}")
        print(spec.pixel_layout)
    else:
        frame, _ = SYNTHETIC_FRAME_GENERATORS["red_building_dummy_overlap"]()
        print("default: red_building_dummy_overlap")

    h, w = frame.shape[:2]
    cx, cy = w / 2, h / 2
    report = detector.inspect_frame(frame, HSV_RED, args.fov, cx, cy)
    print(json.dumps(report, indent=2))

    r = detector.find_best_target(frame, HSV_RED, args.fov, 60.0, cx, cy, debug=True)
    print("active:", r.active)
    for line in r.debug_lines:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
