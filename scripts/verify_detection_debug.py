#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import detector
from tests.test_detector import HSV_RED, _apex_dummy, _horizontal_stripe, _red_building

h, w = 720, 1280
cx, cy = w / 2, h / 2
frame = np.full((h, w, 3), 40, dtype=np.uint8)
_red_building(frame, 40, int(cy - 120), 280, 240)
_apex_dummy(frame, int(cx), int(cy + 100), scale=1.2)
r = detector.find_best_target(frame, HSV_RED, 200, 60.0, cx, cy, debug=True)
print("active:", r.active)
for line in r.debug_lines:
    print(line)
