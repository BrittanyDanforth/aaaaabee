"""Self-check detection contract: body dummy passes, legacy blob does not."""

from __future__ import annotations

import unittest

import detector
from self_check import (
    build_legacy_blob_frame,
    build_selfcheck_body_frame,
    run_blob_mask_sanity,
    run_body_detection_check,
)

HSV_RED = [
    {"lower": [0, 100, 100], "upper": [12, 255, 255]},
    {"lower": [170, 100, 100], "upper": [180, 255, 255]},
]

CONFIG = {
    "hsv_ranges": HSV_RED,
    "fov_radius_pixels": 180,
    "min_target_area": 40,
}


class SelfCheckDetectionTests(unittest.TestCase):
    def test_body_dummy_detected_on_400_frame(self) -> None:
        ok, lines, err = run_body_detection_check(detector.find_best_target, CONFIG)
        self.assertTrue(ok, err or "\n".join(lines))
        self.assertTrue(any("OK" in ln for ln in lines))

    def test_legacy_blob_not_used_as_pass_gate(self) -> None:
        frame = build_selfcheck_body_frame()
        r_body = detector.find_best_target(
            frame, HSV_RED, 180, 40.0, 200.0, 200.0, debug=True
        )
        self.assertTrue(r_body.active, r_body.debug_lines)

        blob = build_legacy_blob_frame()
        r_blob = detector.find_best_target(
            blob, HSV_RED, 180, 40.0, 200.0, 200.0, debug=True
        )
        self.assertFalse(r_blob.active)
        self.assertIsNone(r_blob.target)

    def test_blob_sanity_lines(self) -> None:
        lines = run_blob_mask_sanity(detector.find_best_target, CONFIG)
        self.assertEqual(1, len(lines))
        self.assertIn("legacy red blob", lines[0])


if __name__ == "__main__":
    unittest.main()
