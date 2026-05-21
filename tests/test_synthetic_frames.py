"""Realistic synthetic frame scenarios — not trivial solid rectangles."""

from __future__ import annotations

import unittest

import detector
from tests.synthetic_frame_generators import (
    HSV_RED,
    MIN_AREA,
    FOV_RADIUS,
    CX,
    CY,
    SYNTHETIC_FRAME_GENERATORS,
)


class SyntheticFrameTests(unittest.TestCase):
    def _run(self, name: str) -> detector.DetectionResult:
        gen = SYNTHETIC_FRAME_GENERATORS[name]
        frame, spec = gen()
        return detector.find_best_target(
            frame,
            HSV_RED,
            FOV_RADIUS,
            MIN_AREA,
            CX,
            CY,
            debug=True,
        )

    def test_all_registered_generators(self) -> None:
        for name in SYNTHETIC_FRAME_GENERATORS:
            with self.subTest(name=name):
                frame, spec = SYNTHETIC_FRAME_GENERATORS[name]()
                r = detector.find_best_target(
                    frame, HSV_RED, FOV_RADIUS, MIN_AREA, CX, CY, debug=True
                )
                self.assertEqual(
                    r.active,
                    spec.expect_active,
                    f"{name}: {spec.pass_fail_rationale}\n" + "\n".join(r.debug_lines[-6:]),
                )

    def test_merged_blob_inactive(self) -> None:
        r = self._run("single_merged_blob_from_blur")
        self.assertFalse(r.active)
        self.assertTrue(
            any("solid" in ln or "no_body" in ln or "body=0" in ln for ln in r.debug_lines),
            r.debug_lines,
        )

    def test_inspect_frame_reports_clusters(self) -> None:
        frame, _ = SYNTHETIC_FRAME_GENERATORS["red_building_dummy_overlap"]()
        report = detector.inspect_frame(frame, HSV_RED, FOV_RADIUS, CX, CY)
        self.assertGreater(report["parts"], 0)
        self.assertGreater(len(report["clusters_detail"]), 0)


if __name__ == "__main__":
    unittest.main()
