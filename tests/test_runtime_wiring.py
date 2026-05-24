"""Prove runtime always wires bbox into observe_target."""

from __future__ import annotations

import time
import unittest
from unittest.mock import patch

import numpy as np

from targeting_runtime import TargetingRuntime
from tests.reference_body_frames import CX, CY, FRAME_H, FRAME_W, gen_firing_range_standing

CONFIG = {
    "new_lock_confirm_frames": 1,
    "detection_mode": "apex",
    "hsv_ranges": [
        {"lower": [0, 100, 100], "upper": [12, 255, 255]},
        {"lower": [170, 100, 100], "upper": [180, 255, 255]},
    ],
    "fov_radius_pixels": 200,
    "min_target_area": 40.0,
    "fov_center_x": float(CX),
    "fov_center_y": float(CY),
}


class RuntimeWiringTests(unittest.TestCase):
    def test_observe_target_receives_bbox(self) -> None:
        runtime = TargetingRuntime()
        frame = gen_firing_range_standing()
        bbox_calls: list[tuple] = []

        real_observe = runtime.tracker.observe_target

        def spy(x, y, t, **kwargs):
            if kwargs.get("bbox_w"):
                bbox_calls.append(
                    (kwargs["bbox_x"], kwargs["bbox_y"], kwargs["bbox_w"], kwargs["bbox_h"])
                )
            return real_observe(x, y, t, **kwargs)

        with patch.object(runtime.tracker, "observe_target", side_effect=spy):
            aim = runtime.process_frame(frame, CONFIG, time_sec=0.0)

        self.assertTrue(aim.active)
        self.assertTrue(aim.wired_observe_target)
        self.assertEqual(1, len(bbox_calls))
        self.assertIsNotNone(aim.bbox_used)
        self.assertEqual(aim.bbox_used, bbox_calls[0])
        bw, bh = bbox_calls[0][2], bbox_calls[0][3]
        self.assertGreater(bh, bw * 1.2, "body column should be taller than wide")

    def test_balloon_frame_inactive_no_bbox_call(self) -> None:
        runtime = TargetingRuntime()
        frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8) + 25
        import cv2

        cv2.circle(frame, (CX, int(FRAME_H * 0.14)), 36, (0, 0, 255), -1)
        aim = runtime.process_frame(frame, CONFIG, time_sec=0.0)
        self.assertFalse(aim.active)
        self.assertIsNone(runtime.last_observe_bbox)
        self.assertEqual(0, runtime.observe_target_call_count)


class E2EProofTests(unittest.TestCase):
    def test_eight_proofs(self) -> None:
        import cv2
        import detector
        from motion import TargetTracker
        from tests.reference_body_frames import gen_ads_dummy_with_sky_balloon

        proofs: dict[str, bool] = {}

        # 1 wiring
        rt = TargetingRuntime()
        frame = gen_firing_range_standing()
        a = rt.process_frame(frame, CONFIG, time_sec=0.0)
        proofs["1_runtime_bbox_wired"] = a.active and rt.last_observe_bbox is not None

        # 2 four refs
        for name, gen in [
            ("standing", gen_firing_range_standing),
            ("close", __import__("tests.reference_body_frames", fromlist=["gen_close_vertical_dummy"]).gen_close_vertical_dummy),
        ]:
            r = detector.find_best_target(
                gen(), CONFIG["hsv_ranges"], 200, 40.0, CX, CY, debug=True
            )
            proofs[f"2_{name}_body"] = (
                r.active
                and r.target is not None
                and r.target.torso_score >= 0.2
                and r.target.part_count >= 2
            )

        # 3 balloon only
        f = np.zeros((FRAME_H, FRAME_W, 3), np.uint8) + 25
        cv2.circle(f, (CX, 90), 34, (0, 0, 255), -1)
        proofs["3_balloon_inactive"] = not detector.find_best_target(
            f, CONFIG["hsv_ranges"], 200, 40.0, float(CX), float(CY)
        ).active

        # 4 body + balloon
        r4 = detector.find_best_target(
            gen_ads_dummy_with_sky_balloon()[0],
            CONFIG["hsv_ranges"],
            200,
            40.0,
            float(CX),
            float(CY),
            detection_mode="hybrid",
        )
        proofs["4_body_beats_balloon"] = r4.active and r4.target.centroid_y > CY

        # 5 moving lag (same 3-plate dummy as test_balloon_body_motion)
        tr = TargetTracker()
        raw_x, sm_x = [], []
        sticky = None
        t0 = time.perf_counter()
        for i in range(20):
            fr = np.full((FRAME_H, FRAME_W, 3), 90, dtype=np.uint8)
            ox = (i - 10) * 14
            foot = CY + 120
            for dx, dy, rw, rh in [(0, -90, 20, 18), (-6, -50, 34, 26), (0, -18, 18, 16)]:
                x, y = CX + dx + ox, foot + dy
                cv2.rectangle(fr, (x - rw // 2, y - rh), (x + rw // 2, y), (0, 0, 255), -1)
            det = detector.find_best_target(
                fr, CONFIG["hsv_ranges"], 200, 40.0, float(CX), float(CY),
                sticky_target=sticky, stickiness_pixels=90,
            )
            if det.active and det.target:
                sticky = det.target
                t = det.target
                m = tr.observe_target(
                    t.centroid_x, t.centroid_y, t0 + i / 60.0,
                    bbox_x=t.bbox_x, bbox_y=t.bbox_y, bbox_w=t.bbox_w, bbox_h=t.bbox_h,
                )
                raw_x.append(t.centroid_x)
                sm_x.append(m.x)
        errs = [abs(sm_x[i] - raw_x[i]) for i in range(len(sm_x))] if sm_x else [999]
        proofs["5_low_lag"] = len(sm_x) >= 12 and max(errs) < 55.0

        # 6 self-check body dummy
        from self_check import run_body_detection_check

        ok, _, _ = run_body_detection_check(detector.find_best_target, CONFIG)
        proofs["6_selfcheck_body"] = ok

        # 7 debug lines
        r7 = detector.find_best_target(
            gen_firing_range_standing(),
            CONFIG["hsv_ranges"],
            200,
            40.0,
            float(CX),
            float(CY),
            debug=True,
        )
        joined = "\n".join(r7.debug_lines)
        proofs["7_debug_candidates"] = "cand[" in joined and "reject=" in joined

        failed = [k for k, v in proofs.items() if not v]
        self.assertEqual([], failed, f"proof failures: {failed}")


if __name__ == "__main__":
    unittest.main()
