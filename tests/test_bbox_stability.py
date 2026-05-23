"""Phase-7 CRIT3 regression test — _body_bbox single-frame fragment stability.

Feed ``TargetTracker.observe_target`` the sequence:

  frame 1: (100, 200, 60, 140)        — torso bbox (real body)
  frame 2: (100, 150, 60,  40)        — head fragment (helmet only)
  frame 3: (100, 200, 60, 140)        — torso restored

WITHOUT the CRIT3 stabilization the chest-band clamp follows the
fragment in frame 2 and the smoothed Y jumps up by ~30 px in a single
frame, which is exactly the visible drift the user reported.

The Phase-7 fix keeps the prior stable bbox for ~4 frames when the
new bbox jumps significantly upward OR shrinks dramatically.  After
the bbox stabilises again the tracker accepts the new baseline.
"""

from __future__ import annotations

import unittest

from motion import TargetTracker


class BboxStabilityTests(unittest.TestCase):
    def test_single_frame_head_fragment_does_not_jump_clamp(self) -> None:
        tracker = TargetTracker()
        bx, bw = 100, 60
        # FRAME 1 — torso bbox; aim point at body center.
        tracker.observe_target(
            x=bx + bw / 2.0, y=200 + 140 / 2.0, time_sec=0.000,
            bbox_x=bx, bbox_y=200, bbox_w=bw, bbox_h=140,
            aim_is_body_anchor=True,
        )
        # FRAME 2 — head-only fragment: bbox jumps up by 50 px and shrinks to 40 px.
        m2 = tracker.observe_target(
            x=bx + bw / 2.0, y=150 + 40 / 2.0, time_sec=0.033,
            bbox_x=bx, bbox_y=150, bbox_w=bw, bbox_h=40,
            aim_is_body_anchor=True,
        )
        # FRAME 3 — torso restored.
        m3 = tracker.observe_target(
            x=bx + bw / 2.0, y=200 + 140 / 2.0, time_sec=0.066,
            bbox_x=bx, bbox_y=200, bbox_w=bw, bbox_h=140,
            aim_is_body_anchor=True,
        )

        # PRIMARY assertion (CRIT3): _body_bbox on frame 2 should be the
        # PRIOR stable bbox, not the head fragment — the fragment is
        # detected (upward jump AND shrink) and held for a few frames.
        self.assertEqual(
            tracker._body_bbox,
            (100, 200, 60, 140),
            "CRIT3 regression: head fragment overwrote _body_bbox in one frame",
        )
        # The smoothed Y on frame 2 should land inside the PRIOR chest
        # band [200+140*0.28, 200+140*0.50] = [239.2, 270] — NOT in the
        # head-fragment band [161.2, 170] (which is where the pre-fix
        # tracker would jump to).
        self.assertGreaterEqual(m2.y, 230.0, "frame 2 smoothed Y jumped into fragment band")
        # Frame 3 also lands inside the chest band of the restored bbox.
        self.assertGreaterEqual(m3.y, 230.0)

    def test_dramatic_shrink_holds_prior_bbox(self) -> None:
        tracker = TargetTracker()
        # FRAME 1 — full body (h=200).
        tracker.observe_target(
            x=200, y=200, time_sec=0.000,
            bbox_x=180, bbox_y=100, bbox_w=40, bbox_h=200,
            aim_is_body_anchor=True,
        )
        # FRAME 2 — bbox shrinks to 40 px (20 % of prior).
        tracker.observe_target(
            x=200, y=200, time_sec=0.033,
            bbox_x=180, bbox_y=100, bbox_w=40, bbox_h=40,
            aim_is_body_anchor=True,
        )
        # The clamp should still reference the prior 200-px bbox.
        # _body_bbox is updated with the stabilised value.
        self.assertEqual(tracker._body_bbox[3], 200,
                         "CRIT3 regression: shrink not held — _body_bbox.h was overwritten")

    def test_persistent_fragment_eventually_accepted(self) -> None:
        """After _STABLE_BBOX_HOLD_MAX (4) consecutive fragment frames
        the tracker accepts the smaller bbox as the new baseline so a
        legitimate crouch/distance change isn't held in stale state
        forever."""
        tracker = TargetTracker()
        tracker.observe_target(
            x=200, y=200, time_sec=0.0,
            bbox_x=180, bbox_y=100, bbox_w=40, bbox_h=200,
            aim_is_body_anchor=True,
        )
        # Feed 5 consecutive shrunk frames.
        for i in range(5):
            tracker.observe_target(
                x=200, y=200, time_sec=0.033 * (i + 1),
                bbox_x=180, bbox_y=100, bbox_w=40, bbox_h=40,
                aim_is_body_anchor=True,
            )
        # After 5 frames the new baseline should be accepted (h=40).
        self.assertEqual(tracker._body_bbox[3], 40,
                         "CRIT3 regression: shrunk bbox never accepted as new baseline")


if __name__ == "__main__":
    unittest.main()
