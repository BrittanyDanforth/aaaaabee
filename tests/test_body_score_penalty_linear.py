"""D6 regression: body_shape<0.48 penalty must be a linear ramp.

The previous penalty was a flat ``+ fov_radius * 0.9`` cliff at
``body_shape_score < 0.48``. That dropped marginal real characters
(body_shape ~ 0.47) by ~200 px at FOV=220 — enough to tip them below
the confidence floor. The D6 (audit) fix replaces the cliff with a
linear ramp scaled by how far below 0.48 the score sits:

    penalty += max(0, 0.48 - body_shape) * fov_radius * 2.0

So body_shape=0.47 loses only ~4 px (0.01 * 2 * 220), body_shape=0.30
loses ~80 px, body_shape=0.00 still loses ~190 px. This keeps the
gate's protective effect on truly bad candidates while letting
marginal real bodies through.

We exercise ``score_target`` directly with two synthetic Target objects
that differ only in body_shape_score and assert the score delta is
roughly linear (NOT a step function) within the 0..0.48 range.
"""

from __future__ import annotations

import unittest

import detector


def _make_target(body_shape: float) -> detector.Target:
    return detector.Target(
        centroid_x=640.0,
        centroid_y=360.0,
        area=10000.0,
        distance_to_center=10.0,
        confidence=0.0,
        bbox_x=590, bbox_y=275, bbox_w=100, bbox_h=200,
        solidity=0.6,
        humanoid_score=body_shape,
        part_count=3,
        body_shape_score=body_shape,
        head_score=0.4,
        torso_score=0.5,
        limb_stack_score=0.4,
        red_coverage=0.0,
        fill_ratio=0.5,
        max_circularity=0.30,
    )


class BodyScorePenaltyLinearTests(unittest.TestCase):
    def test_no_cliff_at_0_48(self) -> None:
        """The flat-cliff bug was a >100 px score drop right at 0.48.
        The linear ramp keeps the drop bounded to ~4 px at body=0.47."""
        s_47 = detector.score_target(_make_target(0.47), 220.0, 2.0, 0.015, center_y=360.0)
        s_48 = detector.score_target(_make_target(0.48), 220.0, 2.0, 0.015, center_y=360.0)
        delta = s_48 - s_47
        # body_term alone changes by 0.01 * 220 * 1.15 = 2.53 px.
        # Linear penalty adds another 0.01 * 220 * 2.0 = 4.4 px to the gap.
        # Total ~6.9 px. Pin upper bound at 15 px so a future cliff drift
        # (which would push delta toward ~200 px) is caught.
        self.assertLess(
            delta, 15.0,
            f"score delta across body=0.48 boundary must be small under "
            f"linear ramp; got delta={delta:.1f}px",
        )

    def test_linear_ramp_monotonic(self) -> None:
        scores = [
            detector.score_target(_make_target(b), 220.0, 2.0, 0.015, center_y=360.0)
            for b in (0.10, 0.20, 0.30, 0.40, 0.48, 0.55, 0.65)
        ]
        # Strictly monotonic increasing.
        for a, b in zip(scores, scores[1:]):
            self.assertLess(a, b, f"score must be monotonic in body_shape; {scores}")

    def test_penalty_collapses_to_zero_above_0_48(self) -> None:
        """At body_shape >= 0.48, no body-shape penalty applies — the
        score should match the term-sum without the ramp."""
        s_50 = detector.score_target(_make_target(0.50), 220.0, 2.0, 0.015, center_y=360.0)
        s_60 = detector.score_target(_make_target(0.60), 220.0, 2.0, 0.015, center_y=360.0)
        # Delta should be exactly body_term * 0.10 (no penalty term).
        delta = s_60 - s_50
        body_term_delta = 0.10 * 220.0 * 1.15  # 25.3
        self.assertAlmostEqual(
            delta, body_term_delta, delta=3.0,
            msg=(
                f"above 0.48 the body-shape penalty must be zero so delta "
                f"matches body_term contribution (expected ~{body_term_delta:.1f}px, "
                f"got {delta:.1f})"
            ),
        )


if __name__ == "__main__":
    unittest.main()
