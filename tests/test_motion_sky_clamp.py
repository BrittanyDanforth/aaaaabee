"""Overlay must not climb above body chest band when body bbox is active."""

from __future__ import annotations

import motion as motion_mod


def test_overlay_follow_disables_upward_velocity_lead_with_body_bbox() -> None:
    tr = motion_mod.TargetTracker()
    bx, by, bw, bh = 400, 200, 80, 160
    m0 = tr.observe_target(
        440.0, 280.0, 0.0,
        bbox_x=bx, bbox_y=by, bbox_w=bw, bbox_h=bh,
        aim_is_body_anchor=True,
    )
    tr._vx = 0.0
    tr._vy = -200.0
    m1 = tr.observe_target(
        442.0, 278.0, 1.0 / 30.0,
        bbox_x=bx, bbox_y=by, bbox_w=bw, bbox_h=bh,
        aim_is_body_anchor=True,
    )
    y_hi = by + bh * motion_mod._body_y_hi_frac
    _, oy = m1.overlay_xy()
    assert oy <= y_hi + 2.0, f"overlay climbed above chest band: {oy} > {y_hi}"
    assert m1.y <= y_hi + 2.0


def test_point_inside_body_bbox_chest_band() -> None:
    bx, by, bw, bh = 100, 100, 50, 100
    assert motion_mod.TargetTracker.point_inside_body_bbox(125, 140, bx, by, bw, bh)
    assert not motion_mod.TargetTracker.point_inside_body_bbox(125, 90, bx, by, bw, bh)
