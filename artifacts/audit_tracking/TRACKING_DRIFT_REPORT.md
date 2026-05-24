# Tracking drift / sky dot — root cause and fix report

## Symptom (user-reported)

- Red dot lags behind a moving character, then climbs into the air or pulls aim off-body.
- Camera continues turning after the body target is weak or wrong.

## Exact failure chain (code audit)

### Primary: motion overlay upward velocity lead (not detector-only)

**Stage:** `motion.TargetTracker._advance_overlay_follow()` after a valid body lock.

When `speed > 25 px/s`, overlay follow added:

```python
fy += self._vy * lead * 0.38   # full vertical lead
```

Pull anchor already disables prediction when `aim_is_body_anchor` and `_body_bbox` are set, but **overlay used a separate path** with upward velocity lead. Mask/detector jitter with negative `inst_vy` (upward) made the **visible dot** creep above the chest band even while pull was partially clamped.

**Fix:** With active body bbox: no upward velocity lead; horizontal lead reduced; hard cap `overlay_y <= bbox_y + bh * 0.50` after follow step.

### Secondary: detector sticky pool adopting head/sky fragments

**Stage:** `detector.find_best_target()` sticky pool when `currently_locked=True`.

On a weak frame, a **smaller, higher bbox** (head outline / sky tile) could sit inside stickiness distance and win `sticky_best`, shrinking `bbox_h` and raising `bbox_y`. Motion then clamped to that fragment’s chest band → dot jumps up.

**Fix:** `is_upward_fragment_vs_locked()` filters candidates from the sticky pool when locked.

### Tertiary: stale lock holding sky-shifted centroid

**Stage:** `target_lock.apply_target_lock()` grace return.

During `target_lost_frames > 0`, frozen lock could still have centroid drifted >28px above `_lock_anchor_cy` from earlier bad refine.

**Fix:** Purge lock (return no target) if stale locked centroid is above anchor beyond `MAX_LOCK_UPWARD_DRIFT_PX` or mid-bbox is in sky band.

### Not the main bug (verified wiring)

| Stage | Behavior |
|--------|----------|
| `runtime._smooth_aim(stale=True)` | Does **not** call `observe_target` — holds `_last_motion` (M1). |
| Pull on stale | Allowed only within `mouse_gate_stale_grace_frames`; pull velocity decays after grace. |
| Stable bbox hold | Can preserve **good** bbox against 1-frame fragments; wrong if fragment adopted as new stable — mitigated by detector filter. |
| Deadband | Can lag dot on micro-movement but does not add upward lead; overlay has separate follow. |
| `enumerate` vs `find_best_target` prune | `_prune_lineup_candidates` is enumerate-only (7-box audit); runtime single-lock uses `find_best_target` + lock — intentional. |

## Before / after (real GIF frames)

Harness: `python3 scripts/audit_tracking_chain.py --out after --stride 2`

Data: `artifacts/real_apex_test/_gif_frames/frame_*.png` (34 frames in workspace; 17 sampled at stride 2)

| Metric | After fix |
|--------|-----------|
| `sky_violations` | **0** |
| `overlay_inside_body` failures | **0** |
| Frames traced | 17 |

Per-frame logs: `artifacts/audit_tracking/after/frame_XXX/trace.log`

Fields logged: `raw_bbox`, `raw_anchor`, `body_shape_score`, `red_coverage`, `torso_score`, `stiff/limb`, `stale`, `target_lost_frames`, `motion_output`, `overlay_xy`, `motion_trace` (deadband, stable_bbox, body_bbox).

## Tests (real images + GIF)

- `tests/test_gif_drift_regression.py` — 10 GIF indices, no sky drift, no upward fragment switch, **overlay inside body bbox**
- `tests/test_motion_sky_clamp.py` — synthetic body bbox, upward `vy` cannot push overlay above chest
- `tests/test_detector_fragment_sticky.py` — fragment rejection helper
- `tests/test_stale_lock_holds_motion.py` — stale does not feed smoother
- `tests/test_lineup_detector_wiring.py` — lineup refine path

**391** pytest pass (headless).

## Pull trace (live)

When `trace_pull=True`, `runtime._emit_pull_trace` now includes:

`body_shape_score`, `red_coverage`, `reject_reason`, `candidate_id`, `last_stable_bbox`, `motion_body_bbox`, `in_deadband`, `meas_drift`, `inst_speed`, `overlay_inside_body`, `pull_inside_body`, `overlay_pull_delta_px`, `motion_output`.

## Expected behavior after fix

1. Fresh detection on body → dot stays in upper-chest band.
2. One-frame head fragment → sticky pool ignores it; stable bbox hold keeps prior column.
3. Stale grace → frozen motion, no new upward observe; lock purged if centroid floats into sky.
4. Moving target → overlay follows horizontally; **no upward velocity lead** on body lock.
5. Lost target past grace → pull resets; no chase into air.
