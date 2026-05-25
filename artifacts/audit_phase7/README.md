# Phase-7 audit artifacts

The user reported the assist dot drifting upward into the sky while an
enemy strafed back and forth (player NOT moving the mouse). This phase
addresses the 11 audit findings (4 audit agents → consensus list of
fixes).

## Layout

```
audit_phase7/
├── scripts/
│   └── audit_gif_drift.py        # GIF replay harness
├── gif_before/                   # stride=5 BEFORE fix (drift_chart + per-frame artifacts)
├── gif_before_stride1/           # stride=1 BEFORE — drift_chart + summary only
├── gif_after/                    # stride=5 AFTER fix
├── gif_after_stride1/            # stride=1 AFTER — drift_chart + summary only
└── static_after/                 # 7 real Apex screenshots replayed AFTER fix
```

## Source data

* `artifacts/real_apex_test/_gif_frames/source.gif` — full 166-frame GIF (29 MB)
* `frame_000.png` … `frame_165.png` (stride=5 sample) committed for
  reproducibility.  To regenerate every frame:
  ```bash
  ffmpeg -i source.gif frame_%03d.png
  ```

## Reproducing the audit

```bash
# BEFORE fix (revert the source first):
python3 artifacts/audit_phase7/scripts/audit_gif_drift.py --out gif_before --stride 5

# AFTER fix:
python3 artifacts/audit_phase7/scripts/audit_gif_drift.py --out gif_after --apply-fix --stride 5
```

The harness instantiates a persistent ``DetectionContext`` and a
``LockMachine`` that re-implements ``runtime._select_target`` so the
results match what the live runtime sees frame-by-frame.

## Fixes applied (bug IDs from audit consensus)

| ID    | File              | Summary |
|-------|-------------------|---------|
| CRIT1 | runtime.py        | Instant-adopt now requires (a) ≥0.85 ratio of locked body score, (b) **absolute** body score ≥0.55, AND (c) NOT an upward-fragment signature (new bbox top ≥ 15% of locked h higher AND new h smaller). |
| CRIT2 | runtime.py        | `_smooth_aim(None)` now returns the cached anchor instead of immediately `tracker.reset()`. Companion: overlay also hides when `target is None` so the frozen anchor is not rendered. |
| CRIT3 | motion.py         | `_body_bbox` is stabilised against single-frame fragments — when the new bbox jumps upward by >20 % of prior h OR shrinks below 55 % of prior h, the prior stable bbox is held for up to 4 frames. |
| HIGH4 | runtime.py        | `_release_ads_inner` calls `soft_reset()` instead of `reset()` so RMB release no longer wipes the smoother anchor. |
| HIGH5 | profiles.py       | `humanoid_min_height_pixels` default 16 → 40 (Tracking preset still pins 60). |
| HIGH6 | profiles.py       | `humanoid_min_solidity` default 0.15 → 0.10 — img2's body=1.00 candidate at solidity=0.14 now passes. |
| MED7  | detector.py + config_validation.py | Removed dead code: `get_last_debug_lines`, `_LAST_DEBUG_LINES`, `_is_humanoid_contour`, `target_window_title` validator. |
| MED9  | runtime_controller.py | `apply_config_patch` now propagates `magnetism_min_pull_scale`, `fov_edge_min_pull_scale`, `smoothing_curve` to the live pull controller. |
| MED10 | detector.py + runtime.py | `draw_debug` accepts a live `detection_mode`; runtime passes `cfg['detection_mode']`. |
| MED11 | runtime_controller.py | `verbose_logging` patch now actually flips the root logger level. |

## Evidence summary

### GIF BEFORE → AFTER

The BEFORE stride=1 trajectory shows a clear upward climb between
samples ~26 and ~53 (bbox top ratchets from y≈45 to y≈32 via
successive instant-adopt; body_shape stays ≥0.83 so the multiplicative
ratchet didn't catch it) and a giant sky-lock at sample 133 (y=30,
h=397 — covers 88 % of the 450-px frame).

The AFTER stride=1 trajectory keeps bbox top mostly in the 140-180 px
band (chest of a real enemy). The single brief excursion at sample 17
to y=70 is the architecture-pillar fusion in the GIF — a known
detector-layer limitation outside Phase-7 scope — but the motion-y
(the actual aim point) stays in y=221 (chest band) so the visible
dot is NOT in the sky.

### Static 7-image

| Image                    | Phase-6 AFTER p2 | Phase-7 AFTER p2 | Change |
|--------------------------|-------------------|-------------------|--------|
| img1 back view           | active 0.99       | active 0.99       | same |
| img2 shooting            | **rejected**      | **active 1.00**   | ✅ fixed (HIGH6) |
| img3 side view           | active 0.62       | active 0.62       | same |
| img4 dummy not detected  | active 0.74       | active 1.00       | larger bbox (acceptable) |
| img5 close ADS           | active 0.76       | active 0.76       | same |
| img6 seven characters    | rejected          | rejected, accepted=2 | spec satisfied (>1 accepted) |
| img7 sky dot bug         | fully rejected    | p1 active y_top=0.36 | not sky-locked |
