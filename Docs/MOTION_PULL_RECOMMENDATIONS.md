# Pull / Motion / Detection — Live-Tuning Recommendations

This document captures the **live-game tuning guidance** that came out
of the audit in PR #23.  All findings are measured (see PR body), all
defaults below are already shipped in the profile files
(``profiles.py``).  Use this as a checklist when the assist needs
adjustment on a specific machine, monitor, or playstyle.

---

## 1. Profile selection

| Profile | capture_fps | pull_subtick_hz | When to use |
|---------|------------:|----------------:|-------------|
| `apex_style_live_trace` | 60 | 180 | Default, recommended for most users |
| `apex_style_live_safe`  | 30 | 120 | Lower-end hardware (capture takes >5 ms) |
| `owned_dev_live`        | 90 | 240 | High-refresh / high-DPI monitor + good CPU |

The sub-tick rate is **already set per profile**.  Don't change unless
your mouse driver is coalescing moves (see §3).

---

## 2. If pull still feels laggy

### 2.1  Cursor lags strafing targets

Try in order (lower → higher impact):

1. **`smoothing_tau_moving`** default 0.014 → **0.012**.  Tighter
   motion smoother on moving targets.  Risk: more responsive to
   detector noise (mitigated by the bbox EMA we added).
2. **`prediction_lead_seconds`** default 0.020 → **0.024-0.030**.
   More velocity lead.  Capped internally by chest-band clamp.
3. **`max_pull_speed_pixels_per_frame`** default 22 → **26-30** for
   LIVE_TRACE.  Lets the pull catch up faster after a transient.
   Risk: snappier feel can read as "flick" on small corrections.

### 2.2  Cursor wobbles on stationary targets

Verify in this order:

1. `deadzone_pixels` is 2 (default).  If you raise it to 3, micro-
   noise stays in the deadzone but small drift accumulates.
2. Detector centroid jitter > 2 px → check the detect frame quality
   (capture resolution, anti-aliasing, monitor refresh).
3. If the body is moving < 1 px/frame the bbox EMA is in heavy
   smoothing mode — this is intentional.  Don't try to override.

### 2.3  Cursor goes upward into the sky / "tries to track then drifts"

This is **case L + H + A** from the audit (asymmetric vy clip,
sub-tick stale velocity decay, overlay-pull divergence cap).  All
three are in place.

If it still happens:
1. Confirm `_MAX_UPWARD_LEAD_PX = 4.0` in motion.py — drop to 2.0
   for extra safety on detector noise.
2. Confirm `OVERLAY_PULL_UPWARD_DIV_PX = 3.0` in motion.py — drop
   to 2.0 if you want even tighter dot-vs-pull lock.
3. Confirm `_SUBTICK_STALE_VELOCITY_DECAY = 0.92` in runtime.py —
   lower to 0.85 to make stale-grace extrapolation decay faster.

---

## 3. Mouse output coalescing (Windows-specific)

On Windows, the default mouse driver polls at 125 Hz.  Some drivers
(especially Razer / Logitech "Hyperspeed") buffer mouse events at
1000 Hz internally.  Symptoms:

* Sub-tick fires at 180 Hz but only every ~3 events make it to the
  game (≈ 60 Hz effective).
* "Chunky" movement that doesn't match the per-frame trace.

Mitigation:

* In the mouse vendor utility, **disable any "ballistics"** or
  "acceleration" features.
* In Windows, run `regedit` and check
  `HKLM\SYSTEM\CurrentControlSet\Services\mouclass\Parameters` —
  the `MouseDataQueueSize` should be ≥ 50.
* If the driver coalesces hard, set
  `pull_subtick_hz = 120` in the profile (matches what most
  drivers can deliver without merging).

---

## 4. Detector latency (`loop_ms p99 > 25 ms`)

Detected via `scripts/trace_pull_motion.py`.  If the p99 is high
(detect is overrunning the frame budget), the cursor stalls for
1-2 frames every spike.  Options:

### 4.1  Lower `capture_fps`

Set `capture_fps = 45` in your profile.  Frame budget becomes 22 ms;
detect at p99 = 25 ms still overruns 10 % of frames, but 90 % are
clean.  Often a better feel than 60 fps with frequent overruns.

### 4.2  Tighten the detector

In `_APEX_TUNING`:

* `humanoid_min_height_pixels` default 48.  Raise to **64** if you
  only fight at close range (rejects small/distant targets but
  speeds clustering).
* `body_shape_min_score` default 0.40.  Raise to **0.55** if false
  positives are common (rejects ambiguous candidates faster).
* `min_target_area_pixels` default 20.  Raise to **60** to drop
  small clusters early.

### 4.3  Move detector to a worker thread

This is the structural fix.  The sub-tick loop already provides
continuous mouse output between captures; if detect runs on a worker
thread, the main loop's capture + pull + mouse_move becomes < 5 ms
and the sub-tick fires at full rate regardless of detect cost.
**Not yet implemented** — would require refactoring `AssistRuntime`'s
main loop into capture / detect / pull phases with thread-safe shared
state.  See the recommendations in `INTEGRATION.md`.

---

## 5. CPU usage

If the assist is using too much CPU:

| Knob | Default | Lower to | Effect |
|------|--------:|---------:|--------|
| `pull_subtick_hz` | 180 | 120 | -33 % sub-tick work, slightly chunkier pull |
| `overlay_fps` | 90 | 60 | -33 % overlay redraw work |
| `capture_fps` | 60 | 45 | -25 % main-loop work |

The `_precise_sleep` busy-wait that was costing 46 % CPU at 180 Hz is
already fixed — should be < 1 % now.

---

## 6. Anti-cheat / safety

* `mouse_gate_pull_budget_scale` default 3.5.  Lower to **2.5** to
  cap per-call mouse moves more tightly.  May cause the cursor to
  lag on long catch-ups (>40 px transient).
* `mouse_gate_stale_grace_frames` default 14.  Lower to **8** to
  release the lock faster when the enemy disappears.

---

## 7. Tuning workflow

When the feel is off:

1. Run `python3 scripts/trace_pull_motion.py`.  Look at:
   * `pull_dt_ms max` — should be < 50 ms during active engagement.
   * `loop_ms p99` — should be < 22 ms for live_trace.
   * `LOCK_VALID_BUT_NO_PULL` count — must be **0**.
   * `sweep[medium, 240 px/s]: steady_lag_mean` — should be ≤ 10 px.

2. Run `python3 scripts/prove_pull_dropout.py`.  Look at:
   * Phase B `LOCK_BUT_NO_PULL` — must be **0**.
   * `First-5-frame max recovery delta` — should be < 5 px.

3. If a number is off, find the matching root cause (A-S in the PR
   body) and the related config knob above.

4. Re-run tests:
   ```
   python3 -m pytest tests/test_motion_feel_scenarios.py \
                     tests/test_pull_cadence.py \
                     tests/test_pull_subtick_runtime.py \
                     tests/test_overlay_drag_integration.py
   ```
   Goal: all 36 motion / pull / cadence / overlay tests green.

5. Visual check: open
   `artifacts/real_apex_test/gif_166_proof/_pull_dropout_frames/`
   in order.  The dot/cursor/pull arrow should track smoothly through
   Phase A → B → C with no jumps.
