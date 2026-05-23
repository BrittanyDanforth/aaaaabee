# Deep audit (second pass)

Confirmed fixes applied — tuning constants unchanged.

## Runtime / safety
- `try/finally` teardown: listeners, overlay, OpenCV always stopped on exit/exception
- Stats `begin_frame`/`update` under lock; no post-stop frame write
- `snapshot.running` = assist loop active (`tel["running"]`), not thread-alive alone
- `STOPPING` until join completes; controller `_stopping` not cleared in worker `finally` early
- Shared `ProcessPresenceDebouncer` on `RuntimeController` (GUI + runtime + gate)
- Gate uses debounced process check
- Overlay FOV center uses `crosshair_offset`
- Re-check `_should_run()` before mouse move and overlay update

## UI honesty
- `ACTIVE LIVE` only when `dry_run=false`; `ACTIVE SIMULATED` when dry-run
- `GAME CLOSED` when runtime paused (single debouncer)
- Telemetry clears to `unavailable` / `waiting…` when stats invalid
- Benchmark text: not inverted; persists after stop; blocked while runtime on
- Kill/hooks labels match dry-run vs live

## Windows / paths
- `resolve_config_path` anchored under `OverlayAssist/`
- Debug HUD config containment via `relative_to`
- Self-check `require_venv` on Windows
- `aba.py --debug` matches assist debug flags
- Config reload from disk on Start / Benchmark

## Not changed (evidence-based)
- `_APEX_TUNING` numeric values — scenario matrix still passes
- 30 FPS dry-run cap — still appropriate for safe sanity

Run on Windows: `aba.py --benchmark`, `tuning_analysis.py`, firing range visual check.
