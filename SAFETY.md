# ABA safety and trust model

## Account ban risk — read first

**Using ABA while Easy Anti-Cheat, BattlEye, Vanguard, or similar is active on a live game WILL risk a permanent account or hardware ban.**

“No injection” and “external only” do **not** make this safe. Anti-cheat detects:

| Profile | `apex_style_dry_run` | `apexaimbot` / `owned_dev_live` + explicit `allow_live_mouse: true` |
| Behavior | Default when `allow_live_mouse: false` | When `allow_live_mouse: true` |
| 30 FPS dry-run cap | Mask/detection **sanity check only** — not fast-tracking gameplay proof | Use `apex_style_perf_test` + `aba.py --benchmark` for timing |
|----------|----------------------------------------|-------------------------------|
| Screen capture (mss) | Yes while runtime active (throttled to **30 FPS** in `apex_style_dry_run`; **no capture when paused** / game closed) | Yes |
| Benchmark / self-check / Debug HUD | Blocked while `target_process_name` is running | Same |
| Global input hooks (pynput) | **No** | **Yes** — same class as cheats |
| Synthetic mouse (Win32) | **No** | **Yes** |
| Process scan for `r5apex.exe` | Yes (GUI + runtime) | Yes |

**Apex Legends uses EAC on live clients.** Firing range with `r5apex.exe` online is still EAC unless you use a special offline build.

### Shipped default (`config.json`)

```json
"offline_dev_mode": true,
"profile": "apexaimbot",
"allow_live_mouse": true
```

- **Shipped ApexAimBot:** YOLO + Apex PID, with GUI ban acknowledgment before live assist starts.
- **Dry-run option:** set `"profile": "apex_style_dry_run"` and `"allow_live_mouse": false` for capture + detection telemetry only; **no** OS mouse movement; **no** global hooks.
- **Live assist:** set `"allow_live_mouse": true` only for offline/private builds **without** anti-cheat. GUI requires ban acknowledgment. Config blocks `allow_live_mouse` + `r5apex.exe` together.

---

## Process detection (weak signal)

`target_process_name` only checks Task Manager executable name — spoofable, not proof the real game is running.

---

## What ABA does NOT do

- No DLL injection, hooks **into the game process**, or memory reads
- No registry / startup persistence
- No admin in `run_windows.bat`
- No hidden downloads (pip from `requirements.txt` / `requirements-yolo.txt` only)

## What ABA still does (ban-relevant)

- Desktop screen capture
- **Global** low-level hooks when `allow_live_mouse: true` (pynput — not “safe because not in-game”)
- `GetAsyncKeyState` RMB poll when `ads_input_mode` is `both` / `win32_poll`
- Synthetic mouse via `mouse_event` when live assist is enabled

---

## Windows setup bugs (fixed)

Earlier `run_windows.bat` failures were caused by unquoted `:Log` (only first word logged), `%time%` colons breaking `>>` redirects, and `goto :SetupFail "msg"` not passing messages. Current script uses quoted logs and `set FAILMSG` + `goto :SetupFail`.

---

## Setup script (`run_windows.bat`)

- Plain `cmd` — no PowerShell persistence
- Quoted paths for spaces in `C:\Users\...\OverlayAssist`
- Single failure exit (`pause` once)
- `setup_doctor.py --require-venv` before UI

---

## Runtime safety

- **Start ABA** — `RuntimeController` → `AssistRuntime` (dry-run or live per config)
- **Stop ABA** — stop flag, ADS clear, mouse gated off, join up to 8s
- **F8** — kill switch **only when `allow_live_mouse: true`**
- **Target closed** — pause pull/mouse when `pause_on_target_closed: true`
- **`max_pull_speed_pixels_per_frame`** capped at 80

---

## If something goes wrong

1. **Stop ABA** or close the window  
2. Send `logs/aba_setup.log`, `logs/aba_selfcheck.log`  
3. Never use `allow_live_mouse: true` on live Apex / ranked / online
