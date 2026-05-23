# ABA on Windows

Use a **controlled offline / dev** environment (firing range, private build). **Not verified for online play or anti-cheat.**

## One-command start

```bat
run_windows.bat
```

- `cd /d` to the batch folder (works with spaces in path)
- Finds Python: **py -3** → **python** → **python3**
- Creates `.venv` with quoted paths
- Stops immediately on first failure (no cascade of fake errors)
- Runs `setup_doctor.py` then `aba.py --self-check`
- Opens the **ABA** UI

Failures **pause once** with a clear message. Log: `logs\aba_setup.log`

### If setup fails

```bat
cd path\to\OverlayAssist
python setup_doctor.py
```

Or after venv exists:

```bat
.venv\Scripts\python.exe setup_doctor.py --require-venv
```

## Manual

```powershell
cd OverlayAssist
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python aba.py --self-check
python aba.py
```

## Apex / r5apex.exe

| Item | Detail |
|------|--------|
| Executable | `r5apex.exe` (Steam / EA App) |
| Detection | Name in process list only — **not** memory, not injection |
| Spoofing | Any process can be named `r5apex.exe` — weak signal |
| UI | STATUS updates from real `process_presence.py` poll |

Change for your game:

```json
"target_process_name": "YourGame.exe"
```

## GUI crash `cannot use geometry manager pack inside ... grid`

Fixed in current `aba_gui.py`: do not mix `pack` and `grid` in the same Tk parent. Pull latest and re-run `run_windows.bat`.

If the UI still fails, run:

```bat
.venv\Scripts\python.exe setup_doctor.py
```

Check the **tkinter_gui** line.

## UI controls

| Control | Action |
|---------|--------|
| **Start ABA** | Same runtime as `assist.py --cli` |
| **Stop ABA** | Stops thread, clears ADS latch, blocks mouse |
| **F8** (configurable) | Kill switch — **live assist only**; inactive in default dry-run |
| **Debug HUD** | Separate OpenCV process; **Q** to quit |
| **Run Self-Check** | Same config path as UI |
| **Close** | Stops runtime, terminates debug HUD if running |

## Settings

| Key | Recommendation |
|-----|----------------|
| `profile` | `apex_style_dry_run` |
| `mouse_backend` | `auto` → Win32 on Windows |
| `allow_live_mouse` | **`false`** default (dry-run). **`true`** = live assist — **will ban** on live EAC |
| `ads_input_mode` | Used only if `allow_live_mouse: true`; `both` adds global RMB poll |
| `monitor_index` | `1` = primary |
| `capture_fov_crop` | `true` |
| `pause_on_target_closed` | `true` — no pull when game closed |

## Borderless vs fullscreen

| Mode | Capture |
|------|---------|
| Borderless windowed | Best |
| Exclusive fullscreen | Often black/wrong monitor — avoid |

## Multi-monitor / DPI

- Set `monitor_index` to the monitor where the game runs (self-check lists monitors).
- Crosshair offset in config if DPI scaling misaligns capture center.

## If it fails

Send:

1. `logs/aba_setup.log`
2. `logs/aba_selfcheck.log`
3. Screenshot of ABA STATUS + telemetry panel
4. Borderless yes/no, monitor index, game resolution
