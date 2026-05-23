# Run path (HWID then ABA)

## 1) HWID pre-step (hardware only — not ABA)

**Double-click** `HWIDTool\Run_As_Admin.bat` (best — fixes false "need admin" in PowerShell).

If `cargo build` fails with **memory allocation** or **can't find crate for std**, that is usually out-of-RAM during compile, not a broken Rust install. Close apps, retry, or copy a prebuilt `hwspoof.exe` into `HWIDTool\bin\`. To install ABA without HWID: `set HWID_SKIP=1` then `run_windows.bat`.

The full BEFORE/AFTER report prints in the terminal. You do not need to open log files.

Optional logs: `HWIDTool\logs\hwid_verify_report.txt` — look for **VERDICT: SUCCESS** and **CHANGED ✓**.  
If **UNCHANGED** on disk/SMBIOS: restart Windows, then:

```bat
cd HWIDTool
target\release\hwspoof.exe --verify-last
```

## 2) ABA (default: live safe)

```bat
cd OverlayAssist
run_windows.bat
```

Or if `.venv` already exists:

```bat
cd OverlayAssist
.venv\Scripts\python.exe aba.py
```

| Setting | Default |
|---------|---------|
| Profile | `apex_style_live_safe` |
| Mouse | **Real OS movement** when ADS + target (after ban acknowledgment) |
| Hooks | **On** when running (RMB / F8 kill switch) |
| Status when assisting | **ACTIVE LIVE** |
| Capture | 30 FPS cap, pauses when game closed |

Dry-run only (no mouse): set `"profile": "apex_style_dry_run"` and `"allow_live_mouse": false` in `config.json`.

## Ban honesty

Live safe still uses capture + hooks + synthetic mouse. **Assume account loss** on online EAC Apex unless you use an offline/private build.
