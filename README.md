# OverlayAssist targeting modules

## What to copy into your `OverlayAssist` folder

Only **two** files are required for the self-check fix:

| File | Replaces |
|------|----------|
| `detector.py` | Your existing `detector.py` (full body-structure targeting — **one file**, not `detection.py`) |
| `self_check.py` | Your existing `self_check.py` (body dummy test + debug logging; helpers are **inside** this file) |

Do **not** copy `detection.py` — that name was repo-only. Your project already uses `detector.py`.

Do **not** copy `synthetic_selfcheck.py` — it is inlined into `self_check.py`.

Optional (earlier fixes): `overlay_assist.py`, `motion.py`, `setup_doctor.py` if you use them.

Do **not** replace `aba.py` or `run_windows.bat`.

## Verify

```powershell
python setup_doctor.py
python aba.py --self-check
```

Expect: `Detection (synthetic body dummy): OK`
