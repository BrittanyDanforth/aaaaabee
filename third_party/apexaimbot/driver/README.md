# Logitech G HUB mouse DLL (optional)

ApexAimBot moves the cursor through **`ghub_mouse.dll`** (Logitech virtual HID), not generic `SendInput`.

ABA **cannot build or redistribute** this DLL — it is proprietary and tied to a specific G HUB version.

## Install (Windows)

1. Clone [ApexAimBot](https://github.com/1bit-monster7/ApexAimBot) or use your existing copy.
2. Copy from upstream `function/` into this folder:
   - `ghub_mouse.dll`
   - `logitech.driver.dll` (if present)
3. Or run:

```bat
python scripts\ensure_logitech_driver.py
```

(set `APEXAIMBOT_SRC` to your ApexAimBot repo root)

## Config

```json
"mouse_backend": "apexaimbot"
```

Tries Logitech DLL first, then **Win32 SendInput** (still good on Windows without G HUB).

Use `"logitech_ghub"` to require the DLL (fail if missing).

## vs Win32 SendInput

| Backend | Needs G HUB | Typical use |
|---------|-------------|-------------|
| `apexaimbot` | Optional | **Recommended** on Windows for Apex preset |
| `logitech_ghub` | Yes | 1:1 upstream injection path |
| `win32_sendinput` | No | Default `auto` on Windows |
| `pynput` | No | Linux dev / fallback |

Ban risk is the same for all synthetic input paths — see `ban_safety.py`.
