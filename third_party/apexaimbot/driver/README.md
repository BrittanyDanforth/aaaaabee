# Mouse DLL — `aba_mouse.dll` (safe, built from source)

## Do NOT download random `ghub_mouse.dll` files

Cheat repos often ship **malware disguised as mouse DLLs**. ABA does **not** use those.

We ship **`aba_mouse.dll`**: open-source, built from:

`third_party/aba_mouse_driver/aba_mouse.c`

Uses only Windows **SendInput** (same effect as our Python `win32_sendinput` backend, exposed as a DLL for Apex API compatibility).

## SHA-256 (verify after rebuild)

| File | SHA-256 |
|------|---------|
| `aba_mouse.dll` | `4f9c8d6e5b7de1a0d9eb2bebea92352e43ced56812758727fbfc05a34df35992` |

If you rebuild on Windows, run `certutil -hashfile aba_mouse.dll SHA256` and compare.

## Rebuild (optional)

```bat
scripts\build_aba_mouse_dll.bat
```

Or Linux cross-compile:

```bash
./scripts/build_aba_mouse_dll.sh
```

## Config

```json
"mouse_backend": "apexaimbot"
```

Loads `aba_mouse.dll` from this folder first. Optional legacy `ghub_mouse.dll` (user-supplied only, not recommended).

## No download step

The DLL is **in the repo** or **built locally** from C source — never fetched from the internet by ABA.
