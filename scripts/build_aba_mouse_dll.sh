#!/usr/bin/env bash
# Cross-build aba_mouse.dll on Linux (mingw-w64). Optional — Windows users can use .bat
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/third_party/aba_mouse_driver/aba_mouse.c"
DEF="$ROOT/third_party/aba_mouse_driver/aba_mouse.def"
OUT_DIR="$ROOT/third_party/apexaimbot/driver"
OUT="$OUT_DIR/aba_mouse.dll"
CC="${MINGW_CC:-x86_64-w64-mingw32-gcc}"

mkdir -p "$OUT_DIR"
if ! command -v "$CC" >/dev/null 2>&1; then
  echo "Install mingw: sudo apt install mingw-w64" >&2
  exit 1
fi
"$CC" -shared -O2 -o "$OUT" "$SRC" "$DEF" -luser32 -lkernel32 -static-libgcc
echo "Built $OUT"
sha256sum "$OUT"
