@echo off
REM Build aba_mouse.dll from open-source C (no third-party cheat DLLs).
setlocal
cd /d "%~dp0.."
set SRC=third_party\aba_mouse_driver\aba_mouse.c
set DEF=third_party\aba_mouse_driver\aba_mouse.def
set OUT=third_party\apexaimbot\driver\aba_mouse.dll

if not exist "third_party\apexaimbot\driver" mkdir "third_party\apexaimbot\driver"

where cl >nul 2>&1
if %ERRORLEVEL%==0 (
  cl /nologo /LD /O2 "%SRC%" /Fe:"%OUT%" user32.lib /link /DEF:"%DEF%"
  if errorlevel 1 exit /b 1
  echo Built %OUT% with MSVC
  exit /b 0
)

where gcc >nul 2>&1
if %ERRORLEVEL%==0 (
  gcc -shared -O2 -o "%OUT%" "%SRC%" "%DEF%" -luser32 -lkernel32
  if errorlevel 1 exit /b 1
  echo Built %OUT% with gcc
  exit /b 0
)

echo Install Visual Studio Build Tools or MinGW-w64, then re-run this script.
exit /b 1
