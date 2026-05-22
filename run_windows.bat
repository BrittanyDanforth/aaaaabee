@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Always run from this script's folder (Explorer double-click or cmd).
cd /d "%~dp0"
set "ROOT=%CD%"
set "LOGDIR=%ROOT%\logs"
set "LOGFILE=%LOGDIR%\aba_setup.log"
set "VENV=%ROOT%\.venv"
set "PY=%VENV%\Scripts\python.exe"
set "PIP=%VENV%\Scripts\pip.exe"
set "DEPS_OK=%VENV%\.deps_ok"
set "EXITCODE=1"

if not exist "%LOGDIR%" mkdir "%LOGDIR%" 2>nul
if not exist "%LOGDIR%" (
  echo ERROR: Cannot create folder: %LOGDIR%
  echo Install path may be read-only or blocked.
  pause
  exit /b 1
)

call :Log "=== ABA setup started ==="
call :Log "Root folder: %ROOT%"

REM --- HWID pre-step (separate tool; not part of ABA UI) ---
set "HWID_ROOT=%ROOT%\..\HWIDTool"
set "HWID_BAT=%HWID_ROOT%\run_hwid.bat"
if /I "%HWID_SKIP%"=="1" goto :HwidSkipped
if not exist "%HWID_BAT%" goto :HwidMissing
call :Log "Running HWID pre-step: %HWID_BAT%"
echo.
echo Running HWID pre-step - Administrator may be required...
set "HWID_QUIET=1"
call "%HWID_BAT%"
set "HWID_QUIET="
if errorlevel 2 goto :HwidVerifyFail
if errorlevel 1 goto :HwidFail
call :Log "HWID pre-step OK"
goto :HwidDone

:HwidSkipped
call :Log "HWID_SKIP=1 - skipping HWID pre-step"
echo Skipping HWID pre-step - HWID_SKIP=1.
goto :HwidDone

:HwidMissing
call :Log "HWID pre-step script not found at %HWID_BAT% - continuing"
echo Note: HWIDTool\run_hwid.bat not found - set HWID_SKIP=1 to continue without it.
goto :HwidDone

:HwidVerifyFail
set "FAILMSG=HWID verification failed. See HWIDTool\logs\hwid_verify_report.txt - reboot then verify-last, or HWID_FORCE=1."
goto :SetupFail

:HwidFail
set "FAILMSG=HWID pre-step failed. Run HWIDTool\run_hwid.bat as Administrator, or set HWID_SKIP=1."
goto :SetupFail

:HwidDone

echo.
echo ========================================
echo   ABA - Apex reference external assist
echo ========================================
echo   Folder: %ROOT%
echo   Log:    %LOGFILE%
echo   Default: apex_style_live_safe (real mouse + hooks after GUI ban ack).
echo   BAN RISK on live EAC/BattlEye/Vanguard — offline/private only.
echo   No injection into game process.
echo.

if exist "%PY%" goto :HaveVenv

REM --- Find Python 3.10+ (py -3, then python, then python3) ---
set "PY_BOOT="
set "PY_BOOT_DISPLAY="

py -3 -c "import sys; raise SystemExit(0 if sys.version_info>=(3,10) else 1)" >nul 2>&1
if not errorlevel 1 (
  set "PY_BOOT=py -3"
  set "PY_BOOT_DISPLAY=py -3"
  goto :FoundPython
)

python -c "import sys; raise SystemExit(0 if sys.version_info>=(3,10) else 1)" >nul 2>&1
if not errorlevel 1 (
  set "PY_BOOT=python"
  set "PY_BOOT_DISPLAY=python"
  goto :FoundPython
)

python3 -c "import sys; raise SystemExit(0 if sys.version_info>=(3,10) else 1)" >nul 2>&1
if not errorlevel 1 (
  set "PY_BOOT=python3"
  set "PY_BOOT_DISPLAY=python3"
  goto :FoundPython
)

set "FAILMSG=Python 3.10+ not found. Install from https://www.python.org/ and enable Add Python to PATH, or install the Python Launcher (py). Then run: python setup_doctor.py"
goto :SetupFail

:FoundPython
call :Log "Using Python: !PY_BOOT_DISPLAY!"
echo Found Python: !PY_BOOT_DISPLAY!
!PY_BOOT! --version
if errorlevel 1 (
  set "FAILMSG=Python !PY_BOOT_DISPLAY! failed --version check."
  goto :SetupFail
)
!PY_BOOT! --version >> "%LOGFILE%" 2>&1

call :Log "Creating virtual environment..."
echo Creating virtual environment at:
echo   "%VENV%"
!PY_BOOT! -m venv "%VENV%"
if errorlevel 1 (
  set "FAILMSG=Failed to create .venv with !PY_BOOT_DISPLAY!. Run setup_doctor.py for details."
  goto :SetupFail
)

if not exist "%PY%" (
  set "FAILMSG=Missing after venv create: %PY%"
  goto :SetupFail
)
if not exist "%PIP%" (
  set "FAILMSG=Missing after venv create: %PIP%"
  goto :SetupFail
)
call :Log "Virtual environment OK."
echo Virtual environment created.

:HaveVenv
if not exist "%PY%" (
  set "FAILMSG=Missing %PY% — delete the .venv folder and run this script again."
  goto :SetupFail
)
if not exist "%PIP%" (
  set "FAILMSG=Missing %PIP% — delete the .venv folder and run this script again."
  goto :SetupFail
)

call :Log "Venv Python: %PY%"
echo Using venv Python:
echo   "%PY%"
"%PY%" --version
if errorlevel 1 (
  set "FAILMSG=Venv Python failed: %PY%"
  goto :SetupFail
)
"%PY%" --version >> "%LOGFILE%" 2>&1

if exist "%DEPS_OK%" goto :DepsDone

call :Log "Installing dependencies..."
echo Installing dependencies (PyPI only; delete .deps_ok to reinstall)...
set "PIP_NO_CACHE_DIR=1"
"%PY%" -m pip install --upgrade pip
if errorlevel 1 (
  set "FAILMSG=pip upgrade failed."
  goto :SetupFail
)
"%PY%" -m pip install -r "%ROOT%\requirements.txt"
if errorlevel 1 (
  set "FAILMSG=pip install -r requirements.txt failed."
  goto :SetupFail
)
echo ok>"%DEPS_OK%"
call :Log "Dependencies installed."
echo Dependencies installed.

:DepsDone
echo.
echo Running setup doctor...
"%PY%" setup_doctor.py --require-venv
if errorlevel 1 (
  set "FAILMSG=Setup doctor found problems. See output above."
  goto :SetupFail
)

echo.
echo Running self-check...
call :Log "Self-check starting"
"%PY%" aba.py --self-check
if errorlevel 1 (
  set "FAILMSG=Self-check failed. See logs\aba_selfcheck.log"
  goto :SetupFail
)

call :Log "Self-check passed"
echo.
echo Self-check passed. Launching ABA UI...
echo STATUS can show r5apex.exe presence — do NOT use live assist online.
call :Log "Launching aba.py"
"%PY%" aba.py
set "EXITCODE=!ERRORLEVEL!"
call :Log "aba.py exited with code !EXITCODE!"
echo.
echo ABA closed (exit !EXITCODE!). Log: %LOGFILE%
pause
endlocal & exit /b %EXITCODE%

:SetupFail
if not defined FAILMSG set "FAILMSG=Unknown setup error"
call :Log "SETUP FAILED: !FAILMSG!"
echo.
echo *** SETUP FAILED ***
echo !FAILMSG!
echo.
echo Full log: %LOGFILE%
echo Run: cd /d "%ROOT%"
echo      python setup_doctor.py
pause
endlocal & exit /b 1

:Log
set "LOGMSG=%~1"
echo !LOGMSG!
>>"%LOGFILE%" echo !LOGMSG!
exit /b 0
