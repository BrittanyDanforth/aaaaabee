@echo off
REM Explorer / "Run as administrator" uses cmd /c - stay open on errors.
REM NOTE: Paths with parentheses e.g. "folder (1)" break IF ( ) blocks - use GOTO only.
if /I "%~1"=="_aba_run" goto :AbaMain
cd /d "%~dp0"
cmd /k call "%~f0" _aba_run %*
exit /b 0

:AbaMain
shift
setlocal EnableExtensions EnableDelayedExpansion

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

net session >nul 2>&1
if errorlevel 1 goto :AfterAdmin
set "ABA_IS_ADMIN=1"
set "HWID_NO_ELEVATE=1"
for /f "skip=2 tokens=1,*" %%A in ('reg query "HKCU\Environment" /v Path 2^>nul') do (
  if not "%%B"=="" set "PATH=%%B;!PATH!"
)
:AfterAdmin

if exist "%LOGDIR%" goto :LogdirOk
echo ERROR: Cannot create folder: %LOGDIR%
echo Install path may be read-only or blocked.
echo Tip: avoid parentheses in the folder path, e.g. move out of "Downloads\... (1)".
pause
exit /b 1
:LogdirOk

call :Log "=== ABA setup started ==="
call :Log "Root folder: %ROOT%"
if defined ABA_IS_ADMIN call :Log "Running elevated - merged HKCU Path for Python discovery"

set "HWID_BAT=%ROOT%\HWIDTool\run_hwid.bat"
if not exist "%HWID_BAT%" set "HWID_BAT=%ROOT%\..\HWIDTool\run_hwid.bat"
if /I "%HWID_SKIP%"=="1" goto :HwidSkipped
if not exist "%HWID_BAT%" goto :HwidMissing
call :Log "Running HWID pre-step: %HWID_BAT%"
echo.
echo Running HWID pre-step - Administrator may be required...
set "HWID_QUIET=1"
set "HWID_ARG="
if defined ABA_IS_ADMIN set "HWID_ARG=ELEVATED"
call "%HWID_BAT%" !HWID_ARG!
set "HWID_ERR=!ERRORLEVEL!"
set "HWID_QUIET="
if "!HWID_ERR!"=="2" goto :HwidVerifyFail
if "!HWID_ERR!" GEQ "1" goto :HwidFail
call :Log "HWID pre-step OK"
goto :HwidDone

:HwidSkipped
call :Log "HWID_SKIP=1 - skipping HWID pre-step"
echo Skipping HWID pre-step - HWID_SKIP=1.
goto :HwidDone

:HwidMissing
call :Log "HWID pre-step script not found - continuing"
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
echo   BAN RISK on live EAC/BattlEye/Vanguard - offline/private only.
echo   No injection into game process.
echo.

call :VerifyVenv
if not errorlevel 1 goto :HaveVenv

set "PY_BOOT="
set "PY_BOOT_DISPLAY="

py -3 -c "import sys; raise SystemExit(0 if sys.version_info>=(3,10) else 1)" >nul 2>&1
if not errorlevel 1 goto :UsePyLauncher
python -c "import sys; raise SystemExit(0 if sys.version_info>=(3,10) else 1)" >nul 2>&1
if not errorlevel 1 goto :UsePython
python3 -c "import sys; raise SystemExit(0 if sys.version_info>=(3,10) else 1)" >nul 2>&1
if not errorlevel 1 goto :UsePython3

set "FAILMSG=Python 3.10+ not found. Install from https://www.python.org/ and enable Add Python to PATH, or install the Python Launcher (py). Then run: python setup_doctor.py"
goto :SetupFail

:UsePyLauncher
set "PY_BOOT=py -3"
set "PY_BOOT_DISPLAY=py -3"
goto :FoundPython

:UsePython
set "PY_BOOT=python"
set "PY_BOOT_DISPLAY=python"
goto :FoundPython

:UsePython3
set "PY_BOOT=python3"
set "PY_BOOT_DISPLAY=python3"
goto :FoundPython

:FoundPython
call :Log "Using Python: !PY_BOOT_DISPLAY!"
echo Found Python: !PY_BOOT_DISPLAY!
!PY_BOOT! --version
if errorlevel 1 goto :PyVersionFail
!PY_BOOT! --version >> "%LOGFILE%" 2>&1

call :Log "Creating virtual environment..."
echo Creating virtual environment at:
echo   "%VENV%"
set "ENSURE_VENV=%ROOT%\scripts\ensure_venv.py"
if not exist "%ENSURE_VENV%" goto :VenvBatchFallback
!PY_BOOT! "%ENSURE_VENV%" "%ROOT%"
if errorlevel 1 goto :VenvCreateFail
call :VerifyVenv
if errorlevel 1 goto :VenvCreateFail
call :Log "Virtual environment OK."
echo Virtual environment created.
goto :HaveVenv

:VenvBatchFallback
if exist "%VENV%" rmdir /s /q "%VENV%" 2>nul
!PY_BOOT! -m venv "%VENV%"
if errorlevel 1 goto :VenvCreateFail
call :VerifyVenv
if errorlevel 1 goto :VenvCreateFail
call :Log "Virtual environment OK."
echo Virtual environment created.
goto :HaveVenv

:HaveVenv
call :VerifyVenv
if errorlevel 1 goto :VenvBroken

call :Log "Venv Python: %PY%"
echo Using venv Python:
echo   "%PY%"
"%PY%" --version
if errorlevel 1 goto :VenvRunFail
"%PY%" --version >> "%LOGFILE%" 2>&1

if not exist "%DEPS_OK%" goto :DoInstall

"%PY%" -c "import numpy, cv2, mss, psutil, pynput" 1>nul 2>nul
if not errorlevel 1 goto :DepsDone
call :Log "DEPS_OK marker present but imports failed - reinstalling."
echo Dependency check failed - reinstalling missing packages...
del "%DEPS_OK%" 1>nul 2>nul

:DoInstall
call :Log "Installing dependencies..."
echo Installing dependencies (PyPI only; delete .deps_ok to reinstall)...
set "PIP_NO_CACHE_DIR=1"
"%PY%" -m pip install --upgrade pip
if errorlevel 1 goto :PipUpgradeFail
"%PY%" -m pip install -r "%ROOT%\requirements.txt"
if errorlevel 1 goto :PipInstallFail
echo ok>"%DEPS_OK%"
call :Log "Dependencies installed."
echo Dependencies installed.

:DepsDone
echo.
echo Running setup doctor...
"%PY%" setup_doctor.py --require-venv
if errorlevel 1 goto :DoctorFail

echo.
echo Running self-check...
call :Log "Self-check starting"
"%PY%" aba.py --self-check
if errorlevel 1 goto :SelfCheckFail

call :Log "Self-check passed"
echo.
echo Self-check passed. Launching ABA UI...
echo STATUS can show r5apex.exe presence - do NOT use live assist online.
call :Log "Launching aba.py"
"%PY%" aba.py
set "EXITCODE=!ERRORLEVEL!"
call :Log "aba.py exited with code !EXITCODE!"
echo.
echo ABA closed (exit !EXITCODE!). Log: %LOGFILE%
pause
endlocal & exit /b !EXITCODE!

:PyVersionFail
set "FAILMSG=Python !PY_BOOT_DISPLAY! failed --version check."
goto :SetupFail

:VenvCreateFail
set "FAILMSG=Failed to create .venv with !PY_BOOT_DISPLAY!. Delete the .venv folder, move the project to a shorter path (e.g. C:\OverlayAssist), then run this script again."
goto :SetupFail

:VenvBroken
set "FAILMSG=Broken .venv at %VENV% - delete the .venv folder or move the project to C:\OverlayAssist (long Downloads paths often break venv), then run run_windows.bat again."
goto :SetupFail

:VenvRunFail
set "FAILMSG=Venv Python failed: %PY%"
goto :SetupFail

:PipUpgradeFail
set "FAILMSG=pip upgrade failed."
goto :SetupFail

:PipInstallFail
set "FAILMSG=pip install -r requirements.txt failed."
goto :SetupFail

:DoctorFail
set "FAILMSG=Setup doctor found problems. See output above."
goto :SetupFail

:SelfCheckFail
set "FAILMSG=Self-check failed. See logs\aba_selfcheck.log"
goto :SetupFail

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

:VerifyVenv
if not exist "%PY%" exit /b 1
if not exist "%PIP%" exit /b 1
"%PY%" -c "import sys" 1>nul 2>nul
if errorlevel 1 exit /b 1
exit /b 0

:Log
set "LOGMSG=%~1"
echo !LOGMSG!
>>"%LOGFILE%" echo !LOGMSG!
exit /b 0
