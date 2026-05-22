@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM HWID pre-step - use Run_As_Admin.bat if Admin PowerShell still fails
cd /d "%~dp0"
set "ROOT=%CD%"
set "ABA_ROOT=%ROOT%\..\OverlayAssist"
set "LOGDIR=%ROOT%\logs"
set "LOGFILE=%LOGDIR%\hwid_setup.log"
set "MARKER=%LOGDIR%\hwid_step.ok"
set "EXE=%ROOT%\target\x86_64-pc-windows-msvc\release\hwspoof.exe"
if not exist "%EXE%" set "EXE=%ROOT%\target\release\hwspoof.exe"
set "SKIP_APPLY=0"
set "FAILMSG="

if not "%~1"=="ELEVATED" goto :MaybeElevate
shift
goto :Main

:MaybeElevate
if /I not "%HWID_NO_ELEVATE%"=="1" call :TryElevate
goto :Main

:TryElevate
if exist "%EXE%" (
  "%EXE%" --check-admin >nul 2>&1
  if not errorlevel 1 exit /b 0
)
fsutil dirty query %systemdrive% >nul 2>&1
if not errorlevel 1 exit /b 0
whoami /groups | findstr /i "S-1-16-12288" >nul 2>&1
if not errorlevel 1 exit /b 0
whoami /groups | findstr /i "High Mandatory Level" >nul 2>&1
if not errorlevel 1 exit /b 0
echo.
echo This window is NOT Administrator.
echo Opening UAC prompt - click Yes...
echo.
powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process -FilePath 'cmd.exe' -ArgumentList '/c \"\"\"%~f0\"\" ELEVATED\"' -Verb RunAs -WorkingDirectory '%ROOT%' -Wait"
set "ELEV_ERR=!ERRORLEVEL!"
exit /b !ELEV_ERR!

:Main
if not exist "%LOGDIR%" mkdir "%LOGDIR%" 2>nul

echo.
echo ========================================
echo   HWID Spoofer - hardware IDs only
echo ========================================
echo   Folder: %ROOT%
echo.

call :EnsureExe
if errorlevel 1 goto :Fail

"%EXE%" --check-admin
if errorlevel 1 (
  set "FAILMSG=Not Administrator. Double-click Run_As_Admin.bat in this folder."
  goto :Fail
)

if exist "%MARKER%" if /I not "!HWID_FORCE!"=="1" set "SKIP_APPLY=1"

if "!SKIP_APPLY!"=="1" goto :DoVerifyLast

echo.
echo ===== BEFORE - current hardware IDs =====
"%EXE%" --report
echo.

echo ===== APPLYING SPOOF =====
"%EXE%" --apply-all
set "EXITCODE=!ERRORLEVEL!"
if "!EXITCODE!"=="2" goto :VerifyFailed
if not "!EXITCODE!"=="0" goto :ApplyExitFail
if not exist "%MARKER%" (
  set "FAILMSG=Verification failed - full BEFORE/AFTER printed above."
  goto :Fail
)
goto :DoneOk

:DoVerifyLast
echo Re-checking saved BEFORE and AFTER...
"%EXE%" --verify-last
set "EXITCODE=!ERRORLEVEL!"
if "!EXITCODE!"=="0" goto :DoneOk
if "!EXITCODE!"=="2" goto :VerifyFailed
set "FAILMSG=verify-last failed code !EXITCODE!"
goto :Fail

:EnsureExe
if exist "%EXE%" exit /b 0
where cargo >nul 2>&1
if errorlevel 1 (
  if not exist "%EXE%" (
    set "FAILMSG=Install Rust from https://rustup.rs/ or use Run_As_Admin after building once."
    exit /b 1
  )
  exit /b 0
)
echo Building hwspoof.exe - first run may take several minutes...
pushd "%ROOT%"
cargo build --release
set "BERR=!ERRORLEVEL!"
popd
if not "!BERR!"=="0" (
  set "FAILMSG=cargo build failed."
  exit /b 1
)
if not exist "%EXE%" (
  set "FAILMSG=Build finished but EXE missing."
  exit /b 1
)
exit /b 0

:VerifyFailed
set "FAILMSG=VERIFICATION FAILED - see full BEFORE/AFTER report above."
goto :Fail

:ApplyExitFail
set "FAILMSG=hwspoof failed with exit code !EXITCODE!"
goto :Fail

:DoneOk
echo.
echo ===== HWID DONE - NEXT RUN ABA =====
echo   cd /d "%ABA_ROOT%"
echo   run_windows.bat
echo =====================================
if /I not "!HWID_QUIET!"=="1" pause
endlocal & exit /b 0

:Fail
if not defined FAILMSG set "FAILMSG=Unknown error"
echo.
echo ***** HWID FAILED *****
echo !FAILMSG!
echo.
echo Try: double-click Run_As_Admin.bat
echo Logs copy: %LOGFILE%
if /I not "!HWID_QUIET!"=="1" pause
endlocal & exit /b 1
