@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Paths with parentheses e.g. "Downloads\repo (1)" break IF ( ) blocks - use GOTO.
cd /d "%~dp0"
set "ROOT=%CD%"
set "ABA_ROOT=%ROOT%\..\OverlayAssist"
set "LOGDIR=%ROOT%\logs"
set "LOGFILE=%LOGDIR%\hwid_setup.log"
set "MARKER=%LOGDIR%\hwid_step.ok"
set "EXE="
call :ResolveExe
set "SKIP_APPLY=0"
set "FAILMSG="

if not "%~1"=="ELEVATED" goto :MaybeElevate
shift
goto :Main

:MaybeElevate
if /I "%HWID_NO_ELEVATE%"=="1" goto :Main
call :TryElevate
if not "!HWID_ELEVATED_CHILD!"=="1" goto :Main
endlocal & exit /b !HWID_CHILD_EXIT!

:TryElevate
if not exist "%EXE%" goto :TryElevateFsutil
"%EXE%" --check-admin >nul 2>&1
if not errorlevel 1 exit /b 0
:TryElevateFsutil
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
set "HWID_ELEVATED_CHILD=1"
powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process -FilePath 'cmd.exe' -ArgumentList '/c \"\"\"%~f0\"\" ELEVATED\"' -Verb RunAs -WorkingDirectory '%ROOT%' -Wait"
set "HWID_CHILD_EXIT=!ERRORLEVEL!"
exit /b !HWID_CHILD_EXIT!

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
if errorlevel 1 goto :NotAdmin

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
if exist "%MARKER%" goto :DoneOk
set "FAILMSG=Verification failed - full BEFORE/AFTER printed above."
goto :Fail

:NotAdmin
set "FAILMSG=Not Administrator. Double-click Run_As_Admin.bat in this folder."
goto :Fail

:DoVerifyLast
echo Re-checking saved BEFORE and AFTER...
"%EXE%" --verify-last
set "EXITCODE=!ERRORLEVEL!"
if "!EXITCODE!"=="0" goto :DoneOk
if "!EXITCODE!"=="2" goto :VerifyFailed
set "FAILMSG=verify-last failed code !EXITCODE!"
goto :Fail

:EnsureExe
call :ResolveExe
if not defined EXE goto :EnsureBuild
if not exist "%EXE%" goto :EnsureBuild
exit /b 0
:EnsureBuild
where cargo >nul 2>&1
if errorlevel 1 goto :EnsureNoCargo
echo Building hwspoof.exe - first run may take several minutes...
pushd "%ROOT%"
cargo build --release
set "BERR=!ERRORLEVEL!"
popd
if not "!BERR!"=="0" goto :EnsureCargoFail
call :ResolveExe
if not defined EXE goto :EnsureExeMissing
if exist "%EXE%" exit /b 0
set "FAILMSG=Build finished but resolved EXE path does not exist: %EXE%"
exit /b 1
:EnsureNoCargo
set "FAILMSG=Install Rust from https://rustup.rs/ or use Run_As_Admin after building once."
exit /b 1
:EnsureCargoFail
set "FAILMSG=cargo build failed."
exit /b 1
:EnsureExeMissing
set "FAILMSG=Build finished but hwspoof.exe was not produced under HWIDTool\target. Open a shell here and run 'cargo build --release' to see the real error."
exit /b 1

:ResolveExe
set "EXE="
if exist "%ROOT%\target\release\hwspoof.exe" goto :ExeRelease
if exist "%ROOT%\target\x86_64-pc-windows-msvc\release\hwspoof.exe" goto :ExeMsvc
if exist "%ROOT%\target\x86_64-pc-windows-gnu\release\hwspoof.exe" goto :ExeGnu
if not exist "%ROOT%\target" exit /b 0
for /r "%ROOT%\target" %%E in (hwspoof.exe) do (
  if not defined EXE set "EXE=%%E"
)
exit /b 0
:ExeRelease
set "EXE=%ROOT%\target\release\hwspoof.exe"
exit /b 0
:ExeMsvc
set "EXE=%ROOT%\target\x86_64-pc-windows-msvc\release\hwspoof.exe"
exit /b 0
:ExeGnu
set "EXE=%ROOT%\target\x86_64-pc-windows-gnu\release\hwspoof.exe"
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
