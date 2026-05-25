@echo off
REM Always opens run_hwid.bat with UAC elevation - use this if Admin PowerShell still says not elevated.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process -FilePath '%~dp0run_hwid.bat' -ArgumentList 'ELEVATED' -Verb RunAs -WorkingDirectory '%~dp0' -Wait"
exit /b %ERRORLEVEL%
