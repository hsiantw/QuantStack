@echo off
setlocal
cd /d "%~dp0"
set "mode=%~1"
if not defined mode (
  echo H: High - 6 workers, up to 45 minutes
  echo L: Low  - 1 worker, up to 10 minutes, reduced priority
  choice /C HL /N /M "Choose mode [H/L]: "
  if errorlevel 2 (set "mode=low") else (set "mode=high")
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Update-Charts.ps1" -Mode "%mode%"
set "result=%errorlevel%"
if "%~1"=="" pause
exit /b %result%
