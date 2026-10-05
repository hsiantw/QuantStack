@echo off
set "PROJECT_DIR=%USERPROFILE%\Desktop\quant stack\QuantStack"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%PROJECT_DIR%\Start-LocalSite.ps1"
if errorlevel 1 pause
