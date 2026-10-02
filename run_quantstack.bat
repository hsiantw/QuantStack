@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" serve_quantstack.py
) else (
  python serve_quantstack.py
)
