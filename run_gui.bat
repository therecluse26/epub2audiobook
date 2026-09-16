@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo Virtual environment not found. Run setup.ps1 first.
  pause
  exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" -m epub2audiobook %1
