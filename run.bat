@echo off
setlocal enableextensions enabledelayedexpansion

set "ROOT=%~dp0"
cd /d "%ROOT%"

set "PY=%ROOT%python-portable\python.exe"

if not exist "%PY%" (
  echo [ERROR] Не знайдено python-portable\python.exe
  pause
  exit /b 1
)

"%PY%" -u app.py

endlocal
