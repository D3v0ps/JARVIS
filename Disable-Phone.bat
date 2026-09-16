@echo off
REM Takes the phone door down again.
REM Double-click this file - no terminal needed.
cd /d "%~dp0"
title J.A.R.V.I.S. - Disabling the phone

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo   JARVIS is not installed yet.
    echo.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" scripts\enable_phone.py --off
echo.
pause
