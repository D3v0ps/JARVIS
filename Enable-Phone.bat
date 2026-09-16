@echo off
REM Lets your phone reach JARVIS from anywhere, over Tailscale.
REM Double-click this file - no terminal needed.
cd /d "%~dp0"
title J.A.R.V.I.S. - Enabling the phone

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo   JARVIS is not installed yet.
    echo   Double-click Install-JARVIS.exe first.
    echo.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" scripts\enable_phone.py
echo.
pause
