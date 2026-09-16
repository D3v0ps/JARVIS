@echo off
REM Brings this JARVIS up to date with the published version.
REM Double-click this file - no terminal needed.
cd /d "%~dp0"
title J.A.R.V.I.S. - Update

set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

"%PY%" scripts\update_jarvis.py %*
echo.
pause
