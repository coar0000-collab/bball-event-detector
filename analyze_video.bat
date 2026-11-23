@echo off
rem Drag a game video onto this file (or double-click for the demo).
rem Needs configs\scorebug.json (+ digits.npz, calib*.json) prepared once per broadcast. See README.
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
if "%~1"=="" (
  python -m gtt analyze --demo --open
) else (
  python -m gtt analyze --video "%~1" --open
)
pause
