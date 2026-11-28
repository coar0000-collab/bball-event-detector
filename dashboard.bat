@echo off
rem Double-click: opens the analysis dashboard in your browser. Add --lan to view from phone on same Wi-Fi.
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
python -m gtt dashboard %*
pause
