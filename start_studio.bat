@echo off
title YOLO Training Studio
echo ===================================================
echo Memulai YOLO Training Studio (Port 5055)
echo ===================================================
cd /d "%~dp0"
if exist "..\venv\Scripts\python.exe" (
    ..\venv\Scripts\python.exe run_studio.py
) else if exist "venv\Scripts\python.exe" (
    venv\Scripts\python.exe run_studio.py
) else (
    python run_studio.py
)
if not "%1"=="--no-pause" pause
