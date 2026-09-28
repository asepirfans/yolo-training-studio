@echo off
:: ==============================================================================
:: YOLO Training Studio - Windows Service Installer (NSSM)
:: ==============================================================================

:: Cek Administrator Privilege
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [INFO] Memerlukan hak akses Administrator...
    powershell -Command "Start-Process cmd -ArgumentList '/c \"\"%~dpnx0\"\"' -Verb RunAs"
    exit /b
)

set SERVICE_NAME=YoloTrainingStudio
set CURRENT_DIR=%~dp0
set CURRENT_DIR=%CURRENT_DIR:~0,-1%
set LOG_DIR="%CURRENT_DIR%\logs"

:: Deteksi NSSM (cek tools lokal, parent, atau sistem PATH)
if exist "%CURRENT_DIR%\tools\nssm.exe" (
    set NSSM="%CURRENT_DIR%\tools\nssm.exe"
) else if exist "%CURRENT_DIR%\..\tools\nssm.exe" (
    set NSSM="%CURRENT_DIR%\..\tools\nssm.exe"
) else (
    set NSSM=nssm
)

:: Deteksi Python Executable
if exist "%CURRENT_DIR%\venv\Scripts\python.exe" (
    set PYTHON="%CURRENT_DIR%\venv\Scripts\python.exe"
    set WORK_DIR="%CURRENT_DIR%"
    set SCRIPT_ARG="run_studio.py"
) else if exist "%CURRENT_DIR%\..\venv\Scripts\python.exe" (
    set PYTHON="%CURRENT_DIR%\..\venv\Scripts\python.exe"
    set WORK_DIR="%CURRENT_DIR%\.."
    set SCRIPT_ARG="yolo_training_studio\run_studio.py"
) else (
    set PYTHON="python"
    set WORK_DIR="%CURRENT_DIR%"
    set SCRIPT_ARG="run_studio.py"
)

echo =====================================================================
echo  Memasang Windows Service: %SERVICE_NAME%
echo  NSSM: %NSSM%
echo  App : %PYTHON% %SCRIPT_ARG%
echo =====================================================================

if not exist %LOG_DIR% mkdir %LOG_DIR%

:: Stop dan bersihkan jika service sudah terpasang sebelumnya
%NSSM% stop %SERVICE_NAME% >nul 2>&1
%NSSM% remove %SERVICE_NAME% confirm >nul 2>&1

:: Install Service ke Windows
%NSSM% install %SERVICE_NAME% %PYTHON% %SCRIPT_ARG%
%NSSM% set %SERVICE_NAME% AppDirectory %WORK_DIR%
%NSSM% set %SERVICE_NAME% DisplayName "YOLO Training Studio Service"
%NSSM% set %SERVICE_NAME% Description "Background Service YOLO Training Studio Web UI (Port 5055)"
%NSSM% set %SERVICE_NAME% Start SERVICE_AUTO_START

:: Konfigurasi Logging & Auto-Rotate
%NSSM% set %SERVICE_NAME% AppStdout "%LOG_DIR%\service.log"
%NSSM% set %SERVICE_NAME% AppStderr "%LOG_DIR%\service_error.log"
%NSSM% set %SERVICE_NAME% AppRotateFiles 1
%NSSM% set %SERVICE_NAME% AppRotateOnline 1
%NSSM% set %SERVICE_NAME% AppRotateSeconds 86400
%NSSM% set %SERVICE_NAME% AppRotateBytes 10485760

:: Konfigurasi Pemulihan / Auto-Restart
%NSSM% set %SERVICE_NAME% AppThrottle 1500
%NSSM% set %SERVICE_NAME% AppExit Default Restart

echo.
echo =====================================================================
echo  Menjalankan Windows Service: %SERVICE_NAME%...
echo =====================================================================
%NSSM% start %SERVICE_NAME%

timeout /t 2 >nul
%NSSM% status %SERVICE_NAME%

echo.
echo =====================================================================
echo  Service BERHASIL dipasang dan aktif otomatis saat Windows menyala!
echo  Akses Web Studio: http://localhost:5055
echo  Log File: %LOG_DIR%\service.log
echo =====================================================================
echo.
pause
