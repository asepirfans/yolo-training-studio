@echo off
:: ==============================================================================
:: YOLO Training Studio - Windows Service Uninstaller (NSSM)
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

if exist "%CURRENT_DIR%\tools\nssm.exe" (
    set NSSM="%CURRENT_DIR%\tools\nssm.exe"
) else if exist "%CURRENT_DIR%\..\tools\nssm.exe" (
    set NSSM="%CURRENT_DIR%\..\tools\nssm.exe"
) else (
    set NSSM=nssm
)

echo =====================================================================
echo  Menghapus Windows Service: %SERVICE_NAME%
echo =====================================================================

%NSSM% stop %SERVICE_NAME%
%NSSM% remove %SERVICE_NAME% confirm

echo.
echo Service %SERVICE_NAME% telah berhasil dihentikan dan dihapus dari sistem.
echo.
pause
