@echo off
title Water Potability AI Server
color 0A

echo.
echo ========================================
echo   Water Potability AI Server Startup
echo ========================================
echo.

cd /d "%~dp0"

echo [INFO] Checking Python installation...
python --version
if errorlevel 1 (
    echo [ERROR] Python not found! Please install Python first.
    pause
    exit /b 1
)

echo [INFO] Installing required packages...
pip install flask pandas numpy scikit-learn joblib

echo.
echo [INFO] Starting Water Potability AI Server...
echo.

python ml_server.py

pause
