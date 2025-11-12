@echo off
echo Starting Water Potability AI Server...
echo =====================================

cd /d "%~dp0"

echo Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python not found! Please install Python first.
    pause
    exit /b 1
)

echo.
echo Installing required packages...
pip install flask pandas numpy scikit-learn joblib

echo.
echo Starting AI Server...
python ml_server.py

pause