@echo off
echo ========================================
echo   Automatic AI Training Scheduler
echo ========================================
echo.
echo Starting automatic training system...
echo This will:
echo - Check for new sensor data every 6 hours
echo - Retrain AI models automatically
echo - Restart ML server with new models
echo.
echo Press Ctrl+C to stop the scheduler
echo.

cd /d "%~dp0"
python auto_train_scheduler.py

pause
