@echo off
echo ========================================
echo   Synthetic Realistic Data Generator
echo ========================================
echo.
echo This will generate realistic sensor data for AI training
echo.
echo Features:
echo - Realistic TDS patterns (0-2000 ppm)
echo - Realistic Turbidity patterns (0-25 NTU)
echo - Time-based variations (morning/evening peaks)
echo - Seasonal patterns
echo - Weather-like variations
echo - WHO-compliant distributions
echo.
echo Press any key to start...
pause

cd /d "%~dp0"
python generate_synthetic_data.py

echo.
echo Press any key to exit...
pause
