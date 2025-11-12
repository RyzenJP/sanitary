@echo off
echo ========================================
echo   ML MODEL TRAINING - REAL DATA
echo ========================================
echo.
echo Starting training with real sensor data...
echo This will use actual readings from your database.
echo.
echo Requirements:
echo - At least 50 TDS readings
echo - At least 50 Turbidity readings
echo - Data from last 30 days
echo.

python train_with_real_data.py

echo.
echo ========================================
echo Training complete!
echo.
echo Models saved in: %CD%
echo - tds_model_real.pkl
echo - turbidity_model_real.pkl
echo.
echo Next steps:
echo 1. View predictions at: http://localhost/sanitary/sensor/predictive_dashboard.php
echo 2. Or access the main dashboard
echo ========================================
echo.
pause

