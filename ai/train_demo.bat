@echo off
echo ========================================
echo   ML MODEL TRAINING - DEMO DATA
echo ========================================
echo.
echo Starting training with demo data...
echo This will create realistic synthetic data and train models.
echo.

python simple_train.py

echo.
echo ========================================
echo Training complete!
echo.
echo Models saved in: %CD%
echo - tds_model.pkl
echo - turbidity_model.pkl
echo.
echo Next steps:
echo 1. View predictions at: http://localhost/sanitary/sensor/predictive_dashboard.php
echo 2. Or access the main dashboard
echo ========================================
echo.
pause

