#!/bin/bash
echo "============================================================"
echo "🚀 STARTING WATER QUALITY ML PREDICTION SERVER"
echo "============================================================"
echo ""
echo "Installing required packages..."
pip install flask flask-cors pandas numpy scikit-learn mysql-connector-python joblib --quiet
echo ""
echo "============================================================"
echo "📊 Starting ML Server on http://localhost:5000"
echo "============================================================"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
cd "$(dirname "$0")"
python3 ml_server.py

