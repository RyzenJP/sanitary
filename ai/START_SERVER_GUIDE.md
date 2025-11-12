# 🚀 How to Start Your Water Potability AI Server

## **Step 1: Train the Models (One-time setup)**
```bash
cd C:\xampp\htdocs\sanitary\sensor\ai
python train_potability_recommendation.py
```

## **Step 2: Start the Python Server**
```bash
cd C:\xampp\htdocs\sanitary\sensor\ai
python ml_server.py
```

## **Step 3: Test the Server**
Open another terminal and run:
```bash
cd C:\xampp\htdocs\sanitary\sensor\ai
python quick_test.py
```

## **Expected Output:**

### **When Training:**
```
WATER POTABILITY RECOMMENDATION TRAINING
==================================================
[SUCCESS] Models trained successfully!
```

### **When Starting Server:**
```
🚀 Starting Water Potability AI Server...
==================================================
[SUCCESS] Models loaded from disk
[SERVER] Water Potability AI Server started on port 5000
[INFO] Algorithm: Random Forest Classifier + Gradient Boosting Regressor
[INFO] Server running... Press Ctrl+C to stop
```

### **When Testing:**
```
✅ Python ML Server is running!
✅ Prediction successful!
   Status: Potable
   Score: 85.2
   Recommendation: Water is safe for drinking. No treatment needed.
```

## **🎯 Your Server Endpoints:**

- **Status**: `http://localhost:5000/status`
- **Predict**: `http://localhost:5000/predict?tds=350&turbidity=0.8`
- **Health**: `http://localhost:5000/health`
- **Test**: `http://localhost:5000/test`

## **📱 For Your Device:**

```python
import requests

# Your device code
response = requests.get("http://YOUR_IP:5000/predict?tds=350&turbidity=0.8")
recommendation = response.json()

print(f"Status: {recommendation['potability_status']}")
print(f"Score: {recommendation['potability_score']}")
print(f"Advice: {recommendation['recommendation']}")
```

## **🔧 Troubleshooting:**

### **If "Models not found":**
- Run training first: `python train_potability_recommendation.py`

### **If "Connection refused":**
- Make sure server is running: `python ml_server.py`
- Check if port 5000 is available

### **If "Module not found":**
- Install packages: `pip install flask pandas numpy scikit-learn joblib`

## **🎯 Perfect for Your Crush! 💕**

Your server will be running just like your main capstone project:
- Professional Python server
- REST API endpoints
- Real-time AI predictions
- Easy to deploy to school server

**Ready to impress! 😄**
