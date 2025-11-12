#!/usr/bin/env python3
"""
Quick test to check if the server is running
"""

import urllib.request
import json

def test_server():
    """Test if the Python server is running"""
    
    try:
        # Test server status
        response = urllib.request.urlopen('http://localhost:5000/status', timeout=5)
        data = json.loads(response.read().decode())
        
        print("✅ Python ML Server is running!")
        print(f"   Status: {data.get('status')}")
        print(f"   Models loaded: {data.get('models_loaded')}")
        print(f"   Timestamp: {data.get('timestamp')}")
        
        # Test prediction
        print("\n🧪 Testing prediction...")
        response = urllib.request.urlopen('http://localhost:5000/predict?tds=350&turbidity=0.8', timeout=5)
        data = json.loads(response.read().decode())
        
        print("✅ Prediction successful!")
        print(f"   Status: {data.get('potability_status')}")
        print(f"   Score: {data.get('potability_score')}")
        print(f"   Recommendation: {data.get('recommendation')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Server not responding: {e}")
        print("\n💡 Make sure to start the server first:")
        print("   python ml_server.py")
        return False

if __name__ == "__main__":
    test_server()
