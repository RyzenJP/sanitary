#!/usr/bin/env python3
"""
Test script for the Python ML Server
"""

import requests
import json

def test_server():
    """Test the Python ML server endpoints"""
    
    base_url = "http://localhost:5000"
    
    print("🧪 Testing Water Potability AI Server...")
    print("=" * 50)
    
    # Test 1: Server status
    print("\n1. Testing server status...")
    try:
        response = requests.get(f"{base_url}/status")
        if response.status_code == 200:
            print("✅ Server is running")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Server error: {response.status_code}")
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return False
    
    # Test 2: Health check
    print("\n2. Testing health check...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")
    
    # Test 3: Prediction with safe water
    print("\n3. Testing prediction (safe water)...")
    try:
        response = requests.get(f"{base_url}/predict?tds=300&turbidity=0.5")
        if response.status_code == 200:
            data = response.json()
            print("✅ Prediction successful")
            print(f"   Status: {data.get('potability_status')}")
            print(f"   Score: {data.get('potability_score')}")
            print(f"   Recommendation: {data.get('recommendation')}")
        else:
            print(f"❌ Prediction failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Prediction error: {e}")
    
    # Test 4: Prediction with unsafe water
    print("\n4. Testing prediction (unsafe water)...")
    try:
        response = requests.get(f"{base_url}/predict?tds=800&turbidity=5.0")
        if response.status_code == 200:
            data = response.json()
            print("✅ Prediction successful")
            print(f"   Status: {data.get('potability_status')}")
            print(f"   Score: {data.get('potability_score')}")
            print(f"   Recommendation: {data.get('recommendation')}")
        else:
            print(f"❌ Prediction failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Prediction error: {e}")
    
    # Test 5: POST request
    print("\n5. Testing POST request...")
    try:
        data = {
            'tds_value': 600,
            'turbidity_value': 2.0,
            'temperature': 25,
            'ph_level': 7.0
        }
        response = requests.post(f"{base_url}/predict", json=data)
        if response.status_code == 200:
            result = response.json()
            print("✅ POST request successful")
            print(f"   Status: {result.get('potability_status')}")
            print(f"   Score: {result.get('potability_score')}")
        else:
            print(f"❌ POST request failed: {response.status_code}")
    except Exception as e:
        print(f"❌ POST request error: {e}")
    
    print("\n🎯 Test completed!")
    return True

if __name__ == "__main__":
    test_server()
