#!/usr/bin/env python3
"""
Test Real Sensor Data Flow
Simulates ESP32 sending data and checks the complete AI pipeline
"""

import requests
import time
import json

def test_sensor_data_flow():
    """Test the complete sensor-to-AI flow"""
    print("=== TESTING REAL SENSOR DATA FLOW ===")
    print()
    
    # Test data (simulating real ESP32 readings)
    test_cases = [
        {
            "name": "Good Water Quality",
            "data": {
                "tds_value": 245.67,
                "analog_value": 1205,
                "voltage": 0.969,
                "turbidity": 0.8,
                "turbidity_raw_adc": 1234,
                "turbidity_vout_esp32": 0.995,
                "turbidity_sensor_voltage": 2.985,
                "device_id": "SENSOR_001"
            }
        },
        {
            "name": "Moderate Water Quality", 
            "data": {
                "tds_value": 650.23,
                "analog_value": 2105,
                "voltage": 1.695,
                "turbidity": 3.2,
                "turbidity_raw_adc": 2234,
                "turbidity_vout_esp32": 1.995,
                "turbidity_sensor_voltage": 5.985,
                "device_id": "SENSOR_001"
            }
        },
        {
            "name": "Poor Water Quality",
            "data": {
                "tds_value": 1250.45,
                "analog_value": 3105,
                "voltage": 2.525,
                "turbidity": 8.7,
                "turbidity_raw_adc": 3234,
                "turbidity_vout_esp32": 2.995,
                "turbidity_sensor_voltage": 8.985,
                "device_id": "SENSOR_001"
            }
        }
    ]
    
    # Test each case
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test Case {i}: {test_case['name']}")
        print(f"Data: TDS={test_case['data']['tds_value']} ppm, Turbidity={test_case['data']['turbidity']} NTU")
        
        # Step 1: Send to TDS Receiver
        print("  Sending to TDS Receiver...")
        try:
            response = requests.post(
                'http://localhost/sanitary/main/sensor/device/tds/tds_receiver.php',
                data=test_case['data'],
                timeout=10
            )
            
            if response.status_code == 200:
                print("  SUCCESS: TDS Receiver accepted data")
                receiver_data = response.json()
                print(f"  Response: {receiver_data.get('message', 'N/A')}")
            else:
                print(f"  ERROR: TDS Receiver returned {response.status_code}")
                continue
                
        except Exception as e:
            print(f"  FAILED: TDS Receiver error - {e}")
            continue
        
        # Step 2: Test AI Prediction
        print("  Testing AI Prediction...")
        try:
            ai_response = requests.get(
                f"http://localhost:5000/predict?tds={test_case['data']['tds_value']}&turbidity={test_case['data']['turbidity']}&temperature=25&ph=7",
                timeout=5
            )
            
            if ai_response.status_code == 200:
                ai_data = ai_response.json()
                print("  SUCCESS: AI Prediction working")
                print(f"  Potability Score: {ai_data.get('potability_score', 'N/A')}%")
                print(f"  Recommendation: {ai_data.get('recommendation', 'N/A')}")
                print(f"  Status: {ai_data.get('status', 'N/A')}")
            else:
                print(f"  ERROR: AI Prediction returned {ai_response.status_code}")
                
        except Exception as e:
            print(f"  FAILED: AI Prediction error - {e}")
        
        print("  Waiting 2 seconds...")
        time.sleep(2)
        print()
    
    print("=== TEST COMPLETE ===")
    print()
    print("SUCCESS: If you see SUCCESS messages above, your automatic flow is working!")
    print("AUTOMATIC: Real ESP32 data will follow the same path automatically")
    print("DASHBOARDS: Check your dashboards to see the AI predictions")

if __name__ == "__main__":
    test_sensor_data_flow()
