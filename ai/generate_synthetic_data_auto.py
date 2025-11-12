#!/usr/bin/env python3
"""
Generate Synthetic Realistic Sensor Data for AI Training (Non-Interactive Version)
Creates realistic TDS and Turbidity readings based on real-world patterns
"""

import mysql.connector
import random
import numpy as np
from datetime import datetime, timedelta
import time

def connect_to_database():
    """Connect to your MySQL database"""
    try:
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            database='u520834156_DBBagoWaters251',
            charset='utf8mb4',
            collation='utf8mb4_general_ci'
        )
        return conn
    except Exception as e:
        print(f"Database connection failed: {e}")
        return None

def generate_realistic_tds_data(num_records=100):
    """Generate realistic TDS sensor data"""
    print(f"Generating {num_records} realistic TDS readings...")
    
    tds_data = []
    base_time = datetime.now() - timedelta(days=30)  # Start 30 days ago
    
    for i in range(num_records):
        # Create realistic TDS patterns
        # Most readings should be in safe ranges (0-500 ppm)
        # Some readings in concerning ranges (500-1200 ppm)
        # Few readings in dangerous ranges (>1200 ppm)
        
        # Weighted random selection for realistic distribution
        rand = random.random()
        
        if rand < 0.6:  # 60% - Safe range (0-500 ppm)
            tds_value = random.uniform(50, 450)
            voltage = random.uniform(0.1, 0.7)
        elif rand < 0.85:  # 25% - Moderate range (500-800 ppm)
            tds_value = random.uniform(500, 800)
            voltage = random.uniform(0.7, 1.2)
        elif rand < 0.95:  # 10% - Concerning range (800-1200 ppm)
            tds_value = random.uniform(800, 1200)
            voltage = random.uniform(1.2, 1.8)
        else:  # 5% - Dangerous range (>1200 ppm)
            tds_value = random.uniform(1200, 2000)
            voltage = random.uniform(1.8, 3.0)
        
        # Add some realistic noise and patterns
        # TDS tends to be higher during certain times
        hour = base_time.hour
        if 6 <= hour <= 10:  # Morning - slightly higher (more activity)
            tds_value *= random.uniform(1.0, 1.1)
        elif 18 <= hour <= 22:  # Evening - slightly higher (more usage)
            tds_value *= random.uniform(1.0, 1.1)
        elif 0 <= hour <= 5:  # Night - slightly lower (less activity)
            tds_value *= random.uniform(0.9, 1.0)
        
        # Add seasonal variation (if applicable)
        day_of_year = base_time.timetuple().tm_yday
        seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * day_of_year / 365)
        tds_value *= seasonal_factor
        
        # Calculate analog value (12-bit ADC)
        analog_value = int((voltage / 3.3) * 4095)
        
        # Add some realistic temperature variation
        temperature = random.uniform(20, 30)  # Room temperature range
        
        tds_data.append({
            'tds_value': round(tds_value, 2),
            'analog_value': analog_value,
            'voltage': round(voltage, 3),
            'temperature': round(temperature, 1),
            'reading_time': base_time.strftime('%Y-%m-%d %H:%M:%S')
        })
        
        # Increment time (random intervals between 30 seconds to 2 hours)
        interval_minutes = random.randint(1, 120)
        base_time += timedelta(minutes=interval_minutes)
    
    return tds_data

def generate_realistic_turbidity_data(num_records=100):
    """Generate realistic Turbidity sensor data"""
    print(f"Generating {num_records} realistic Turbidity readings...")
    
    turbidity_data = []
    base_time = datetime.now() - timedelta(days=30)  # Start 30 days ago
    
    for i in range(num_records):
        # Create realistic Turbidity patterns
        # Most readings should be in excellent range (0-1 NTU)
        # Some readings in acceptable range (1-5 NTU)
        # Few readings in concerning range (>5 NTU)
        
        # Weighted random selection for realistic distribution
        rand = random.random()
        
        if rand < 0.7:  # 70% - Excellent range (0-1 NTU)
            ntu_value = random.uniform(0.1, 0.9)
            sensor_voltage = random.uniform(0.05, 0.45)
        elif rand < 0.9:  # 20% - Acceptable range (1-5 NTU)
            ntu_value = random.uniform(1.0, 4.5)
            sensor_voltage = random.uniform(0.5, 2.0)
        else:  # 10% - Concerning range (>5 NTU)
            ntu_value = random.uniform(5.0, 25.0)
            sensor_voltage = random.uniform(2.0, 4.0)
        
        # Add realistic patterns
        # Turbidity can vary with time of day
        hour = base_time.hour
        if 6 <= hour <= 9:  # Morning - slightly higher (stirring)
            ntu_value *= random.uniform(1.0, 1.2)
        elif 12 <= hour <= 14:  # Midday - peak activity
            ntu_value *= random.uniform(1.1, 1.3)
        
        # Add weather-like variations (simulated)
        weather_factor = random.uniform(0.8, 1.3)
        ntu_value *= weather_factor
        
        # Calculate voltages
        vout_esp32 = sensor_voltage / 3.0  # Divider factor
        turbidity_analog = int((vout_esp32 / 3.3) * 4095)
        
        # Determine water quality based on NTU (enum values for database)
        if ntu_value <= 1.0:
            water_quality = "good"
        elif ntu_value <= 5.0:
            water_quality = "warning"
        else:
            water_quality = "bad"
        
        turbidity_data.append({
            'device_id': 'SENSOR_001',
            'ntu_value': round(ntu_value, 1),
            'analog_value': turbidity_analog,
            'voltage': round(vout_esp32, 3),
            'water_quality': water_quality,
            'reading_time': base_time.strftime('%Y-%m-%d %H:%M:%S')
        })
        
        # Increment time (random intervals between 30 seconds to 2 hours)
        interval_minutes = random.randint(1, 120)
        base_time += timedelta(minutes=interval_minutes)
    
    return turbidity_data

def insert_tds_data(conn, tds_data):
    """Insert TDS data into database"""
    print("Inserting TDS data into database...")
    
    cursor = conn.cursor()
    insert_query = """
    INSERT INTO tds_readings (tds_value, analog_value, voltage, temperature, reading_time)
    VALUES (%(tds_value)s, %(analog_value)s, %(voltage)s, %(temperature)s, %(reading_time)s)
    """
    
    try:
        cursor.executemany(insert_query, tds_data)
        conn.commit()
        print("SUCCESS: Successfully inserted " + str(len(tds_data)) + " TDS records")
    except Exception as e:
        print("ERROR: Error inserting TDS data: " + str(e))
        conn.rollback()
    finally:
        cursor.close()

def insert_turbidity_data(conn, turbidity_data):
    """Insert Turbidity data into database"""
    print("Inserting Turbidity data into database...")
    
    cursor = conn.cursor()
    insert_query = """
    INSERT INTO turbidity_readings (device_id, ntu_value, analog_value, voltage, water_quality, reading_time)
    VALUES (%(device_id)s, %(ntu_value)s, %(analog_value)s, %(voltage)s, %(water_quality)s, %(reading_time)s)
    """
    
    try:
        cursor.executemany(insert_query, turbidity_data)
        conn.commit()
        print("SUCCESS: Successfully inserted " + str(len(turbidity_data)) + " Turbidity records")
    except Exception as e:
        print("ERROR: Error inserting Turbidity data: " + str(e))
        conn.rollback()
    finally:
        cursor.close()

def clear_existing_data(conn):
    """Clear existing sensor data"""
    print("Clearing existing sensor data...")
    
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM tds_readings")
        cursor.execute("DELETE FROM turbidity_readings")
        conn.commit()
        print("SUCCESS: Existing data cleared")
    except Exception as e:
        print("ERROR: Error clearing data: " + str(e))
        conn.rollback()
    finally:
        cursor.close()

def main():
    """Main function to generate synthetic data"""
    print("=== SYNTHETIC REALISTIC SENSOR DATA GENERATOR ===")
    print("This will generate realistic TDS and Turbidity data for AI training")
    print()
    
    # Connect to database
    conn = connect_to_database()
    if not conn:
        print("ERROR: Cannot connect to database. Please check your connection.")
        return
    
    try:
        # Generate 100 records by default
        num_records = 100
        print(f"Generating {num_records} records of each type...")
        print()
        
        # Clear existing data
        clear_existing_data(conn)
        
        # Generate TDS data
        tds_data = generate_realistic_tds_data(num_records)
        insert_tds_data(conn, tds_data)
        
        # Generate Turbidity data
        turbidity_data = generate_realistic_turbidity_data(num_records)
        insert_turbidity_data(conn, turbidity_data)
        
        print()
        print("=== GENERATION COMPLETE ===")
        print("SUCCESS: Generated " + str(num_records) + " TDS records")
        print("SUCCESS: Generated " + str(num_records) + " Turbidity records")
        print()
        print("Your AI can now train with realistic data!")
        print("Run: python train_with_real_db_data.py")
        
    except Exception as e:
        print("ERROR: Error during generation: " + str(e))
    finally:
        conn.close()

if __name__ == "__main__":
    main()
