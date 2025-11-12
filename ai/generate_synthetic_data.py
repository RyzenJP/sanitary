#!/usr/bin/env python3
"""
Generate Synthetic Realistic Sensor Data for AI Training
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
        
        # Determine water quality based on NTU
        if ntu_value <= 1.0:
            water_quality = "Excellent"
        elif ntu_value <= 5.0:
            water_quality = "Acceptable"
        else:
            water_quality = "Concerning"
        
        turbidity_data.append({
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
        print(f"✅ Successfully inserted {len(tds_data)} TDS records")
    except Exception as e:
        print(f"❌ Error inserting TDS data: {e}")
        conn.rollback()
    finally:
        cursor.close()

def insert_turbidity_data(conn, turbidity_data):
    """Insert Turbidity data into database"""
    print("Inserting Turbidity data into database...")
    
    cursor = conn.cursor()
    insert_query = """
    INSERT INTO turbidity_readings (ntu_value, analog_value, voltage, water_quality, reading_time)
    VALUES (%(ntu_value)s, %(analog_value)s, %(voltage)s, %(water_quality)s, %(reading_time)s)
    """
    
    try:
        cursor.executemany(insert_query, turbidity_data)
        conn.commit()
        print(f"✅ Successfully inserted {len(turbidity_data)} Turbidity records")
    except Exception as e:
        print(f"❌ Error inserting Turbidity data: {e}")
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
        print("✅ Existing data cleared")
    except Exception as e:
        print(f"❌ Error clearing data: {e}")
        conn.rollback()
    finally:
        cursor.close()

def generate_water_quality_data(num_records=100):
    """Generate water quality test results"""
    print(f"Generating {num_records} water quality test results...")
    
    conn = connect_to_database()
    if not conn:
        return
    
    cursor = conn.cursor()
    
    # Get some client IDs
    cursor.execute("SELECT client_id FROM clients LIMIT 5")
    client_ids = [row[0] for row in cursor.fetchall()]
    
    if not client_ids:
        print("❌ No clients found. Please add clients first.")
        return
    
    water_quality_data = []
    base_time = datetime.now() - timedelta(days=30)
    
    for i in range(num_records):
        # Generate realistic water quality parameters
        tds_value = random.uniform(50, 1500)
        turbidity_value = random.uniform(0.1, 30.0)
        ph_value = random.uniform(6.0, 8.5)
        temperature = random.uniform(20, 30)
        
        # Determine status based on realistic criteria
        if tds_value <= 500 and turbidity_value <= 1.0 and 6.5 <= ph_value <= 8.0:
            status = "Safe"
        elif tds_value <= 1000 and turbidity_value <= 5.0 and 6.0 <= ph_value <= 8.5:
            status = "Unsafe"  # Marginal
        else:
            status = "Unsafe"
        
        # Generate realistic remarks
        if status == "Safe":
            remarks = ["Water quality excellent", "No treatment needed", "Safe for consumption"]
        else:
            remarks = ["Requires treatment", "Consider filtration", "Not recommended for drinking"]
        
        water_quality_data.append({
            'client_id': random.choice(client_ids),
            'location': f"Location {random.randint(1, 10)}",
            'tds_value': round(tds_value, 2),
            'turbidity_value': round(turbidity_value, 1),
            'ph_value': round(ph_value, 1),
            'temperature': round(temperature, 1),
            'status': status,
            'remarks': random.choice(remarks),
            'tested_by': 'Synthetic Data Generator',
            'test_date': base_time.strftime('%Y-%m-%d %H:%M:%S')
        })
        
        # Increment time
        interval_hours = random.randint(1, 48)
        base_time += timedelta(hours=interval_hours)
    
    # Insert water quality data
    insert_query = """
    INSERT INTO water_test_results (client_id, location, tds_value, turbidity_value, ph_value, temperature, status, remarks, tested_by, test_date)
    VALUES (%(client_id)s, %(location)s, %(tds_value)s, %(turbidity_value)s, %(ph_value)s, %(temperature)s, %(status)s, %(remarks)s, %(tested_by)s, %(test_date)s)
    """
    
    try:
        cursor.executemany(insert_query, water_quality_data)
        conn.commit()
        print(f"✅ Successfully inserted {len(water_quality_data)} water quality records")
    except Exception as e:
        print(f"❌ Error inserting water quality data: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

def main():
    """Main function to generate synthetic data"""
    print("=== SYNTHETIC REALISTIC SENSOR DATA GENERATOR ===")
    print("This will generate realistic TDS and Turbidity data for AI training")
    print()
    
    # Ask user for confirmation
    response = input("Do you want to clear existing data and generate new synthetic data? (y/n): ")
    if response.lower() != 'y':
        print("Operation cancelled.")
        return
    
    # Connect to database
    conn = connect_to_database()
    if not conn:
        print("❌ Cannot connect to database. Please check your connection.")
        return
    
    try:
        # Ask for number of records
        num_records = input("How many records to generate? (default: 100): ")
        try:
            num_records = int(num_records) if num_records else 100
        except ValueError:
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
        
        # Generate water quality test results
        generate_water_quality_data(num_records)
        
        print()
        print("=== GENERATION COMPLETE ===")
        print(f"✅ Generated {num_records} TDS records")
        print(f"✅ Generated {num_records} Turbidity records")
        print(f"✅ Generated {num_records} Water quality test results")
        print()
        print("Your AI can now train with realistic data!")
        print("Run: python train_with_real_db_data.py")
        
    except Exception as e:
        print(f"❌ Error during generation: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
