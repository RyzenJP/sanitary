#!/usr/bin/env python3
"""
Automatic AI Training Scheduler
Automatically retrains potability models when new data is available
"""

import schedule
import time
import subprocess
import os
import sys
import mysql.connector
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('auto_training.log'),
        logging.StreamHandler()
    ]
)

def connect_to_database():
    """Connect to your MySQL database"""
    try:
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            database='u520834156_dbbagoWaters25',
            charset='utf8mb4',
            collation='utf8mb4_general_ci'
        )
        return conn
    except Exception as e:
        logging.error(f"Database connection failed: {e}")
        return None

def check_new_data():
    """Check if there's new sensor data since last training"""
    conn = connect_to_database()
    if not conn:
        return False
    
    try:
        # Check for data in last 24 hours
        query = """
        SELECT COUNT(*) as new_records FROM (
            SELECT reading_time FROM tds_readings 
            WHERE reading_time >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
            UNION ALL
            SELECT reading_time FROM turbidity_readings 
            WHERE reading_time >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
        ) as combined_data
        """
        
        cursor = conn.cursor()
        cursor.execute(query)
        result = cursor.fetchone()
        new_records = result[0] if result else 0
        
        logging.info(f"Found {new_records} new sensor records in last 24 hours")
        
        # Retrain if we have at least 10 new records
        return new_records >= 10
        
    except Exception as e:
        logging.error(f"Error checking new data: {e}")
        return False
    finally:
        conn.close()

def run_training():
    """Run the training script"""
    try:
        logging.info("Starting automatic AI training...")
        
        # Change to AI directory
        ai_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(ai_dir)
        
        # Run training script
        result = subprocess.run([
            sys.executable, 'train_with_real_db_data.py'
        ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        if result.returncode == 0:
            logging.info("✅ AI training completed successfully!")
            logging.info(f"Training output: {result.stdout}")
            
            # Try to restart ML server if it's running
            restart_ml_server()
            
            return True
        else:
            logging.error(f"❌ Training failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logging.error("❌ Training timed out after 5 minutes")
        return False
    except Exception as e:
        logging.error(f"❌ Training error: {e}")
        return False

def restart_ml_server():
    """Restart ML server to load new models"""
    try:
        logging.info("Attempting to restart ML server...")
        
        # Kill existing ML server process
        if os.name == 'nt':  # Windows
            subprocess.run(['taskkill', '/f', '/im', 'python.exe'], 
                         capture_output=True)
        else:  # Linux/Mac
            subprocess.run(['pkill', '-f', 'ml_server.py'], 
                         capture_output=True)
        
        time.sleep(2)  # Wait for process to stop
        
        # Start new ML server in background
        ai_dir = os.path.dirname(os.path.abspath(__file__))
        if os.name == 'nt':  # Windows
            subprocess.Popen([sys.executable, 'ml_server.py'], 
                           cwd=ai_dir, creationflags=subprocess.CREATE_NEW_CONSOLE)
        else:  # Linux/Mac
            subprocess.Popen([sys.executable, 'ml_server.py'], 
                           cwd=ai_dir, start_new_session=True)
        
        logging.info("✅ ML server restart initiated")
        
    except Exception as e:
        logging.error(f"⚠️ Could not restart ML server: {e}")

def scheduled_training():
    """Scheduled training function"""
    logging.info("🔍 Checking for new data...")
    
    if check_new_data():
        logging.info("📊 New data found - starting training...")
        success = run_training()
        
        if success:
            logging.info("🎉 Automatic training completed successfully!")
        else:
            logging.error("❌ Automatic training failed!")
    else:
        logging.info("📊 No significant new data - skipping training")

def force_training():
    """Force training regardless of new data"""
    logging.info("🔄 Force training initiated...")
    success = run_training()
    
    if success:
        logging.info("🎉 Force training completed successfully!")
    else:
        logging.error("❌ Force training failed!")

def main():
    """Main scheduler function"""
    logging.info("🚀 Starting Automatic AI Training Scheduler...")
    logging.info("📅 Schedule: Every 6 hours + daily at 2 AM")
    
    # Schedule automatic training
    schedule.every(6).hours.do(scheduled_training)  # Every 6 hours
    schedule.every().day.at("02:00").do(scheduled_training)  # Daily at 2 AM
    
    # Initial training check
    logging.info("🔍 Running initial training check...")
    scheduled_training()
    
    # Keep scheduler running
    logging.info("⏰ Scheduler is now running... Press Ctrl+C to stop")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        logging.info("🛑 Scheduler stopped by user")
    except Exception as e:
        logging.error(f"❌ Scheduler error: {e}")

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--force":
            force_training()
        elif sys.argv[1] == "--check":
            if check_new_data():
                print("✅ New data available for training")
            else:
                print("📊 No new data for training")
        else:
            print("Usage: python auto_train_scheduler.py [--force|--check]")
    else:
        main()
