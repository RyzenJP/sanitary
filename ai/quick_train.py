#!/usr/bin/env python3
"""
SUPER SIMPLE TRAINING - Just run this!
python quick_train.py
"""

import subprocess
import sys
import os

def main():
    print("🤖 QUICK AI TRAINING")
    print("=" * 30)
    
    # Change to the correct directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    try:
        # Run the ML training
        result = subprocess.run([
            sys.executable, 
            'sanitary/ai/simple_train.py'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ TRAINING SUCCESSFUL!")
            print("\nYour AI models are now trained and ready!")
            print("🌐 Dashboard: http://localhost/sanitary/sanitary/predictive_dashboard.php")
        else:
            print("❌ Training failed")
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
