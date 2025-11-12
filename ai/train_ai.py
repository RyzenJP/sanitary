#!/usr/bin/env python3
"""
ONE-COMMAND AI TRAINING
Just run: python train_ai.py
"""

import os
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("🚀 STARTING AI TRAINING...")
    print("=" * 40)
    
    try:
        # Import and run the training
        from sanitary.ai.realtime_train import main as train_main
        train_main()
        
    except ImportError:
        print("❌ Error: Cannot import training module")
        print("Make sure you're running from the correct directory")
        print("Current directory:", os.getcwd())
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        print("Using demo models instead...")

if __name__ == "__main__":
    main()
