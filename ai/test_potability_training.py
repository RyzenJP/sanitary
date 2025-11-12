#!/usr/bin/env python3
"""
Quick test script to verify potability training works
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from train_potability_recommendation import main
    print("Testing potability training...")
    main()
    print("\n✅ Training completed successfully!")
except Exception as e:
    print(f"❌ Training failed: {e}")
    import traceback
    traceback.print_exc()
