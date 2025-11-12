#!/usr/bin/env python3
"""
Train ML Models with Real Sensor Data
Uses actual TDS and turbidity readings from your database
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib
from datetime import datetime, timedelta
import warnings
import mysql.connector
import os
warnings.filterwarnings('ignore')

def connect_to_database():
    """Connect to your MySQL database"""
    try:
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            database='u520834156_DBBagoWaters25'
        )
        return conn
    except Exception as e:
        print(f"Database connection failed: {e}")
        return None

def get_tds_data():
    """Get TDS readings from database"""
    print("Fetching TDS data from database...")
    
    conn = connect_to_database()
    if not conn:
        return None
    
    try:
        # Get TDS readings from last 30 days
        query = """
        SELECT tds_value, analog_value, voltage, reading_time 
        FROM tds_readings 
        WHERE tds_value > 0 
        AND reading_time >= DATE_SUB(NOW(), INTERVAL 30 DAY)
        ORDER BY reading_time ASC
        """
        
        df = pd.read_sql(query, conn)
        conn.close()
        
        if len(df) < 50:
            print(f"Not enough TDS data: {len(df)} records found")
            return None
        
        print(f"TDS data loaded: {len(df)} records")
        print(f"Date range: {df['reading_time'].min()} to {df['reading_time'].max()}")
        print(f"TDS range: {df['tds_value'].min():.1f} - {df['tds_value'].max():.1f} ppm")
        
        return df
        
    except Exception as e:
        print(f"Error fetching TDS data: {e}")
        conn.close()
        return None

def get_turbidity_data():
    """Get turbidity readings from database"""
    print("Fetching turbidity data from database...")
    
    conn = connect_to_database()
    if not conn:
        return None
    
    try:
        # Get turbidity readings from last 30 days
        query = """
        SELECT ntu_value, analog_value, voltage, reading_time 
        FROM turbidity_readings 
        WHERE ntu_value > 0 
        AND reading_time >= DATE_SUB(NOW(), INTERVAL 30 DAY)
        ORDER BY reading_time ASC
        """
        
        df = pd.read_sql(query, conn)
        conn.close()
        
        if len(df) < 50:
            print(f"Not enough turbidity data: {len(df)} records found")
            return None
        
        print(f"Turbidity data loaded: {len(df)} records")
        print(f"Date range: {df['reading_time'].min()} to {df['reading_time'].max()}")
        print(f"Turbidity range: {df['ntu_value'].min():.1f} - {df['ntu_value'].max():.1f} NTU")
        
        return df
        
    except Exception as e:
        print(f"Error fetching turbidity data: {e}")
        conn.close()
        return None

def prepare_features(data, target_col):
    """Prepare features for training with real data"""
    # Sort by time
    data = data.sort_values('reading_time').reset_index(drop=True)
    
    # Create time features
    data['hour'] = pd.to_datetime(data['reading_time']).dt.hour
    data['day_of_week'] = pd.to_datetime(data['reading_time']).dt.dayofweek
    data['day_of_year'] = pd.to_datetime(data['reading_time']).dt.dayofyear
    
    # Create lagged features
    data['lag_1'] = data[target_col].shift(1)
    data['lag_3'] = data[target_col].shift(3)
    data['lag_6'] = data[target_col].shift(6)
    data['lag_12'] = data[target_col].shift(12)
    
    # Create rolling features
    data['rolling_mean_3'] = data[target_col].rolling(window=3, min_periods=1).mean()
    data['rolling_mean_6'] = data[target_col].rolling(window=6, min_periods=1).mean()
    data['rolling_std_6'] = data[target_col].rolling(window=6, min_periods=1).std()
    
    # Create target (next value)
    data['target'] = data[target_col].shift(-1)
    
    # Remove NaN values
    data = data.dropna()
    
    # Define features
    feature_cols = ['hour', 'day_of_week', 'day_of_year', 'lag_1', 'lag_3', 'lag_6', 'lag_12',
                   'rolling_mean_3', 'rolling_mean_6', 'rolling_std_6']
    
    # Add sensor features if available
    if 'analog_value' in data.columns:
        feature_cols.append('analog_value')
    if 'voltage' in data.columns:
        feature_cols.append('voltage')
    
    X = data[feature_cols]
    y = data['target']
    
    return X, y, feature_cols

def train_tds_model_with_real_data(tds_data):
    """Train TDS model with real sensor data"""
    print("\nTraining TDS Model with Real Data...")
    
    X, y, feature_cols = prepare_features(tds_data, 'tds_value')
    
    if len(X) < 50:
        print("Not enough TDS data for training")
        return False
    
    # Split data (80% train, 20% test)
    split_point = int(len(X) * 0.8)
    X_train, X_test = X[:split_point], X[split_point:]
    y_train, y_test = y[:split_point], y[split_point:]
    
    # Train Random Forest model
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mae = np.mean(np.abs(y_test - y_pred))
    r2 = model.score(X_test, y_test)
    
    # Feature importance
    feature_importance = list(zip(feature_cols, model.feature_importances_))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    # Save model
    joblib.dump(model, 'tds_model_real.pkl')
    
    print(f"TDS Model trained successfully with REAL DATA!")
    print(f"   - Samples: {len(X)}")
    print(f"   - R2 Score: {r2:.3f}")
    print(f"   - MAE: {mae:.2f} ppm")
    print(f"   - Top features: {feature_importance[0][0]} ({feature_importance[0][1]:.3f})")
    
    return True

def train_turbidity_model_with_real_data(turbidity_data):
    """Train turbidity model with real sensor data"""
    print("\nTraining Turbidity Model with Real Data...")
    
    X, y, feature_cols = prepare_features(turbidity_data, 'ntu_value')
    
    if len(X) < 50:
        print("Not enough turbidity data for training")
        return False
    
    # Split data
    split_point = int(len(X) * 0.8)
    X_train, X_test = X[:split_point], X[split_point:]
    y_train, y_test = y[:split_point], y[split_point:]
    
    # Train Gradient Boosting model
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mae = np.mean(np.abs(y_test - y_pred))
    r2 = model.score(X_test, y_test)
    
    # Feature importance
    feature_importance = list(zip(feature_cols, model.feature_importances_))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    # Save model
    joblib.dump(model, 'turbidity_model_real.pkl')
    
    print(f"Turbidity Model trained successfully with REAL DATA!")
    print(f"   - Samples: {len(X)}")
    print(f"   - R2 Score: {r2:.3f}")
    print(f"   - MAE: {mae:.2f} NTU")
    print(f"   - Top features: {feature_importance[0][0]} ({feature_importance[0][1]:.3f})")
    
    return True

def main():
    """Main training function with real data"""
    print("TRAINING WITH REAL SENSOR DATA")
    print("=" * 50)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get real data from database
    tds_data = get_tds_data()
    turbidity_data = get_turbidity_data()
    
    if tds_data is None and turbidity_data is None:
        print("\nNo real sensor data available. Using demo data instead...")
        print("Make sure your TDS and turbidity devices are connected and sending data.")
        return False
    
    success_count = 0
    
    # Train TDS model if data available
    if tds_data is not None:
        if train_tds_model_with_real_data(tds_data):
            success_count += 1
    else:
        print("\nSkipping TDS model - no real data available")
    
    # Train turbidity model if data available
    if turbidity_data is not None:
        if train_turbidity_model_with_real_data(turbidity_data):
            success_count += 1
    else:
        print("\nSkipping turbidity model - no real data available")
    
    # Summary
    print(f"\nTRAINING COMPLETE!")
    print(f"   - Models trained: {success_count}/2")
    print(f"   - Models saved to: ai/")
    print(f"   - Ready for REAL predictions!")
    
    if success_count > 0:
        print("\nSUCCESS! Your models are trained with REAL sensor data!")
        print("Dashboard: http://localhost/sanitary1/sanitary/predictive_dashboard.php")
        print("\nYour models now use actual sensor readings:")
        if tds_data is not None:
            print(f"   - TDS: {len(tds_data)} real readings")
        if turbidity_data is not None:
            print(f"   - Turbidity: {len(turbidity_data)} real readings")
    else:
        print("\nTraining failed. Check your sensor data and try again.")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return success_count > 0

if __name__ == "__main__":
    main()
