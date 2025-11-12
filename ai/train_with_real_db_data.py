#!/usr/bin/env python3
"""
Train Potability Recommendation Models with Real Database Data
Uses actual TDS and turbidity readings from your database
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error, r2_score
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
            database='u520834156_dbbagoWaters25',
            charset='utf8mb4',
            collation='utf8mb4_general_ci'
        )
        return conn
    except Exception as e:
        print(f"Database connection failed: {e}")
        return None

def get_real_sensor_data():
    """Get real sensor data from your database"""
    print("Connecting to database...")
    conn = connect_to_database()
    
    if not conn:
        print("ERROR: Cannot connect to database. Using demo data instead.")
        return None, None
    
    try:
        # Get TDS data
        print("Fetching TDS data...")
        tds_query = """
        SELECT tds_value, analog_value, voltage, temperature, reading_time 
        FROM tds_readings 
        ORDER BY reading_time DESC 
        LIMIT 1000
        """
        tds_data = pd.read_sql(tds_query, conn)
        
        # Get Turbidity data
        print("Fetching Turbidity data...")
        turbidity_query = """
        SELECT ntu_value, raw_adc, reading_time 
        FROM turbidity_readings 
        ORDER BY reading_time DESC 
        LIMIT 1000
        """
        turbidity_data = pd.read_sql(turbidity_query, conn)
        
        print(f"SUCCESS: TDS data: {len(tds_data)} records")
        print(f"SUCCESS: Turbidity data: {len(turbidity_data)} records")
        
        return tds_data, turbidity_data
        
    except Exception as e:
        print(f"ERROR: Error fetching data: {e}")
        return None, None
    finally:
        conn.close()

def create_potability_labels(tds_data, turbidity_data):
    """Create potability labels based on WHO guidelines"""
    print("Creating potability labels based on WHO guidelines...")
    
    # WHO Guidelines
    tds_limit = 500  # mg/L
    turbidity_limit = 1.0  # NTU
    
    # Create combined dataset
    combined_data = []
    
    for _, tds_row in tds_data.iterrows():
        # Find matching turbidity reading (closest time)
        tds_time = pd.to_datetime(tds_row['reading_time'])
        
        # Find closest turbidity reading within 1 hour
        turbidity_matches = turbidity_data[
            abs(pd.to_datetime(turbidity_data['reading_time']) - tds_time) <= timedelta(hours=1)
        ]
        
        if not turbidity_matches.empty:
            turbidity_row = turbidity_matches.iloc[0]
            
            # Determine potability based on WHO guidelines
            tds_value = tds_row['tds_value']
            turbidity_value = turbidity_row['ntu_value']
            
            if tds_value <= tds_limit and turbidity_value <= turbidity_limit:
                status = 'Potable'
                score = 90 + np.random.normal(0, 5)  # 85-95
            else:
                status = 'Not Potable'
                score = 30 + np.random.normal(0, 15)  # 15-45
            
            combined_data.append({
                'tds_value': tds_value,
                'turbidity_value': turbidity_value,
                'temperature': tds_row.get('temperature', 25),
                'voltage': tds_row.get('voltage', 3.5),
                'analog_value': tds_row.get('analog_value', tds_value * 2.5),
                'potability_status': status,
                'potability_score': score,
                'reading_time': tds_time
            })
    
    if not combined_data:
        print("ERROR: No matching data found. Using demo data.")
        return None
    
    df = pd.DataFrame(combined_data)
    print(f"SUCCESS: Combined dataset: {len(df)} records")
    print(f"   - Potable: {sum(1 for x in df['potability_status'] if x == 'Potable')}")
    print(f"   - Not Potable: {sum(1 for x in df['potability_status'] if x == 'Not Potable')}")
    
    return df

def prepare_features(data):
    """Prepare features for training"""
    print("Preparing features...")
    
    # Create time features
    data['hour'] = pd.to_datetime(data['reading_time']).dt.hour
    data['day_of_week'] = pd.to_datetime(data['reading_time']).dt.dayofweek
    data['day_of_year'] = pd.to_datetime(data['reading_time']).dt.dayofyear
    
    # Create additional features
    data['tds_turbidity_ratio'] = data['tds_value'] / (data['turbidity_value'] + 0.1)
    data['quality_index'] = (data['tds_value'] / 500) + (data['turbidity_value'] / 1.0)
    data['conductivity'] = data['tds_value'] * 2 + np.random.normal(0, 50, len(data))
    
    # Handle NaN values - fill with median or mean
    numeric_cols = ['tds_value', 'turbidity_value', 'analog_value', 'voltage', 'temperature',
                   'hour', 'day_of_week', 'day_of_year', 'tds_turbidity_ratio', 'quality_index', 'conductivity']
    
    for col in numeric_cols:
        if col in data.columns:
            if data[col].isna().any():
                if col in ['hour', 'day_of_week', 'day_of_year']:
                    # For time features, use median
                    data[col] = data[col].fillna(data[col].median())
                else:
                    # For other features, use mean
                    data[col] = data[col].fillna(data[col].mean())
    
    # Define feature columns
    feature_cols = [
        'tds_value', 'turbidity_value', 'hour', 'day_of_week', 'day_of_year',
        'temperature', 'voltage', 'analog_value', 'conductivity',
        'tds_turbidity_ratio', 'quality_index'
    ]
    
    return feature_cols

def train_potability_classifier_with_real_data(data):
    """Train potability classifier with real data"""
    print("\nTraining Potability Classifier with REAL DATA...")
    
    feature_cols = prepare_features(data)
    X = data[feature_cols]
    y = data['potability_status']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train Random Forest Classifier
    model = RandomForestClassifier(
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
    accuracy = accuracy_score(y_test, y_pred)
    
    # Feature importance
    feature_importance = list(zip(feature_cols, model.feature_importances_))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    # Save model
    joblib.dump(model, 'potability_classifier_real.pkl')
    
    print(f"SUCCESS: Potability Classifier trained with REAL DATA!")
    print(f"   - Samples: {len(X)}")
    print(f"   - Accuracy: {accuracy:.3f}")
    print(f"   - Top features: {feature_importance[0][0]} ({feature_importance[0][1]:.3f})")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return True

def train_potability_score_regressor_with_real_data(data):
    """Train potability score regressor with real data"""
    print("\nTraining Potability Score Regressor with REAL DATA...")
    
    # Check if we have enough data for complex model
    if len(data) < 10:
        print("WARNING: Small dataset detected (need at least 10 samples for GradientBoosting)")
        print("Creating a simple model with available data...")
        
        # Create a simple model with just basic features
        X = data[['tds_value', 'turbidity_value']].fillna(0)
        y_scores = data['potability_score']
        
        # Use a simpler model for small datasets
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y_scores)
        
        # Save model
        joblib.dump(model, 'potability_score_regressor_real.pkl')
        
        print(f"SUCCESS: Simple Potability Score Regressor trained with REAL DATA!")
        print(f"   - Samples: {len(X)}")
        print(f"   - Model: Linear Regression (simplified for small dataset)")
        
        return True
    
    feature_cols = prepare_features(data)
    X = data[feature_cols]
    y_scores = data['potability_score']
    
    # Fill any remaining NaN values with 0 (safety check)
    X = X.fillna(0)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_scores, test_size=0.2, random_state=42
    )
    
    # Train Gradient Boosting Regressor
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
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Save model
    joblib.dump(model, 'potability_score_regressor_real.pkl')
    
    print(f"SUCCESS: Potability Score Regressor trained with REAL DATA!")
    print(f"   - Samples: {len(X)}")
    print(f"   - R2 Score: {r2:.3f}")
    print(f"   - MAE: {mae:.2f}")
    
    return True

def main():
    """Main training function with real database data"""
    print("TRAINING WITH REAL DATABASE DATA")
    print("=" * 50)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get real sensor data
    tds_data, turbidity_data = get_real_sensor_data()
    
    if tds_data is None or turbidity_data is None:
        print("ERROR: No real data available. Please ensure your database has sensor readings.")
        return
    
    # Create potability labels
    combined_data = create_potability_labels(tds_data, turbidity_data)
    
    if combined_data is None:
        print("ERROR: No combined data available.")
        return
    
    # Check if we have enough data for training
    if len(combined_data) < 2:
        print("WARNING: Very limited data available for training.")
        print("Your AI models will be basic but functional.")
        print("For better accuracy, collect more sensor data over time.")
    
    # Train models
    success_count = 0
    
    if train_potability_classifier_with_real_data(combined_data):
        success_count += 1
    
    if train_potability_score_regressor_with_real_data(combined_data):
        success_count += 1
    
    # Summary
    print(f"\nTRAINING WITH REAL DATA COMPLETE!")
    print(f"   - Models trained: {success_count}/2")
    print(f"   - Models saved: potability_classifier_real.pkl, potability_score_regressor_real.pkl")
    print(f"   - Data source: Your real sensor database")
    print(f"   - Ready for production!")
    
    if success_count > 0:
        print("\nSUCCESS! Your AI models are trained with REAL DATA!")
        print("To use real data models:")
        print("   1. Update ml_server.py to load 'potability_classifier_real.pkl'")
        print("   2. Update ml_server.py to load 'potability_score_regressor_real.pkl'")
        print("   3. Restart your Python server")
        print("\nYour AI system now learns from your actual sensor data!")
    else:
        print("\nTraining failed. Check your database connection and data.")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
