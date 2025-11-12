#!/usr/bin/env python3
"""
Windows-Compatible Training Script
No emoji characters to avoid Unicode errors
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def create_realistic_demo_data():
    """Create realistic demo data based on typical water quality patterns"""
    print("Creating training data...")
    
    # Generate 720 hours (30 days) of data
    dates = pd.date_range(start='2024-01-01', periods=720, freq='H')
    
    # TDS Data - Realistic patterns
    base_tds = 200  # Base TDS level
    daily_cycle = 20 * np.sin(2 * np.pi * np.arange(720) / 24)  # Daily variation
    weekly_cycle = 10 * np.sin(2 * np.pi * np.arange(720) / 168)  # Weekly variation
    noise = np.random.normal(0, 15, 720)  # Random noise
    
    tds_values = base_tds + daily_cycle + weekly_cycle + noise
    tds_values = np.clip(tds_values, 50, 500)  # Keep realistic range
    
    tds_data = pd.DataFrame({
        'tds_value': tds_values,
        'analog_value': tds_values * 2.5 + np.random.normal(0, 10, 720),
        'voltage': tds_values / 100 + np.random.normal(0, 0.1, 720),
        'reading_time': dates
    })
    
    # Turbidity Data - Realistic patterns
    base_ntu = 1.5  # Base turbidity
    daily_pattern = 0.5 * np.sin(2 * np.pi * np.arange(720) / 24)
    weekly_pattern = 0.3 * np.sin(2 * np.pi * np.arange(720) / 168)
    
    # Add some random spikes (rain events, etc.)
    spikes = np.zeros(720)
    spike_indices = np.random.choice(720, size=20, replace=False)
    spikes[spike_indices] = np.random.exponential(5, 20)
    
    noise_ntu = np.random.normal(0, 0.3, 720)
    ntu_values = base_ntu + daily_pattern + weekly_pattern + spikes + noise_ntu
    ntu_values = np.clip(ntu_values, 0.1, 50)  # Keep realistic range
    
    turbidity_data = pd.DataFrame({
        'ntu_value': ntu_values,
        'analog_value': ntu_values * 10 + np.random.normal(0, 5, 720),
        'voltage': ntu_values / 50 + np.random.normal(0, 0.05, 720),
        'raw_adc': (ntu_values * 10).astype(int),
        'reading_time': dates
    })
    
    print("Training data created successfully")
    print(f"   - TDS range: {tds_values.min():.1f} - {tds_values.max():.1f} ppm")
    print(f"   - Turbidity range: {ntu_values.min():.1f} - {ntu_values.max():.1f} NTU")
    
    return tds_data, turbidity_data

def prepare_features(data, target_col):
    """Prepare features for training"""
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

def train_tds_model(tds_data):
    """Train TDS prediction model"""
    print("\nTraining TDS Model...")
    
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
    joblib.dump(model, 'tds_model.pkl')
    
    print(f"TDS Model trained successfully!")
    print(f"   - Samples: {len(X)}")
    print(f"   - R2 Score: {r2:.3f}")
    print(f"   - MAE: {mae:.2f} ppm")
    print(f"   - Top features: {feature_importance[0][0]} ({feature_importance[0][1]:.3f})")
    
    return True

def train_turbidity_model(turbidity_data):
    """Train Turbidity prediction model"""
    print("\nTraining Turbidity Model...")
    
    X, y, feature_cols = prepare_features(turbidity_data, 'ntu_value')
    
    if len(X) < 50:
        print("Not enough Turbidity data for training")
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
    joblib.dump(model, 'turbidity_model.pkl')
    
    print(f"Turbidity Model trained successfully!")
    print(f"   - Samples: {len(X)}")
    print(f"   - R2 Score: {r2:.3f}")
    print(f"   - MAE: {mae:.2f} NTU")
    print(f"   - Top features: {feature_importance[0][0]} ({feature_importance[0][1]:.3f})")
    
    return True

def main():
    """Main training function"""
    print("WINDOWS-COMPATIBLE TRAINING")
    print("=" * 50)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create realistic demo data
    print("\nCreating training data...")
    tds_data, turbidity_data = create_realistic_demo_data()
    
    # Train models
    success_count = 0
    
    if train_tds_model(tds_data):
        success_count += 1
    
    if train_turbidity_model(turbidity_data):
        success_count += 1
    
    # Summary
    print(f"\nTRAINING COMPLETE!")
    print(f"   - Models trained: {success_count}/2")
    print(f"   - Models saved to: ai/")
    print(f"   - Ready for predictions!")
    
    if success_count > 0:
        print("\nSUCCESS! Your models are trained and ready!")
        print("Dashboard: http://localhost/sanitary1/sanitary/predictive_dashboard.php")
        print("\nYour models use realistic water quality patterns:")
        print("   - TDS: 50-500 ppm range with daily/weekly cycles")
        print("   - Turbidity: 0.1-50 NTU with occasional spikes")
        print("   - Features: Time patterns, lagged values, rolling averages")
    else:
        print("\nTraining failed. Check your data and try again.")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
