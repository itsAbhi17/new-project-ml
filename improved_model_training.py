import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump, load
import os
import warnings
warnings.filterwarnings('ignore')

# === IMPROVED DATA LOADING AND CLEANING ===
DATA_FILE_PATH = 'household_power_consumption.csv'

def load_and_clean_data(file_path):
    """Enhanced data loading with better error handling and cleaning."""
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(
            file_path, 
            sep=',',
            parse_dates={'Datetime': ['Date', 'Time']}, 
            low_memory=False, 
            na_values=['?', '', 'nan', 'NaN']
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    df = df.set_index('Datetime')
    
    # Rename columns
    df.columns = [
        'Global_active_power', 'Global_reactive_power', 'Voltage', 
        'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 
        'Sub_metering_3'
    ]

    # Convert to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remove extreme outliers (beyond 3 standard deviations)
    for col in df.columns:
        mean_val = df[col].mean()
        std_val = df[col].std()
        df[col] = df[col].clip(lower=mean_val - 3*std_val, upper=mean_val + 3*std_val)

    # Better missing value handling - forward fill then backward fill
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # If still missing, use interpolation
    df = df.interpolate(method='time')
    
    print(f"Data loaded and cleaned. Shape: {df.shape}")
    return df

def create_enhanced_features(df):
    """Create comprehensive time-based and lag features."""
    df_enhanced = df.copy()
    
    # Time-based features
    df_enhanced['Hour'] = df_enhanced.index.hour
    df_enhanced['DayOfWeek'] = df_enhanced.index.dayofweek
    df_enhanced['Month'] = df_enhanced.index.month
    df_enhanced['DayOfMonth'] = df_enhanced.index.day
    df_enhanced['Quarter'] = df_enhanced.index.quarter
    df_enhanced['IsWeekend'] = (df_enhanced.index.dayofweek >= 5).astype(int)
    
    # Cyclical encoding for time features
    df_enhanced['Hour_sin'] = np.sin(2 * np.pi * df_enhanced['Hour'] / 24)
    df_enhanced['Hour_cos'] = np.cos(2 * np.pi * df_enhanced['Hour'] / 24)
    df_enhanced['DayOfWeek_sin'] = np.sin(2 * np.pi * df_enhanced['DayOfWeek'] / 7)
    df_enhanced['DayOfWeek_cos'] = np.cos(2 * np.pi * df_enhanced['DayOfWeek'] / 7)
    df_enhanced['Month_sin'] = np.sin(2 * np.pi * df_enhanced['Month'] / 12)
    df_enhanced['Month_cos'] = np.cos(2 * np.pi * df_enhanced['Month'] / 12)
    
    # Lag features (previous values)
    target_col = 'Global_active_power'
    for lag in [1, 2, 3, 6, 12, 24]:  # 1min, 2min, 3min, 6min, 12min, 1hour ago
        df_enhanced[f'{target_col}_lag_{lag}'] = df_enhanced[target_col].shift(lag)
    
    # Rolling statistics
    for window in [5, 15, 60]:  # 5min, 15min, 1hour windows
        df_enhanced[f'{target_col}_rolling_mean_{window}'] = df_enhanced[target_col].rolling(window=window).mean()
        df_enhanced[f'{target_col}_rolling_std_{window}'] = df_enhanced[target_col].rolling(window=window).std()
    
    # Rate of change
    df_enhanced[f'{target_col}_diff_1'] = df_enhanced[target_col].diff(1)
    df_enhanced[f'{target_col}_diff_5'] = df_enhanced[target_col].diff(5)
    
    # Drop rows with NaN values created by lag and rolling features
    df_enhanced = df_enhanced.dropna()
    
    return df_enhanced

def preprocess_data_enhanced(df):
    """Enhanced preprocessing with better feature engineering."""
    
    # Create enhanced features
    df_enhanced = create_enhanced_features(df)
    
    # Define feature columns (exclude target and original time features)
    feature_cols = [col for col in df_enhanced.columns if col not in ['Global_active_power', 'Hour', 'DayOfWeek', 'Month']]
    target_col = 'Global_active_power'
    
    X = df_enhanced[feature_cols]
    y = df_enhanced[target_col]
    
    # Use RobustScaler instead of StandardScaler (better for outliers)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)
    
    # Clustering data (daily profiles)
    df_temp = df_enhanced.copy()
    df_temp['Date'] = df_temp.index.date
    
    daily_avg = df_temp.groupby(['Date', 'Hour'])['Global_active_power'].mean().unstack(level='Hour')
    daily_avg = daily_avg.fillna(daily_avg.mean())

    clus_scaler = RobustScaler()
    X_clus_scaled = clus_scaler.fit_transform(daily_avg)
    X_clus_scaled = pd.DataFrame(X_clus_scaled, index=daily_avg.index, columns=[f'Hour_{h}' for h in range(24)])
    
    return X_scaled, y, scaler, X_clus_scaled, daily_avg

def train_models_enhanced(X, y, X_clus, random_state=42):
    """Enhanced model training with time series validation."""
    print("Starting enhanced model training...")
    
    # Use TimeSeriesSplit for proper time series validation
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Get the last split for final training
    splits = list(tscv.split(X))
    train_idx, test_idx = splits[-1]
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    models = {}
    
    # 1. Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    models['LinearRegression'] = lr
    print("- Trained Linear Regression")

    # 2. Decision Tree with better parameters
    dt = DecisionTreeRegressor(
        max_depth=15, 
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=random_state
    )
    dt.fit(X_train, y_train)
    models['DecisionTree'] = dt
    print("- Trained Decision Tree Regressor")
    
    # 3. Random Forest with better parameters
    rf = RandomForestRegressor(
        n_estimators=100, 
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=random_state, 
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    models['RandomForest'] = rf
    print("- Trained Random Forest Regressor")

    # 4. K-Means Clustering
    kmeans = KMeans(n_clusters=3, random_state=random_state, n_init=10)
    kmeans.fit(X_clus)
    models['KMeans'] = kmeans
    print("- Trained K-Means Clustering")
    
    # Print model performance
    print("\nModel Performance on Test Set:")
    for name, model in models.items():
        if name != 'KMeans':
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            print(f"{name}: R² = {r2:.4f}, MSE = {mse:.4f}")
    
    return models, X_test, y_test

if __name__ == '__main__':
    ARTIFACT_FILE = 'ml_artifacts_improved.pkl'
    
    # Load and preprocess data
    data = load_and_clean_data(DATA_FILE_PATH)
    
    if data is None:
        exit()
        
    X, y, scaler, X_clus, daily_avg_data = preprocess_data_enhanced(data)
    
    # Train models
    models, X_test, y_test = train_models_enhanced(X, y, X_clus)
    
    # Prepare artifacts
    artifacts = {
        'models': models,
        'X_test': X_test,
        'y_test': y_test,
        'scaler': scaler,
        'daily_avg_data': daily_avg_data,
        'X_clus': X_clus,
        'original_data_head': data.head(),
        'feature_columns': X.columns.tolist()
    }
    
    # Save artifacts
    print(f"\nSaving improved ML artifacts to {ARTIFACT_FILE}...")
    try:
        dump(artifacts, ARTIFACT_FILE)
        print("Training complete and artifacts saved successfully!")
        print(f"\nFeatures used: {len(X.columns)}")
        print(f"Training samples: {len(X_test)}")
        
    except Exception as e:
        print(f"Error saving artifacts: {e}")