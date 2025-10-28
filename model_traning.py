import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from joblib import dump, load
import os

# --- 1. Real Data Loading and Cleaning ---

# === IMPORTANT: SET YOUR DATA FILE PATH HERE ===
# Setting the default file name based on your error trace.
# Ensure your dataset file is in the same directory as this script.
DATA_FILE_PATH = 'household_power_consumption.csv' 

def load_and_clean_data(file_path):
    """
    Loads the power consumption data, combines Date and Time, 
    and handles missing values that are common in this dataset.
    """
    print(f"Attempting to load data from {file_path}...")
    try:
        # Load the data. We are changing the separator from ';' to ',' 
        # as a common fix, but you might need to adjust this again 
        # (e.g., to ' ' for space-separated files).
        df = pd.read_csv(
            file_path, 
            sep=',', # Changed from ';' to ',' to test a common alternative
            parse_dates={'Datetime': ['Date', 'Time']}, 
            low_memory=False, 
            na_values=['?'] # Identify '?' as NaN, a common issue in this dataset
        )
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        print("Please ensure your dataset is in the same directory and the DATA_FILE_PATH variable is correct.")
        return None
    except ValueError as e:
        # Catch the specific ValueError about missing columns and provide better guidance
        if "Missing column provided to 'parse_dates'" in str(e):
             print("\n--- DATA LOADING ERROR ---")
             print("The `Date` and `Time` columns could not be found.")
             print("Please check the `sep` parameter in pd.read_csv (it is currently set to ',').")
             print("Try changing it to ';' or other separators like ' ' (space) depending on your file's format.")
             return None
        raise e
    
    df = df.set_index('Datetime')
    
    # Rename columns based on the standard dataset structure
    df.columns = [
        'Global_active_power', 'Global_reactive_power', 'Voltage', 
        'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 
        'Sub_metering_3'
    ]

    # Convert all columns to numeric, coercing any non-numeric values to NaN
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Handle missing values by imputing with the column mean (a simple approach)
    df.fillna(df.mean(), inplace=True)
    
    print(f"Data loaded and cleaned. Total samples: {len(df)}.")
    return df
def perform_eda(df):
    """Performs basic Exploratory Data Analysis (EDA) on the cleaned data."""
    print("\n" + "="*50)
    print("=== EXPLORATORY DATA ANALYSIS (EDA) ===")
    print("="*50)
    
    # 1. Data Dimensions
    print(f"Shape of Data (Rows, Columns): {df.shape}")
    
    # 2. Data Types and Non-Null Count (after cleaning, should all be float/int)
    print("\n--- Data Types and First 5 Rows ---")
    print(df.info(verbose=False)) # Shows dtypes and non-null count summary
    print(df.head())
    
    # 3. Summary Statistics
    print("\n--- Summary Statistics (Min, Max, Mean, Std Dev) ---")
    print(df.describe().T[['count', 'mean', 'std', 'min', 'max']])
    
    # 4. Check for Imbalance/Distribution (simple check on target variable)
    # For time series, this is less about class imbalance but about value distribution
    target_var = 'Global_active_power'
    print(f"\n--- Target Variable ({target_var}) Quartiles ---")
    print(df[target_var].quantile([0.25, 0.5, 0.75]))
    
    # 5. Time Range Check
    print(f"\n--- Data Time Range ---")
    print(f"Start Date: {df.index.min()}")
    print(f"End Date: {df.index.max()}")
    print("="*50)

# --- 2. Data Preprocessing and Feature Engineering ---

def preprocess_data(df):
    """Prepares data for Regression and Clustering tasks."""
    
    # Derive time features from the Datetime index
    df['Hour'] = df.index.hour
    df['DayOfWeek'] = df.index.dayofweek
    
    # --- Regression Prep ---
    # Features for predicting Global_active_power
    REGRESSION_FEATURES = ['Hour', 'DayOfWeek', 'Voltage', 'Global_intensity', 
                           'Global_reactive_power', 'Sub_metering_1', 
                           'Sub_metering_2', 'Sub_metering_3']
    TARGET_VAR = 'Global_active_power'
    
    X_reg = df[REGRESSION_FEATURES]
    y_reg = df[TARGET_VAR]
    
    # Scale Regression Features
    reg_scaler = StandardScaler()
    X_reg_scaled = reg_scaler.fit_transform(X_reg)
    X_reg_scaled = pd.DataFrame(X_reg_scaled, columns=REGRESSION_FEATURES, index=X_reg.index)
    
    # --- Clustering Prep (Daily Profiles) ---
    # For Clustering, we analyze the *daily* average consumption profile
    # Pivot the data to get one row per day, with 24 columns for each hour's average power
    # Note: Using a copy to avoid SettingWithCopyWarning
    df_temp = df.copy() 
    df_temp['Date'] = df_temp.index.date
    
    daily_avg = df_temp.groupby(['Date', 'Hour'])['Global_active_power'].mean().unstack(level='Hour')
    daily_avg = daily_avg.fillna(daily_avg.mean()) # Handle missing hours/days for robustness (e.g., first day)

    clus_scaler = StandardScaler()
    X_clus_scaled = clus_scaler.fit_transform(daily_avg)
    X_clus_scaled = pd.DataFrame(X_clus_scaled, index=daily_avg.index, columns=[f'Hour_{h}' for h in range(24)])
    
    # Remove the temporary 'Hour' and 'DayOfWeek' columns from the main df if necessary for consistency, 
    # but they are kept in X_reg_scaled for training.
    
    return X_reg_scaled, y_reg, reg_scaler, X_clus_scaled, daily_avg

# --- 3. Model Training ---

def train_models(X_reg, y_reg, X_clus, random_state=42):
    """Trains Regression (Linear, DT, RF) and K-Means models."""
    print("Starting model training...")
    
    # Split data for Regression models
    X_train, X_test, y_train, y_test = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=random_state, shuffle=False # Time-series friendly split
    )
    
    models = {}
    
    # 1. Linear Regression (Base Model)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    models['LinearRegression'] = lr
    print("- Trained Linear Regression.")

    # 2. Decision Tree Regressor
    dt = DecisionTreeRegressor(max_depth=10, random_state=random_state)
    dt.fit(X_train, y_train)
    models['DecisionTree'] = dt
    print("- Trained Decision Tree Regressor.")
    
    # 3. Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=random_state, n_jobs=-1)
    rf.fit(X_train, y_train)
    models['RandomForest'] = rf
    print("- Trained Random Forest Regressor.")

    # 4. K-Means Clustering (Pattern Recognition)
    # Choosing K=3 clusters for typical patterns: Low, Medium, High usage days
    kmeans = KMeans(n_clusters=3, random_state=random_state, n_init=10)
    kmeans.fit(X_clus)
    models['KMeans'] = kmeans
    print("- Trained K-Means Clustering (K=3).")
    
    return models, X_test, y_test

# --- 4. Main Execution and Saving Artifacts ---

if __name__ == '__main__':
    # Define file path
    ARTIFACT_FILE = 'ml_artifacts.pkl'
    
    # 1. Load and Preprocess Data
    data = load_and_clean_data(DATA_FILE_PATH)
    
    if data is None:
        exit() # Stop execution if data loading failed
        
    X_reg, y_reg, reg_scaler, X_clus, daily_avg_data = preprocess_data(data)
    
    # 2. Train Models
    models, X_test, y_test = train_models(X_reg, y_reg, X_clus)
    
    # 3. Prepare Artifacts for Streamlit
    artifacts = {
        'models': models,
        'X_test': X_test,
        'y_test': y_test,
        'reg_scaler': reg_scaler,
        'daily_avg_data': daily_avg_data, # Unscaled daily profiles
        'X_clus': X_clus, # Scaled clustering features
        'original_data_head': data.head()
    }
    
    # 4. Save Artifacts
    print(f"\nSaving ML artifacts to {ARTIFACT_FILE}...")
    try:
        dump(artifacts, ARTIFACT_FILE)
        print("Training complete and artifacts saved successfully!")
        
        # Also print a summary of where the user can find the data for context
        print("\n--- Regression Task ---")
        print(f"X_test shape: {X_test.shape}")
        print("Regression Features used for prediction:")
        print(X_test.columns.tolist())
        print("\n--- Clustering Task ---")
        print(f"Daily Average Data shape: {daily_avg_data.shape}")
        print("Clustering attempts to group these daily consumption profiles into K=3 patterns.")
        
    except Exception as e:
        print(f"Error saving artifacts: {e}")