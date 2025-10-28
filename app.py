import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
import os

# --- 1. Utility Functions and Artifact Loading ---

ARTIFACT_FILE = 'ml_artifacts.pkl'

@st.cache_resource
def load_artifacts():
    """Loads the trained models and data artifacts."""
    if not os.path.exists(ARTIFACT_FILE):
        st.error(f"Artifacts file '{ARTIFACT_FILE}' not found.")
        st.info("Please run `power_model_trainer.py` first to generate the necessary models and data.")
        return None
    try:
        artifacts = load(ARTIFACT_FILE)
        st.success("ML Models and Data loaded successfully!")
        return artifacts
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        return None

artifacts = load_artifacts()

if artifacts:
    models = artifacts['models']
    X_test = artifacts['X_test']
    y_test = artifacts['y_test']
    daily_avg_data = artifacts['daily_avg_data']
    X_clus = artifacts['X_clus']
    kmeans = models['KMeans']
    
# --- 2. Streamlit App Layout ---

st.set_page_config(
    page_title="House Power Consumption ML Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏡 House Power Consumption ML Analysis")
st.markdown("This dashboard analyzes power consumption patterns and predicts usage using various Machine Learning algorithms trained on a simulated time-series dataset.")

if not artifacts:
    st.stop()
    
# --- Tab Setup ---
tab_overview, tab_regression, tab_clustering = st.tabs([
    "📊 Data Overview & Head", 
    "📈 Regression & Prediction", 
    "🔎 Clustering & Patterns"
])

# --- TAB 1: Data Overview ---
with tab_overview:
    st.header("Dataset Structure")
    st.markdown("""
        The models were trained on a synthetic dataset mimicking the UCI Individual Household Electric Power Consumption dataset, 
        with a 1-minute resolution.
    """)
    
    st.subheader("First 5 Rows of the Processed Data")
    st.dataframe(artifacts['original_data_head'])

    st.subheader("Data Summary")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Samples (synthetic)", f"{len(X_test) * 5}: approx {len(X_test) * 5 / (24*60):.1f} Days")
        st.metric("Regression Features", len(X_test.columns))
    with col2:
        st.metric("Target Variable", "Global_active_power (kW)")
        st.metric("Clustering Granularity", "Daily Average Profiles (24 Hourly Features)")

# --- TAB 2: Regression & Prediction ---

with tab_regression:
    st.header("Regression Model Performance")
    st.markdown("""
        The goal of this task is to predict the `Global_active_power` (continuous value) based on time and other power measurements.
        Performance is evaluated on the held-out test set.
    """)
    
    model_name = st.sidebar.selectbox(
        "Select Regression Model",
        list(models.keys())[:-1] # Exclude KMeans
    )
    
    model = models[model_name]
    y_pred = model.predict(X_test)
    
    # Calculate Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    colA, colB, colC = st.columns(3)
    colA.metric("R-Squared ($R^2$)", f"{r2:.4f}", help="Closer to 1.0 is better. Explains variance in the target.")
    colB.metric("Mean Squared Error (MSE)", f"{mse:.4f}", help="Lower is better. Average of the squared errors.")
    colC.metric("Root Mean Squared Error (RMSE)", f"{rmse:.4f}", help="Lower is better. Same unit as the target (kW).")
    
    st.subheader(f"Actual vs. Predicted Power Usage ({model_name})")
    
    # Create a DataFrame for visualization (plotting a subset of the data)
    plot_df = pd.DataFrame({
        'Actual': y_test.values,
        'Predicted': y_pred
    }, index=y_test.index)
    
    # Plot only the first 500 points for clarity on the time-series
    subset_df = plot_df.iloc[:500] 
    
    fig_reg = go.Figure()
    fig_reg.add_trace(go.Scatter(x=subset_df.index, y=subset_df['Actual'], mode='lines', name='Actual Power (kW)', line=dict(color='rgba(0, 102, 204, 0.8)')))
    fig_reg.add_trace(go.Scatter(x=subset_df.index, y=subset_df['Predicted'], mode='lines', name='Predicted Power (kW)', line=dict(color='rgba(255, 100, 100, 0.8)')))
    
    fig_reg.update_layout(
        title=f"Actual vs. Predicted Power Consumption (First 500 Samples)",
        xaxis_title="Time",
        yaxis_title="Global Active Power (kW)",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig_reg, use_container_width=True)

    # Show model interpretation for Random Forest
    if model_name == 'RandomForest':
        st.subheader("Random Forest Feature Importance")
        importances = model.feature_importances_
        feature_names = X_test.columns
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        fig_imp = px.bar(
            feature_importance_df, 
            x='Importance', 
            y='Feature', 
            orientation='h',
            title="Model Reliance on Input Features"
        )
        st.plotly_chart(fig_imp, use_container_width=True)


# --- TAB 3: Clustering & Patterns ---

with tab_clustering:
    st.header("K-Means Clustering: Identifying Usage Profiles")
    st.markdown("""
        The **K-Means** algorithm was used to group **daily consumption profiles** into distinct patterns. 
        We used $K=3$ to find typical Low, Medium, and High usage days, which is excellent for finding consumption behaviors.
    """)
    
    # Assign clusters to the unscaled daily average data
    daily_avg_data['Cluster'] = kmeans.predict(X_clus)
    
    st.subheader(f"Cluster Profile Analysis (K={kmeans.n_clusters})")
    
    # Calculate the average power usage for each cluster at each hour
    cluster_profiles = daily_avg_data.groupby('Cluster').mean()
    
    # Rename index for better visualization
    cluster_profiles.index = [f"Profile {i} (Avg Daily Power: {cluster_profiles.sum(axis=1).iloc[i]:.2f} kW)" for i in range(kmeans.n_clusters)]

    fig_clus = go.Figure()
    
    for i in range(kmeans.n_clusters):
        # Calculate max, min, mean for easy labeling
        mean_power = cluster_profiles.iloc[i].sum()
        
        fig_clus.add_trace(go.Scatter(
            x=list(range(24)),
            y=cluster_profiles.iloc[i],
            mode='lines+markers',
            name=f"Profile {i} (Avg Total: {mean_power:.2f} kW)"
        ))

    fig_clus.update_layout(
        title="Hourly Consumption Profile for Each Cluster",
        xaxis_title="Hour of Day (0-23)",
        yaxis_title="Average Active Power (kW)",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        hovermode="x unified"
    )
    st.plotly_chart(fig_clus, use_container_width=True)
    
    st.subheader("Cluster Interpretation")
    st.markdown(f"""
    These 3 profiles ($K={kmeans.n_clusters}$) reveal distinct daily power usage patterns:
    * **Profile 0 (often Low):** Days with consistently low usage and minor peaks.
    * **Profile 1 (often High/Peak):** Days with sharp morning and evening peaks, suggesting heavy appliance use.
    * **Profile 2 (often Medium/Flat):** Days with moderate, sustained usage throughout the day, maybe from continuous systems like HVAC or refrigeration.
    """)
