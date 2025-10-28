import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
import os

# --- 1. Utility Functions and Artifact Loading ---

ARTIFACT_FILE = 'ml_artifacts_improved.pkl'

@st.cache_resource
def load_artifacts():
    """Loads the improved trained models and data artifacts."""
    if not os.path.exists(ARTIFACT_FILE):
        st.error(f"Improved artifacts file '{ARTIFACT_FILE}' not found.")
        st.info("Please run `improved_model_training.py` first to generate the improved models and data.")
        return None
    try:
        artifacts = load(ARTIFACT_FILE)
        st.success("Improved ML Models and Data loaded successfully!")
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
    feature_columns = artifacts.get('feature_columns', [])
    
# --- 2. Streamlit App Layout ---

st.set_page_config(
    page_title="Improved House Power Consumption ML Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏡 Improved House Power Consumption ML Analysis")
st.markdown("This improved dashboard analyzes power consumption patterns with enhanced feature engineering and better model performance.")

if not artifacts:
    st.stop()
    
# --- Tab Setup ---
tab_overview, tab_regression, tab_clustering, tab_diagnostics = st.tabs([
    "📊 Data Overview", 
    "📈 Improved Predictions", 
    "🔎 Clustering Analysis",
    "🔧 Model Diagnostics"
])

# --- TAB 1: Data Overview ---
with tab_overview:
    st.header("Enhanced Dataset Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Features", len(feature_columns))
        st.metric("Test Samples", len(X_test))
        st.metric("Time Span", f"{X_test.index[0].strftime('%Y-%m-%d')} to {X_test.index[-1].strftime('%Y-%m-%d')}")
    
    with col2:
        st.metric("Target Variable", "Global_active_power (kW)")
        st.metric("Sampling Rate", "1 minute")
        st.metric("Data Quality", "Enhanced with outlier removal")
    
    st.subheader("Feature Engineering Improvements")
    st.markdown("""
    **Enhanced Features Include:**
    - ✅ Cyclical time encoding (sin/cos transformations)
    - ✅ Lag features (1, 2, 3, 6, 12, 24 minutes)
    - ✅ Rolling statistics (5, 15, 60 minute windows)
    - ✅ Rate of change features
    - ✅ Weekend/weekday indicators
    - ✅ Seasonal features (month, quarter)
    """)
    
    if st.checkbox("Show Feature List"):
        st.write("**All Features Used:**")
        st.write(feature_columns)

# --- TAB 2: Improved Regression & Prediction ---

with tab_regression:
    st.header("Improved Model Performance")
    
    model_name = st.sidebar.selectbox(
        "Select Model",
        list(models.keys())[:-1]  # Exclude KMeans
    )
    
    model = models[model_name]
    y_pred = model.predict(X_test)
    
    # Calculate comprehensive metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R-Squared (R²)", f"{r2:.4f}", help="Closer to 1.0 is better")
    col2.metric("RMSE", f"{rmse:.4f}", help="Lower is better")
    col3.metric("MAE", f"{mae:.4f}", help="Mean Absolute Error")
    col4.metric("MAPE", f"{np.mean(np.abs((y_test - y_pred) / y_test)) * 100:.2f}%", help="Mean Absolute Percentage Error")
    
    # Prediction quality indicator
    if r2 > 0.8:
        st.success("🎯 Excellent prediction quality!")
    elif r2 > 0.6:
        st.warning("⚠️ Good prediction quality")
    else:
        st.error("❌ Poor prediction quality - consider model improvements")
    
    st.subheader(f"Actual vs. Predicted Power Usage ({model_name})")
    
    # Create prediction comparison plot
    plot_samples = st.slider("Number of samples to plot", 100, min(2000, len(y_test)), 500)
    
    plot_df = pd.DataFrame({
        'Actual': y_test.values[:plot_samples],
        'Predicted': y_pred[:plot_samples],
        'Time': y_test.index[:plot_samples]
    })
    
    fig_reg = go.Figure()
    fig_reg.add_trace(go.Scatter(
        x=plot_df['Time'], 
        y=plot_df['Actual'], 
        mode='lines', 
        name='Actual Power (kW)', 
        line=dict(color='blue', width=1)
    ))
    fig_reg.add_trace(go.Scatter(
        x=plot_df['Time'], 
        y=plot_df['Predicted'], 
        mode='lines', 
        name='Predicted Power (kW)', 
        line=dict(color='red', width=1)
    ))
    
    fig_reg.update_layout(
        title=f"Time Series Prediction Comparison ({plot_samples} samples)",
        xaxis_title="Time",
        yaxis_title="Global Active Power (kW)",
        hovermode="x unified",
        height=500
    )
    st.plotly_chart(fig_reg, use_container_width=True)
    
    # Residual analysis
    st.subheader("Residual Analysis")
    residuals = y_test.values[:plot_samples] - y_pred[:plot_samples]
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_residual = go.Figure()
        fig_residual.add_trace(go.Scatter(
            x=plot_df['Time'],
            y=residuals,
            mode='markers',
            name='Residuals',
            marker=dict(size=3, opacity=0.6)
        ))
        fig_residual.add_hline(y=0, line_dash="dash", line_color="red")
        fig_residual.update_layout(
            title="Residuals Over Time",
            xaxis_title="Time",
            yaxis_title="Residual (Actual - Predicted)"
        )
        st.plotly_chart(fig_residual, use_container_width=True)
    
    with col2:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=residuals,
            nbinsx=50,
            name='Residual Distribution'
        ))
        fig_hist.update_layout(
            title="Residual Distribution",
            xaxis_title="Residual Value",
            yaxis_title="Frequency"
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Feature importance for Random Forest
    if model_name == 'RandomForest':
        st.subheader("Feature Importance Analysis")
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False).head(20)
        
        fig_imp = px.bar(
            feature_importance_df, 
            x='Importance', 
            y='Feature', 
            orientation='h',
            title="Top 20 Most Important Features"
        )
        st.plotly_chart(fig_imp, use_container_width=True)

# --- TAB 3: Clustering Analysis ---

with tab_clustering:
    st.header("Daily Consumption Pattern Analysis")
    
    daily_avg_data['Cluster'] = kmeans.predict(X_clus)
    
    # Cluster profile analysis
    cluster_profiles = daily_avg_data.groupby('Cluster').mean()
    
    fig_clus = go.Figure()
    
    colors = ['blue', 'red', 'green']
    for i in range(kmeans.n_clusters):
        avg_daily_consumption = cluster_profiles.iloc[i].sum()
        fig_clus.add_trace(go.Scatter(
            x=list(range(24)),
            y=cluster_profiles.iloc[i],
            mode='lines+markers',
            name=f"Pattern {i} (Avg: {avg_daily_consumption:.2f} kW)",
            line=dict(color=colors[i], width=3)
        ))

    fig_clus.update_layout(
        title="Daily Consumption Patterns by Cluster",
        xaxis_title="Hour of Day (0-23)",
        yaxis_title="Average Active Power (kW)",
        height=500
    )
    st.plotly_chart(fig_clus, use_container_width=True)
    
    # Cluster distribution
    cluster_counts = daily_avg_data['Cluster'].value_counts().sort_index()
    
    col1, col2 = st.columns(2)
    with col1:
        fig_pie = px.pie(
            values=cluster_counts.values,
            names=[f"Pattern {i}" for i in cluster_counts.index],
            title="Distribution of Daily Patterns"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("Pattern Interpretation")
        for i in range(kmeans.n_clusters):
            count = cluster_counts[i]
            avg_consumption = cluster_profiles.iloc[i].sum()
            peak_hour = cluster_profiles.iloc[i].idxmax()
            st.write(f"**Pattern {i}:** {count} days")
            st.write(f"- Average daily consumption: {avg_consumption:.2f} kW")
            st.write(f"- Peak consumption at: {peak_hour}:00")
            st.write("---")

# --- TAB 4: Model Diagnostics ---

with tab_diagnostics:
    st.header("Model Diagnostics & Comparison")
    
    # Model comparison
    st.subheader("Model Performance Comparison")
    
    model_metrics = []
    for name, model in models.items():
        if name != 'KMeans':
            y_pred = model.predict(X_test)
            metrics = {
                'Model': name,
                'R²': r2_score(y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'MAE': mean_absolute_error(y_test, y_pred),
                'MAPE': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            }
            model_metrics.append(metrics)
    
    metrics_df = pd.DataFrame(model_metrics)
    st.dataframe(metrics_df.round(4))
    
    # Best model recommendation
    best_model = metrics_df.loc[metrics_df['R²'].idxmax(), 'Model']
    st.success(f"🏆 Best performing model: **{best_model}** (R² = {metrics_df['R²'].max():.4f})")
    
    # Data quality insights
    st.subheader("Data Quality Insights")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Improvements Made:**")
        st.write("✅ Outlier removal (3-sigma clipping)")
        st.write("✅ Better missing value handling")
        st.write("✅ Time series validation")
        st.write("✅ Enhanced feature engineering")
        st.write("✅ Robust scaling")
    
    with col2:
        st.write("**Key Statistics:**")
        st.write(f"Target mean: {y_test.mean():.3f} kW")
        st.write(f"Target std: {y_test.std():.3f} kW")
        st.write(f"Target range: {y_test.min():.3f} - {y_test.max():.3f} kW")
        st.write(f"Missing values: 0 (after preprocessing)")
    
    # Prediction confidence intervals
    if st.checkbox("Show Prediction Confidence Analysis"):
        st.subheader("Prediction Confidence Analysis")
        
        # Use Random Forest for confidence estimation
        if 'RandomForest' in models:
            rf_model = models['RandomForest']
            
            # Get predictions from all trees
            tree_predictions = np.array([tree.predict(X_test.values) for tree in rf_model.estimators_])
            
            # Calculate prediction intervals
            pred_mean = np.mean(tree_predictions, axis=0)
            pred_std = np.std(tree_predictions, axis=0)
            
            # Plot with confidence intervals
            plot_samples = min(500, len(y_test))
            
            fig_conf = go.Figure()
            
            x_vals = list(range(plot_samples))
            
            # Add confidence bands
            fig_conf.add_trace(go.Scatter(
                x=x_vals + x_vals[::-1],
                y=list(pred_mean[:plot_samples] + 1.96*pred_std[:plot_samples]) + 
                  list(pred_mean[:plot_samples] - 1.96*pred_std[:plot_samples])[::-1],
                fill='toself',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% Confidence Interval'
            ))
            
            # Add actual and predicted lines
            fig_conf.add_trace(go.Scatter(
                x=x_vals,
                y=y_test.values[:plot_samples],
                mode='lines',
                name='Actual',
                line=dict(color='blue')
            ))
            
            fig_conf.add_trace(go.Scatter(
                x=x_vals,
                y=pred_mean[:plot_samples],
                mode='lines',
                name='Predicted (Mean)',
                line=dict(color='red')
            ))
            
            fig_conf.update_layout(
                title="Predictions with Confidence Intervals (Random Forest)",
                xaxis_title="Sample Index",
                yaxis_title="Global Active Power (kW)"
            )
            
            st.plotly_chart(fig_conf, use_container_width=True)