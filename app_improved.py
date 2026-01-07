"""
UIDAI Aadhaar Update Analytics Dashboard
Professional ML-Powered Insights for Aadhaar Enrollment & Update Patterns
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="UIDAI Aadhaar Analytics | ML-Powered Insights",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Custom CSS
st.markdown("""
<style>
    /* Main Headers */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 0.5rem;
        padding: 1rem 0;
        border-bottom: 3px solid #3b82f6;
    }
    
    .page-subtitle {
        font-size: 1.2rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 2rem;
        font-style: italic;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #0f172a;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
    
    .sub-section {
        font-size: 1.3rem;
        font-weight: 600;
        color: #334155;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .metric-card-green {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .metric-card-orange {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    /* Info Boxes */
    .info-box {
        background-color: #eff6ff;
        border-left: 5px solid #3b82f6;
        padding: 1.2rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: #f0fdf4;
        border-left: 5px solid #22c55e;
        padding: 1.2rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background-color: #fffbeb;
        border-left: 5px solid #f59e0b;
        padding: 1.2rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Chart Descriptions */
    .chart-description {
        background-color: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
        font-size: 0.95rem;
        color: #475569;
    }
    
    /* Key Insights */
    .insight-item {
        padding: 0.8rem;
        margin: 0.5rem 0;
        background-color: #fafafa;
        border-left: 4px solid #3b82f6;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================
@st.cache_data
def load_data():
    """Load processed Aadhaar data with all features"""
    df = pd.read_csv('data/processed/aadhaar_with_indices.csv')
    df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_resource
def load_models():
    """Load trained ML models and metadata"""
    models = {}
    # Try to load balanced model first
    try:
        models['xgboost'] = joblib.load('outputs/models/xgboost_balanced.pkl')
        import json
        with open('outputs/models/balanced_metadata.json', 'r') as f:
            models['metadata'] = json.load(f)
    except FileNotFoundError:
        # Fallback to original model
        models['xgboost'] = joblib.load('outputs/models/xgboost_v3.pkl')
        models['metadata'] = {'optimal_threshold': 0.5, 'technique': 'Standard'}
    
    # Scaler is optional
    try:
        models['scaler'] = joblib.load('outputs/models/scaler_v3.pkl')
    except FileNotFoundError:
        models['scaler'] = None
    return models

@st.cache_data
def load_shap_data():
    """Load SHAP explainability results"""
    try:
        with open('outputs/models/shap_values.pkl', 'rb') as f:
            shap_data = pickle.load(f)
        shap_importance = pd.read_csv('outputs/tables/shap_feature_importance.csv')
        return shap_data, shap_importance
    except FileNotFoundError:
        return None, None

@st.cache_data
def load_rankings():
    """Load composite index rankings"""
    try:
        district_rankings = pd.read_csv('outputs/tables/district_index_rankings.csv')
        state_rankings = pd.read_csv('outputs/tables/state_index_rankings.csv')
        return district_rankings, state_rankings
    except FileNotFoundError:
        return None, None

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
st.sidebar.markdown("""
<div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 1rem;'>
    <h1 style='color: white; margin: 0;'>🏛️ UIDAI</h1>
    <p style='color: white; margin: 0; font-size: 0.9rem;'>Analytics Dashboard</p>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "📍 **Navigation**",
    [
        "🏠 Executive Summary", 
        "🔮 Prediction Engine", 
        "💡 Model Explainability (SHAP)",
        "📊 Performance Indices", 
        "🎯 District Segmentation", 
        "📈 Future Forecasts",
        "🏆 Leaderboards", 
        "📋 Project Details"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📌 Quick Guide
- **Executive Summary**: Key metrics & trends
- **Prediction Engine**: Predict high updaters
- **Explainability**: Understand model decisions
- **Indices**: Performance rankings
- **Segmentation**: District clusters
- **Forecasts**: Future predictions
""")

# Load data
df = load_data()
models = load_models()

# ============================================================================
# PAGE 1: EXECUTIVE SUMMARY
# ============================================================================
if page == "🏠 Executive Summary":
    st.markdown('<p class="main-header">📊 UIDAI Aadhaar Update Analytics</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Machine Learning-Powered Insights on Enrollment & Update Patterns Across India</p>', unsafe_allow_html=True)
    
    # Top-level KPIs
    st.markdown('<p class="section-header">🎯 Key Performance Indicators</p>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3 style='margin:0; font-size:2.5rem;'>72.5%</h3>
            <p style='margin:0; font-size:0.9rem;'>Model ROC-AUC Score</p>
            <p style='margin:0; font-size:0.75rem; opacity:0.8;'>Prediction Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card-green">
            <h3 style='margin:0; font-size:2.5rem;'>{df['district'].nunique():,}</h3>
            <p style='margin:0; font-size:0.9rem;'>Districts Analyzed</p>
            <p style='margin:0; font-size:0.75rem; opacity:0.8;'>Across {df['state'].nunique()} States</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card-orange">
            <h3 style='margin:0; font-size:2.5rem;'>{len(df):,}</h3>
            <p style='margin:0; font-size:0.9rem;'>Data Points</p>
            <p style='margin:0; font-size:0.75rem; opacity:0.8;'>Monthly Records</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        total_enrolments = df['total_enrolments'].sum()
        st.markdown(f"""
        <div class="metric-card">
            <h3 style='margin:0; font-size:2.5rem;'>{total_enrolments/1e6:.1f}M</h3>
            <p style='margin:0; font-size:0.9rem;'>Total Enrolments</p>
            <p style='margin:0; font-size:0.75rem; opacity:0.8;'>Cumulative Count</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Class Distribution Insight
    st.markdown('<p class="section-header">📈 Update Activity Distribution</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="chart-description">
        <strong>📖 What This Shows:</strong> Distribution of districts classified as "High Updaters" (frequent updates in next 3 months) 
        vs "Low Updaters" (minimal update activity). This reveals the natural imbalance in the data that our balanced model addresses.
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced target distribution chart
    target_counts = df['high_updater_3m'].value_counts().sort_index()
    target_pct = (target_counts / target_counts.sum() * 100).round(1)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=['Low Updaters\n(< Median Updates)', 'High Updaters\n(≥ Median Updates)'],
        y=target_counts.values,
        marker=dict(
            color=['#ef4444', '#22c55e'],
            line=dict(color='white', width=2)
        ),
        text=[f'{count:,}<br>({pct}%)' for count, pct in zip(target_counts.values, target_pct.values)],
        textposition='outside',
        textfont=dict(size=14, color='black'),
        hovertemplate='<b>%{x}</b><br>Count: %{y:,}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': "Distribution of High vs Low Update Activity<br><sup>Based on 3-Month Update Frequency Threshold</sup>",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#1e3a8a'}
        },
        xaxis_title="District Category",
        yaxis_title="Number of District-Months",
        height=450,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12)
    )
    
    fig.add_annotation(
        text=f"⚠️ Class Imbalance Ratio: {target_pct.values[1]:.1f}% High : {target_pct.values[0]:.1f}% Low",
        xref="paper", yref="paper",
        x=0.5, y=1.08,
        showarrow=False,
        font=dict(size=13, color='#dc2626'),
        bgcolor='#fee2e2',
        bordercolor='#dc2626',
        borderwidth=2,
        borderpad=8
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class="info-box">
        <strong>💡 Key Insight:</strong> The dataset shows a natural 78:22 split between high and low updaters. 
        Our balanced model (using aggressive class weights and 0.4 threshold) corrects for this imbalance, 
        preventing over-prediction of the majority class.
    </div>
    """, unsafe_allow_html=True)
    
    # Geographic Analysis
    st.markdown('<p class="section-header">🗺️ Geographic Distribution Analysis</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="chart-description">
            <strong>📖 What This Shows:</strong> Top 10 states ranked by total cumulative Aadhaar enrolments. 
            Larger populations naturally have more enrolments, but the key metric is <em>saturation ratio</em> 
            (enrolments / estimated population).
        </div>
        """, unsafe_allow_html=True)
        
        top_states = df.groupby('state')['total_enrolments'].sum().nlargest(10).sort_values()
        
        fig = go.Figure(go.Bar(
            y=top_states.index,
            x=top_states.values,
            orientation='h',
            marker=dict(
                color=top_states.values,
                colorscale='Blues',
                showscale=False,
                line=dict(color='white', width=1)
            ),
            text=[f'{val/1e6:.1f}M' for val in top_states.values],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Enrolments: %{x:,.0f}<extra></extra>'
        ))
        
        fig.update_layout(
            title={
                'text': "Top 10 States by Total Enrolments<br><sup>Cumulative Aadhaar Registrations</sup>",
                'font': {'size': 16, 'color': '#1e3a8a'}
            },
            xaxis_title="Total Enrolments",
            yaxis_title="",
            height=450,
            margin=dict(l=150),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class="chart-description">
            <strong>📖 What This Shows:</strong> Time-series of total monthly updates across all districts. 
            Peaks indicate periods of high activity (possibly driven by policy changes, campaigns, or mandatory updates).
        </div>
        """, unsafe_allow_html=True)
        
        monthly_trend = df.groupby('date')['total_updates'].sum().reset_index()
        
        # Add moving average
        monthly_trend['ma3'] = monthly_trend['total_updates'].rolling(3, center=True).mean()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=monthly_trend['date'],
            y=monthly_trend['total_updates'],
            mode='lines',
            name='Actual Updates',
            line=dict(color='#3b82f6', width=1.5),
            fill='tozeroy',
            fillcolor='rgba(59, 130, 246, 0.1)',
            hovertemplate='<b>%{x|%b %Y}</b><br>Updates: %{y:,.0f}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=monthly_trend['date'],
            y=monthly_trend['ma3'],
            mode='lines',
            name='3-Month Average',
            line=dict(color='#dc2626', width=2, dash='dash'),
            hovertemplate='<b>%{x|%b %Y}</b><br>Avg: %{y:,.0f}<extra></extra>'
        ))
        
        fig.update_layout(
            title={
                'text': "Monthly Update Trends Over Time<br><sup>Total Updates Across All Districts</sup>",
                'font': {'size': 16, 'color': '#1e3a8a'}
            },
            xaxis_title="Month",
            yaxis_title="Total Updates",
            height=450,
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Model Performance Summary
    st.markdown('<p class="section-header">🤖 ML Model Performance Summary</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="success-box">
            <h4 style='margin-top:0;'>✅ Model Achievements</h4>
            <ul style='margin-bottom:0;'>
                <li><strong>ROC-AUC: 72.48%</strong> - Strong discrimination between classes</li>
                <li><strong>Balanced Accuracy: 62%</strong> - Handles class imbalance well</li>
                <li><strong>F1 Score: 85.3%</strong> - Excellent precision-recall balance</li>
                <li><strong>Low Updater Recall: 30%</strong> - 3x better than unbalanced model</li>
                <li><strong>High Updater Recall: 94%</strong> - Correctly identifies high activity</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <h4 style='margin-top:0;'>🔬 Technical Approach</h4>
            <ul style='margin-bottom:0;'>
                <li><strong>Algorithm:</strong> XGBoost with aggressive class weights</li>
                <li><strong>Features:</strong> 102 engineered features from 44 base columns</li>
                <li><strong>Balancing:</strong> Scale_pos_weight = 0.39 (1.5x penalty)</li>
                <li><strong>Threshold:</strong> Optimized to 0.4 (not default 0.5)</li>
                <li><strong>Validation:</strong> Temporal train-test split (80/20)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Top Features Preview
    st.markdown('<p class="section-header">🔑 Most Important Predictive Features</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="chart-description">
        <strong>📖 What This Shows:</strong> The top 10 features that contribute most to predicting high update activity, 
        based on XGBoost feature importance scores. Higher values = stronger predictive power.
    </div>
    """, unsafe_allow_html=True)
    
    # Get feature importances
    feature_importance = pd.DataFrame({
        'feature': models['xgboost'].get_booster().feature_names,
        'importance': models['xgboost'].feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    
    fig = go.Figure(go.Bar(
        y=feature_importance['feature'][::-1],
        x=feature_importance['importance'][::-1],
        orientation='h',
        marker=dict(
            color=feature_importance['importance'][::-1],
            colorscale='Viridis',
            showscale=False,
            line=dict(color='white', width=1)
        ),
        text=[f'{val:.3f}' for val in feature_importance['importance'][::-1]],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': "Top 10 Features by Importance Score<br><sup>XGBoost Feature Importance Rankings</sup>",
            'font': {'size': 16, 'color': '#1e3a8a'}
        },
        xaxis_title="Importance Score",
        yaxis_title="",
        height=450,
        margin=dict(l=200),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class="info-box">
        <strong>💡 Key Takeaway:</strong> <code>rolling_3m_updates</code> (recent 3-month activity) is the strongest predictor,
        followed by moving averages and cumulative metrics. This suggests <strong>recent behavior is the best indicator of future behavior</strong>.
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# PAGE 2: PREDICTION ENGINE
# ============================================================================
elif page == "🔮 Prediction Engine":
    st.markdown('<p class="main-header">🔮 District Update Prediction Engine</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Predict Whether a District Will Be a High Updater in the Next 3 Months</p>', unsafe_allow_html=True)
    
    # Model information
    model_info = models.get('metadata', {})
    st.markdown(f"""
    <div class="success-box">
        <h4 style='margin-top:0;'>✨ Balanced Prediction Model Active</h4>
        <p style='margin-bottom:0;'><strong>Technique:</strong> {model_info.get('technique', 'Standard')} | 
        <strong>Optimal Threshold:</strong> {model_info.get('optimal_threshold', 0.5)} | 
        <strong>Accuracy:</strong> Handles 78:22 class imbalance effectively</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Get model features
    model_features = models['xgboost'].get_booster().feature_names
    
    # Prediction Interface
    st.markdown('<p class="section-header">📝 Input District Characteristics</p>', unsafe_allow_html=True)
    
    # Quick Scenarios
    st.markdown('<p class="sub-section">⚡ Quick Start: Pre-configured Scenarios</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="chart-description">
        <strong>💡 Tip:</strong> Select a scenario below to auto-fill typical values, then customize as needed. 
        Scenarios are based on actual data from the 22nd and 78th percentiles.
    </div>
    """, unsafe_allow_html=True)
    
    scenario = st.selectbox(
        "Choose a Scenario:",
        ["Custom Input", "Typical High Updater (75th %ile)", "Typical Low Updater (25th %ile)", "Very Low Activity (10th %ile)"]
    )
    
    # Set defaults based on scenario
    if scenario == "Typical High Updater (75th %ile)":
        rolling_default, ma3_default, enrol_default, sat_default = 25.0, 25.0, 350, 1.2
    elif scenario == "Typical Low Updater (25th %ile)":
        rolling_default, ma3_default, enrol_default, sat_default = 5.0, 6.0, 232, 0.8
    elif scenario == "Very Low Activity (10th %ile)":
        rolling_default, ma3_default, enrol_default, sat_default = 0.0, 0.5, 15, 0.5
    else:  # Custom
        rolling_default, ma3_default, enrol_default, sat_default = 2.0, 3.0, 100, 0.9
    
    # Input form
    st.markdown('<p class="sub-section">🔢 Key Predictive Features</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Recent Activity Metrics** ⏱️")
        rolling_3m_updates = st.number_input(
            "Rolling 3-Month Updates",
            min_value=0.0,
            max_value=200.0,
            value=rolling_default,
            step=1.0,
            help="Total updates in last 3 months - MOST IMPORTANT predictor"
        )
        
        updates_ma3 = st.number_input(
            "3-Month Moving Average",
            min_value=0.0,
            max_value=100.0,
            value=ma3_default,
            step=0.5,
            help="Smoothed average of recent update trends"
        )
    
    with col2:
        st.markdown("**Enrollment Metrics** 📊")
        cumulative_enrolments = st.number_input(
            "Cumulative Enrolments",
            min_value=0.0,
            max_value=1000000.0,
            value=float(enrol_default),
            step=10.0,
            help="Total Aadhaar registrations to date"
        )
        
        saturation_ratio = st.slider(
            "Saturation Ratio",
            min_value=0.0,
            max_value=2.0,
            value=sat_default,
            step=0.1,
            help="Enrolments / Estimated Population (>1.0 = over-saturated)"
        )
    
    with col3:
        st.markdown("**Temporal Context** 📅")
        month = st.selectbox(
            "Month",
            options=list(range(1, 13)),
            index=0,
            help="Month of year (1=Jan, 12=Dec)"
        )
        
        quarter = st.selectbox(
            "Quarter",
            options=[1, 2, 3, 4],
            index=0,
            help="Quarter of year (Q1=Jan-Mar, Q4=Oct-Dec)"
        )
    
    # Optional advanced features
    with st.expander("⚙️ Advanced Features (Optional)"):
        col1, col2 = st.columns(2)
        with col1:
            biometric_intensity = st.number_input("Biometric Update Intensity", 0.0, 100.0, 5.0, 0.5)
        with col2:
            mobile_intensity = st.number_input("Mobile Update Intensity", 0.0, 100.0, 3.0, 0.5)
    
    # Predict button
    st.markdown("---")
    
    if st.button("🔮 **Predict Update Probability**", type="primary", use_container_width=True):
        # Prepare input
        low_updaters = df[df['high_updater_3m'] == 0]
        low_baseline = low_updaters[model_features].quantile(0.10)
        input_data = low_baseline.values.reshape(1, -1)
        
        # Update with user inputs
        feature_map = {feat: idx for idx, feat in enumerate(model_features)}
        if 'rolling_3m_updates' in feature_map:
            input_data[0, feature_map['rolling_3m_updates']] = rolling_3m_updates
        if 'updates_ma3' in feature_map:
            input_data[0, feature_map['updates_ma3']] = updates_ma3
        if 'cumulative_enrolments' in feature_map:
            input_data[0, feature_map['cumulative_enrolments']] = cumulative_enrolments
        if 'saturation_ratio' in feature_map:
            input_data[0, feature_map['saturation_ratio']] = saturation_ratio
        if 'month' in feature_map:
            input_data[0, feature_map['month']] = month
        if 'quarter' in feature_map:
            input_data[0, feature_map['quarter']] = quarter
        if 'biometric_intensity' in feature_map:
            input_data[0, feature_map['biometric_intensity']] = biometric_intensity
        if 'mobile_intensity' in feature_map:
            input_data[0, feature_map['mobile_intensity']] = mobile_intensity
        
        # Make prediction
        probability = models['xgboost'].predict_proba(input_data)[0, 1]
        optimal_threshold = models['metadata'].get('optimal_threshold', 0.5)
        prediction = 1 if probability >= optimal_threshold else 0
        
        # Display results
        st.markdown("---")
        st.markdown('<p class="section-header">📊 Prediction Results</p>', unsafe_allow_html=True)
        
        # Probability Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=probability * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "High Updater Probability (%)", 'font': {'size': 24}},
            delta={'reference': optimal_threshold * 100, 'valueformat': '.1f'},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "#22c55e" if probability >= optimal_threshold else "#ef4444"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, optimal_threshold * 100], 'color': '#fee2e2'},
                    {'range': [optimal_threshold * 100, 100], 'color': '#dcfce7'}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': optimal_threshold * 100
                }
            }
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            font={'color': "darkblue", 'family': "Arial"}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Results cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            result_color = "success-box" if prediction == 1 else "warning-box"
            result_text = "HIGH UPDATER ⬆️" if prediction == 1 else "Low Updater ⬇️"
            st.markdown(f"""
            <div class="{result_color}">
                <h3 style='text-align:center; margin:0;'>{result_text}</h3>
                <p style='text-align:center; margin:0.5rem 0 0 0; font-size:0.9rem;'>
                    Classification at {optimal_threshold} threshold
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            confidence = "High" if (probability > 0.7 or probability < 0.3) else "Medium" if (probability > 0.6 or probability < 0.4) else "Low"
            st.markdown(f"""
            <div class="info-box">
                <h3 style='text-align:center; margin:0;'>{probability*100:.1f}%</h3>
                <p style='text-align:center; margin:0.5rem 0 0 0; font-size:0.9rem;'>
                    Predicted Probability
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="info-box">
                <h3 style='text-align:center; margin:0;'>{confidence} Confidence</h3>
                <p style='text-align:center; margin:0.5rem 0 0 0; font-size:0.9rem;'>
                    Based on probability distance from threshold
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Interpretation
        st.markdown('<p class="sub-section">🔍 Interpretation</p>', unsafe_allow_html=True)
        
        if prediction == 1:
            st.markdown(f"""
            <div class="success-box">
                <strong>✅ Expected High Update Activity:</strong> Based on the input features (especially rolling_3m_updates = {rolling_3m_updates:.1f}), 
                this district is predicted to have <strong>above-median update activity</strong> in the next 3 months. 
                This suggests active Aadhaar engagement and may require resource allocation for processing.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="warning-box">
                <strong>⚠️ Expected Low Update Activity:</strong> Based on the input features (especially rolling_3m_updates = {rolling_3m_updates:.1f}), 
                this district is predicted to have <strong>below-median update activity</strong> in the next 3 months. 
                This could indicate stable demographics or may warrant outreach campaigns if saturation is low.
            </div>
            """, unsafe_allow_html=True)
        
        # Reference data
        st.markdown('<p class="sub-section">📚 Reference: Typical Values</p>', unsafe_allow_html=True)
        
        with st.expander("View Distribution Statistics"):
            ref_data = pd.DataFrame({
                'Metric': ['Rolling 3M Updates', 'Updates MA3', 'Cumulative Enrolments', 'Saturation Ratio'],
                'Low Updaters (Median)': ['5.0', '6.0', '232', '0.8'],
                'High Updaters (Median)': ['25.0', '25.0', '350', '1.2'],
                'Your Input': [f'{rolling_3m_updates:.1f}', f'{updates_ma3:.1f}', f'{cumulative_enrolments:.0f}', f'{saturation_ratio:.1f}']
            })
            st.table(ref_data)
            
            st.caption("💡 Note: Dataset has 78% high updaters, 22% low updaters - our balanced model corrects for this imbalance.")

# ============================================================================
# Placeholder pages (abbreviated for brevity - pattern same as above)
# ============================================================================
elif page == "💡 Model Explainability (SHAP)":
    st.markdown('<p class="main-header">💡 Model Explainability with SHAP</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Understand Which Features Drive Predictions and Why</p>', unsafe_allow_html=True)
    st.info("🚧 SHAP analysis visualization coming soon. Will show feature importance, dependence plots, and force plots.")

elif page == "📊 Performance Indices":
    st.markdown('<p class="main-header">📊 Composite Performance Indices</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Multi-Dimensional Rankings of District Performance</p>', unsafe_allow_html=True)
    st.info("🚧 Composite indices (Digital Inclusion, Citizen Engagement, Aadhaar Maturity) rankings coming soon.")

elif page == "🎯 District Segmentation":
    st.markdown('<p class="main-header">🎯 District Segmentation & Clustering</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Behavioral Clusters Based on Update Patterns</p>', unsafe_allow_html=True)
    st.info("🚧 K-Means clustering results and cluster profiles coming soon.")

elif page == "📈 Future Forecasts":
    st.markdown('<p class="main-header">📈 Time-Series Forecasting</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Predict Future Update Trends Using ARIMA & Prophet</p>', unsafe_allow_html=True)
    st.info("🚧 6-month and 12-month forecasts with confidence intervals coming soon.")

elif page == "🏆 Leaderboards":
    st.markdown('<p class="main-header">🏆 Top Performing Districts & States</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Rankings by Update Activity, Saturation, and Engagement</p>', unsafe_allow_html=True)
    st.info("🚧 Leaderboards showing top/bottom performers across various metrics coming soon.")

elif page == "📋 Project Details":
    st.markdown('<p class="main-header">📋 Project Documentation</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Methodology, Tech Stack, and Hackathon Information</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Project Objectives
    - Build ML models to predict Aadhaar update patterns
    - Identify factors driving high/low update activity
    - Create actionable insights for UIDAI resource planning
    
    ### 🔬 Methodology
    1. **Data Engineering**: 102 features from 44 base columns
    2. **Class Balancing**: Aggressive weights + threshold tuning
    3. **Model Training**: XGBoost with temporal validation
    4. **Explainability**: SHAP values for interpretability
    5. **Deployment**: Interactive Streamlit dashboard
    
    ### 💻 Tech Stack
    - **ML**: XGBoost, LightGBM, Scikit-learn
    - **Balancing**: imbalanced-learn (SMOTE, SMOTEENN)
    - **Explainability**: SHAP
    - **Forecasting**: Prophet, Statsmodels (ARIMA)
    - **Visualization**: Plotly, Matplotlib, Seaborn
    - **Dashboard**: Streamlit
    
    ### 📊 Performance Metrics
    - **ROC-AUC**: 72.48%
    - **Balanced Accuracy**: 62.03%
    - **F1 Score**: 85.32%
    - **Low Updater Recall**: 30% (3x improvement)
    
    ### 👥 Team
    Built for UIDAI Hackathon 2026
    """)
