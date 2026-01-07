# 🏛️ UIDAI Aadhaar Analytics Dashboard

Interactive Streamlit dashboard showcasing comprehensive ML analytics on UIDAI Aadhaar enrollment and update patterns.

## 🚀 Quick Start

### Prerequisites
- Python 3.14+
- Virtual environment activated

### Installation

```bash
# Install required packages
pip install streamlit plotly
```

### Run Dashboard

```bash
streamlit run app.py
```

The dashboard will open automatically in your default browser at `http://localhost:8501`

## 📊 Dashboard Features

### 1. 🏠 Overview
- Project summary and key metrics
- Data distribution visualizations
- Geographic and temporal trends

### 2. 🔮 Prediction Tool
- Interactive prediction interface
- Real-time probability calculation
- Confidence scoring
- Sample district data loading

### 3. 💡 SHAP Explainability
- Top features by SHAP importance
- Interactive visualizations (bar, beeswarm, dependence plots)
- Individual prediction waterfall charts
- Model transparency and interpretability

### 4. 📊 Composite Indices
- 4 multi-dimensional performance scores:
  - Digital Inclusion Index
  - Service Quality Score
  - Aadhaar Maturity Index
  - Citizen Engagement Index
- State and district rankings
- Distribution analysis

### 5. 🎯 Clustering Analysis
- 5 distinct behavioral clusters
- Cluster characteristics and profiles
- PCA visualization

### 6. 📈 Forecasting
- ARIMA time-series predictions
- Prophet seasonality decomposition
- Multi-horizon forecasts (7/30/90 days)

### 7. 🏆 Top Performers
- Top 10 states ranking
- Top 15 districts ranking
- Multi-index radar charts
- Medal podium for top 3

### 8. 📋 About
- Project methodology
- Technical stack details
- Key insights summary

## 🎨 Dashboard Navigation

Use the sidebar to navigate between different sections. Each page provides:
- Interactive visualizations (Plotly)
- Real-time data filtering
- Downloadable insights
- Responsive design

## 📁 Required Files

The dashboard expects the following files to exist:

### Data Files
- `data/processed/aadhaar_with_indices.csv` - Main dataset
- `outputs/tables/shap_feature_importance.csv` - SHAP results
- `outputs/tables/district_index_rankings.csv` - District rankings
- `outputs/tables/state_index_rankings.csv` - State rankings

### Model Files
- `outputs/models/xgboost_v3.pkl` - Trained XGBoost model
- `outputs/models/scaler_v3.pkl` - Feature scaler
- `outputs/models/shap_values.pkl` - SHAP analysis data

### Visualization Files
- `outputs/figures/shap_summary_bar.png`
- `outputs/figures/shap_summary_beeswarm.png`
- `outputs/figures/shap_dependence_plots.png`
- `outputs/figures/shap_waterfall_*.png`
- `outputs/figures/clustering_pca_visualization.png`
- `outputs/figures/ts_*.png` (time-series forecasts)

## 🔧 Customization

### Modify Color Scheme
Edit the CSS in `app.py` at the top of the file.

### Add New Pages
Add new sections in the sidebar navigation and create corresponding page blocks.

### Adjust Metrics
Modify the calculation logic for composite indices or add new derived metrics.

## 💡 Tips for Best Experience

1. **Full Screen Mode**: Press `F11` in your browser for immersive experience
2. **Dark Mode**: Click ⚙️ Settings → Theme → Dark
3. **Responsive**: Works on desktop, tablet, and mobile
4. **Interactive**: Click on charts to zoom, pan, and explore
5. **Search**: Use search boxes to filter specific districts/states

## 🐛 Troubleshooting

### Dashboard won't start
```bash
# Check if streamlit is installed
pip show streamlit

# Reinstall if needed
pip install streamlit --upgrade
```

### Missing visualizations
Ensure all required PNG files exist in `outputs/figures/`

### Model loading errors
Verify that model files exist in `outputs/models/` and were trained with compatible scikit-learn/xgboost versions

## 📈 Performance

- **Load time**: ~3-5 seconds
- **Memory usage**: ~500MB (with full dataset)
- **Cached data**: First load caches data for faster subsequent navigation

## 🎯 For Hackathon Presentation

**Demo Flow:**
1. Start with **Overview** - Show project scope and key metrics
2. Go to **Prediction Tool** - Live demo of making a prediction
3. Switch to **SHAP Explainability** - Explain why the model works
4. Show **Composite Indices** - Policy-relevant insights
5. End with **Top Performers** - Actionable rankings

**Talking Points:**
- "72.48% ROC-AUC with full explainability"
- "Not a black box - we can explain every prediction"
- "4 composite indices for multi-dimensional policy insights"
- "Interactive dashboard for real-time exploration"

## 📄 License

MIT License - Open for educational and research purposes

---

**Built with** ❤️ **using Streamlit, XGBoost, and SHAP**
