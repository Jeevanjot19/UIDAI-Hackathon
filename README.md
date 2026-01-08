---
title: UIDAI Hackathon 2026 - Aadhaar Analytics
emoji: 🏛️
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.29.0
app_file: app.py
pinned: false
license: mit
---

# Aadhaar Societal Intelligence Project
## *Unlocking Societal Trends in Aadhaar Enrolment and Updates*

### 🎯 Project Vision
**"Aadhaar as a Societal Sensor: AI-Driven Insights for Identity Lifecycle, Mobility & Digital Stability"**

We are not analyzing Aadhaar data — we are analyzing **society through Aadhaar data**.

---

## 📂 Project Structure

```
UIDAI Hackathon/
├── data/                          # Raw and processed datasets
│   ├── raw/                       # Original UIDAI datasets
│   │   ├── enrolment/
│   │   ├── demographic_update/
│   │   └── biometric_update/
│   └── processed/                 # Cleaned and merged data
│
├── notebooks/                     # Jupyter notebooks for analysis
│   ├── 01_data_loading.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_eda_univariate.ipynb
│   ├── 04_eda_bivariate.ipynb
│   ├── 05_eda_trivariate.ipynb
│   ├── 06_forecasting_models.ipynb
│   ├── 07_anomaly_detection.ipynb
│   ├── 08_clustering_migration.ipynb
│   ├── 09_classification_stability.ipynb
│   ├── 10_network_analysis.ipynb
│   └── 11_visualization_gallery.ipynb
│
├── src/                           # Source code modules
│   ├── __init__.py
│   ├── data_loader.py            # Data loading utilities
│   ├── feature_engineering.py    # All 25+ feature calculations
│   ├── models/                   # ML model implementations
│   │   ├── __init__.py
│   │   ├── forecasting.py
│   │   ├── anomaly.py
│   │   ├── clustering.py
│   │   └── classification.py
│   ├── visualization.py          # Plotting functions
│   └── utils.py                  # Helper functions
│
├── models/                        # Saved trained models
├── outputs/                       # Generated visualizations and reports
│   ├── figures/
│   ├── tables/
│   └── insights/
│
├── config/                        # Configuration files
│   └── config.yaml
│
├── requirements.txt               # Python dependencies
├── environment.yml                # Conda environment
└── README.md                      # This file
```

---

## 🔬 Feature Engineering Framework

### 25+ Engineered Features Across 8 Layers

#### LAYER 1: Base Features (15)
- Enrolment data (total, age groups)
- Demographic updates (name, address, DOB, gender, mobile)
- Biometric updates (fingerprint, iris, face)

#### LAYER 2: Normalized Features (5)
- Growth rates
- Adult/child shares
- Update rates

#### LAYER 3: Societal Indicators (6) ⭐ CORE DIFFERENTIATORS
1. **Mobility Indicator** - Migration proxy
2. **Digital Instability Index** - Mobile churn
3. **Identity Stability Score** - Composite stability measure
4. **Update Burden Index** - Service load
5. **Manual Labor Proxy** - Fingerprint degradation
6. **Lifecycle Transition Spike** - Age transition stress

#### LAYER 4: Temporal Features (3)
- Seasonal variance
- Rolling averages

#### LAYER 6: Equity & Inclusion (4)
- Gender disparity
- Child transition stress
- Service accessibility
- Digital divide indicator

#### LAYER 7: Network & Flow (3)
- Migration flow networks
- Update cascade patterns
- Spatial autocorrelation

#### LAYER 8: Resilience & Crisis (3)
- Anomaly severity
- Recovery rate
- Volatility index

---

## 🤖 ML/AI Models

### 1. Forecasting
- **LSTM** (multivariate temporal forecasting)
- **Prophet** (seasonal decomposition)
- **Comparison & Ensemble**

### 2. Anomaly Detection
- **Transformer-based** (attention mechanism)
- **Isolation Forest** (baseline)
- **Ensemble approach**

### 3. Clustering
- **K-Means** (migration grouping)
- **DBSCAN** (density-based patterns)

### 4. Classification
- **Random Forest** (identity stability levels)
- **XGBoost** (feature importance)

### 5. Network Analysis
- **Graph Neural Network** (migration prediction)
- **PageRank** (hub identification)

---

## 📊 Visualization Strategy

15 publication-quality visualizations:
- Animated migration heatmaps
- Sankey diagrams (state-to-state flows)
- 3D surface plots (time × age × geography)
- Network graphs (district connectivity)
- Choropleth maps with annotations
- Cohort retention curves
- Small multiples for trend comparison

---

## 🎯 Impact & Policy Recommendations

### Direct UIDAI Applications
1. **Forecasted Demand** → Resource allocation
2. **Migration Hotspots** → Temporary enrollment centers
3. **Instability Zones** → Targeted awareness drives
4. **Equity Gaps** → Service accessibility improvements

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Using conda
conda env create -f environment.yml
conda activate aadhaar-analysis

# Or using pip
pip install -r requirements.txt
```

### 2. Download Data
Place UIDAI datasets in `data/raw/` folders

### 3. Run Analysis
```bash
jupyter notebook
# Open notebooks in sequence: 01 → 02 → 03...
```

### 4. Generate Report
Final outputs will be in `outputs/` folder

---

## 📈 Evaluation Criteria Alignment

| Criteria | Our Approach |
|----------|--------------|
| **Data Analysis** | Uni/bi/tri-variate with 25+ features |
| **Creativity** | Original indices, migration networks, GNN |
| **Technical** | Modular code, tests, reproducible |
| **Visualization** | 15 annotated plots, interactive dashboards |
| **Impact** | Direct UIDAI policy recommendations |

---

## 👥 Team
UIDAI Hackathon 2026 Participant

---

## 📄 License
This project is created for the UIDAI Hackathon 2026.
