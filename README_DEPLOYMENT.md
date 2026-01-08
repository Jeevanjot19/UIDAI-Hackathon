# UIDAI Hackathon 2026 - Streamlit Dashboard

## 🎯 Aadhaar Update Analytics & Prediction System

A comprehensive AI-powered dashboard for analyzing and predicting Aadhaar update patterns across India, built for the UIDAI Hackathon 2026.

### ✨ Key Features

#### 📊 **13 Interactive Pages**
1. **Overview** - Executive summary with key metrics
2. **Prediction Tool** - Real-time update prediction
3. **SHAP Explainability** - Model interpretability with SHAP values
4. **Composite Indices** - Digital inclusion, service quality, maturity scores
5. **Clustering Analysis** - District segmentation (5 clusters)
6. **Forecasting** - Time series predictions (Prophet/ARIMA)
7. **Top Performers** - Best and worst performing districts
8. **Policy Simulator** - Interactive policy intervention testing
9. **Risk & Governance** - Risk assessment framework
10. **Fairness Analytics** - Urban/rural equity analysis
11. **Model Trust Center** - Confidence scoring & failure modes
12. **National Intelligence** - Migration, urban stress, digital divide
13. **About** - System roadmap & constitutional ethics

#### 🤖 **ML Models**
- **XGBoost Classifier** - 83.29% ROC-AUC
- **193 Engineered Features**
- **SHAP Explainability** - Full model transparency
- **Prophet/ARIMA** - Time series forecasting
- **KMeans + DBSCAN** - District clustering
- **Isolation Forest** - Anomaly detection

#### 🎨 **Category-Winning Differentiators**
1. ✅ Decision quality metrics (confidence, regret, uncertainty)
2. ✅ Aadhaar as national intelligence (migration, urbanization)
3. ✅ Model failure modes & trust boundaries
4. ✅ Intervention effectiveness tracking
5. ✅ Ethical & constitutional alignment
6. ✅ System evolution roadmap
7. ✅ Human-in-the-loop design
8. ✅ Uncertainty communication
9. ✅ Failure recovery & resilience

### 🚀 Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

### 📊 Data

- **180 columns** with comprehensive feature engineering
- **3 datasets integrated**: Enrolment, Demographic, Biometric
- **Granularity**: National, State, District, PIN code, Temporal
- **Advanced features**: Growth rates, seasonality, indices, anomaly scores

### 🏆 Technical Excellence

- **Data Engineering**: 180 features from 3 raw datasets
- **ML Pipeline**: XGBoost + Prophet + KMeans + Isolation Forest
- **Explainability**: Full SHAP integration with waterfall plots
- **Business Impact**: Policy simulation, intervention tracking, risk governance
- **Constitutional Ethics**: Privacy-by-design, transparency, accountability

### 📁 Project Structure

```
UIDAI-Hackathon/
├── app.py                          # Main Streamlit dashboard
├── requirements.txt                # Python dependencies
├── data/
│   └── processed/
│       └── aadhaar_extended_features.csv  # 180 feature dataset
├── outputs/
│   ├── models/
│   │   ├── xgboost_balanced.pkl
│   │   ├── extended_metadata.json
│   │   └── ...
│   └── shap/
│       └── shap_values.pkl
└── README.md
```

### 🎯 Built For

UIDAI Hackathon 2026 - Category: Aadhaar Update Pattern Analysis & Prediction

**Team**: Jeevanjot Singh  
**GitHub**: [UIDAI-Hackathon](https://github.com/Jeevanjot19/UIDAI-Hackathon)  
**License**: MIT
