# 🚀 Hugging Face Spaces Deployment Guide

## ✅ Files Successfully Pushed to GitHub (3 Commits)

### Commit 1: Core Files & Data
- ✅ `app.py` - Main Streamlit dashboard (4,831 lines)
- ✅ `data/processed/aadhaar_extended_features.csv` - 456 MB (via Git LFS)
- ✅ `.gitattributes` - Git LFS configuration
- ✅ `.gitignore` - Updated to allow deployment files
- ✅ `requirements_minimal.txt` - Streamlined dependencies

### Commit 2: Model Files
- ✅ `outputs/models/xgboost_balanced.pkl` - 1.75 MB (via Git LFS)
- ✅ `outputs/models/extended_metadata.json` - Model metadata
- ✅ `outputs/models/balanced_metadata.json` - Balanced model metadata

### Commit 3: Requirements & Features
- ✅ `requirements.txt` - Updated with SHAP library
- ✅ `outputs/models/extended_features.txt` - 193 features list
- ✅ `outputs/models/balanced_features.txt` - Balanced features list

---

## 🎯 Deployment Instructions for Hugging Face Spaces

### Step 1: Create New Space
1. Go to [Hugging Face Spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Configure:
   - **Space name**: `uidai-hackathon-2026`
   - **License**: MIT
   - **SDK**: Streamlit
   - **Space hardware**: CPU basic (free) or upgrade if needed
   - **Visibility**: Public

### Step 2: Connect to GitHub Repository
1. In Space settings, select "Repository" tab
2. Choose "Import from GitHub"
3. Connect to: `https://github.com/Jeevanjot19/UIDAI-Hackathon`
4. Branch: `main`

### Step 3: Configure Space Files
Hugging Face will automatically detect:
- `app.py` as the main Streamlit app
- `requirements.txt` for dependencies
- Git LFS files will be downloaded automatically

### Step 4: Environment Configuration (Optional)
If needed, create `packages.txt` for system dependencies:
```txt
build-essential
```

### Step 5: Verify Deployment
After deployment (5-10 minutes):
1. Check build logs for errors
2. Test the app at: `https://huggingface.co/spaces/YOUR_USERNAME/uidai-hackathon-2026`
3. Verify all 13 pages load correctly

---

## 📊 What's Included in Deployment

### Data (478 MB total)
- ✅ 180-column dataset with all engineered features
- ✅ 3 integrated datasets (Enrolment, Demographic, Biometric)
- ✅ Full temporal and geographic coverage

### Models (1.75 MB)
- ✅ XGBoost Classifier (83.29% ROC-AUC)
- ✅ 193 engineered features
- ✅ Metadata with feature importance

### Dashboard Features
- ✅ 13 interactive pages
- ✅ SHAP explainability
- ✅ Policy simulation
- ✅ Risk & governance framework
- ✅ National intelligence insights
- ✅ Model trust center

---

## ⚙️ Technical Details

### Git LFS Configuration
Large files tracked via Git LFS:
- `*.csv` files (456 MB data)
- `*.pkl` files (1.75 MB models)

### Dependencies
Core libraries in `requirements.txt`:
- streamlit>=1.25.0
- pandas>=2.0.0
- xgboost>=2.0.0
- shap>=0.42.0
- plotly>=5.14.0
- prophet>=1.1.0
- scikit-learn>=1.3.0

### Resource Requirements
- **CPU**: 2 cores minimum
- **RAM**: 4 GB minimum (8 GB recommended)
- **Disk**: 1 GB for files
- **Bandwidth**: 500 MB initial download

---

## 🔧 Troubleshooting

### If build fails:
1. Check build logs in Hugging Face Space
2. Verify Git LFS files downloaded correctly
3. Reduce dependencies if memory issues occur
4. Use `requirements_minimal.txt` instead of `requirements.txt`

### If app is slow:
1. Upgrade Space hardware (CPU basic → CPU upgrade)
2. Enable caching in Streamlit
3. Reduce data loading operations

### If features missing:
1. Verify `aadhaar_extended_features.csv` loaded (180 columns)
2. Check model files present in `outputs/models/`
3. Verify `high_updater_3m` column exists in data

---

## 📝 Repository Structure (Deployed)

```
UIDAI-Hackathon/
├── app.py                                      # Main dashboard
├── requirements.txt                            # Dependencies
├── requirements_minimal.txt                    # Minimal deps (backup)
├── .gitattributes                             # Git LFS config
├── .gitignore                                 # Deployment-ready
├── README_DEPLOYMENT.md                       # This file
├── data/
│   └── processed/
│       └── aadhaar_extended_features.csv      # 456 MB (LFS)
└── outputs/
    └── models/
        ├── xgboost_balanced.pkl               # 1.75 MB (LFS)
        ├── extended_metadata.json
        ├── balanced_metadata.json
        ├── extended_features.txt              # 193 features
        └── balanced_features.txt
```

---

## ✅ Deployment Checklist

- [x] App.py updated to use `aadhaar_extended_features.csv`
- [x] Git LFS installed and configured
- [x] Large files tracked with LFS
- [x] .gitignore updated to allow deployment files
- [x] Requirements.txt includes SHAP
- [x] Model files committed (xgboost_balanced.pkl)
- [x] Metadata files committed
- [x] Feature lists committed
- [x] All files pushed to GitHub (3 commits)
- [x] Repository public and accessible

---

## 🎉 Success!

Your UIDAI Hackathon 2026 dashboard is now ready for deployment on Hugging Face Spaces!

**GitHub Repository**: https://github.com/Jeevanjot19/UIDAI-Hackathon  
**Latest Commits**: 3 deployment commits (c89a60f, 1fd2054)  
**Total Size**: ~480 MB (456 MB data + 1.75 MB models + code)

Ready to deploy! 🚀
