"""
Quick SHAP Value Generation for Dashboard
Generates SHAP values for the dashboard explainability page
"""

import pandas as pd
import numpy as np
import joblib
import pickle
import shap
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("GENERATING SHAP VALUES FOR DASHBOARD")
print("="*80)

# Load data
print("\n[1/5] Loading data...")
df = pd.read_csv('data/processed/aadhaar_with_indices.csv')
print(f"✅ Loaded {len(df):,} records")

# Load model
print("\n[2/5] Loading trained model...")
try:
    model = joblib.load('outputs/models/xgboost_balanced.pkl')
    print("✅ Loaded balanced XGBoost model")
except FileNotFoundError:
    model = joblib.load('outputs/models/xgboost_v3.pkl')
    print("✅ Loaded XGBoost v3 model")

# Prepare features
print("\n[3/5] Preparing features...")
feature_cols = model.get_booster().feature_names
X = df[feature_cols].fillna(0)

# Sample for SHAP (use 1000 samples for speed)
sample_size = min(1000, len(X))
X_sample = X.sample(sample_size, random_state=42)
print(f"✅ Using {sample_size} samples, {len(feature_cols)} features")

# Compute SHAP values
print("\n[4/5] Computing SHAP values (this may take 2-3 minutes)...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)
print("✅ SHAP values computed")

# Calculate feature importance
print("\n[5/5] Calculating feature importance...")
shap_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': np.abs(shap_values).mean(axis=0)
}).sort_values('importance', ascending=False)

# Save results
print("\nSaving results...")

# Save SHAP values
with open('outputs/models/shap_values.pkl', 'wb') as f:
    pickle.dump({
        'shap_values': shap_values,
        'X_sample': X_sample,
        'feature_names': feature_cols,
        'explainer': explainer
    }, f)
print("✅ Saved: outputs/models/shap_values.pkl")

# Save importance table
shap_importance.to_csv('outputs/tables/shap_feature_importance.csv', index=False)
print("✅ Saved: outputs/tables/shap_feature_importance.csv")

# Display top features
print("\n" + "="*80)
print("TOP 10 FEATURES BY SHAP IMPORTANCE")
print("="*80)
print(shap_importance.head(10).to_string(index=False))

print("\n" + "="*80)
print("✅ SHAP ANALYSIS COMPLETE!")
print("="*80)
print(f"""
Results:
- SHAP values computed for {sample_size} samples
- {len(feature_cols)} features analyzed
- Top feature: {shap_importance['feature'].iloc[0]} (importance: {shap_importance['importance'].iloc[0]:.4f})
- Dashboard ready: SHAP Explainability page will now work
""")
