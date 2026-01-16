"""
Debug why SHAP is crashing - systematic investigation
"""

import pandas as pd
import numpy as np
import joblib
import sys
from sklearn.model_selection import train_test_split

print("SHAP DEBUG - SYSTEMATIC INVESTIGATION")
print("=" * 60)

# Step 1: Check data
print("\n[1] Loading data...")
df = pd.read_csv('data/processed/aadhaar_with_indices.csv')
print(f"   Rows: {len(df):,}")
print(f"   Columns: {len(df.columns)}")

# Step 2: Check model
print("\n[2] Loading model...")
xgb_model = joblib.load('outputs/models/xgboost_v3.pkl')
print(f"   Model type: {type(xgb_model)}")
print(f"   Model class: {xgb_model.__class__.__name__}")

# Step 3: Prepare features
feature_cols = [col for col in df.columns if col not in [
    'date', 'state', 'district', 'pincode', 'stability_category',
    'high_updater_3m', 'high_updater_6m', 'future_updates_3m', 'future_updates_6m',
    'future_biometric_updates', 'will_need_biometric',
    'mobile_digital_score', 'saturation_score', 'stability_score', 'online_update_score',
    'accessibility_score', 'burden_score', 'compliance_score', 'resilience_score',
    'maturity_saturation', 'maturity_stability', 'maturity_compliance', 'maturity_steady',
    'engagement_frequency', 'engagement_biometric', 'engagement_mobility', 'engagement_address',
    'digital_inclusion_index', 'service_quality_score', 'aadhaar_maturity_index', 'citizen_engagement_index'
]]

X = df[feature_cols].fillna(0)
y = df['high_updater_3m']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n[3] Data diagnostics...")
print(f"   Features: {len(feature_cols)}")
print(f"   Test samples: {len(X_test)}")
print(f"   NaN values: {X_test.isna().sum().sum()}")
print(f"   Inf values: {np.isinf(X_test).sum().sum()}")
print(f"   Data types: {X_test.dtypes.value_counts().to_dict()}")

# Step 4: Test prediction
print("\n[4] Testing model prediction...")
X_tiny = X_test.head(10)
try:
    preds = xgb_model.predict_proba(X_tiny)
    print(f"   SUCCESS: Predictions shape {preds.shape}")
    print(f"   Sample predictions: {preds[:3, 1]}")
except Exception as e:
    print(f"   FAILED: {e}")
    sys.exit(1)

# Step 5: Try importing SHAP
print("\n[5] Testing SHAP import...")
try:
    import shap
    print(f"   SHAP version: {shap.__version__}")
except Exception as e:
    print(f"   FAILED: {e}")
    sys.exit(1)

# Step 6: Try creating explainer
print("\n[6] Testing SHAP TreeExplainer creation...")
try:
    explainer = shap.TreeExplainer(xgb_model)
    print(f"   SUCCESS: Explainer created")
    print(f"   Explainer type: {type(explainer)}")
except Exception as e:
    print(f"   FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 7: Try calculating SHAP for 1 sample
print("\n[7] Testing SHAP calculation (1 sample)...")
try:
    X_one = X_test.head(1)
    shap_one = explainer(X_one)
    print(f"   SUCCESS: SHAP values shape {shap_one.values.shape}")
    print(f"   Sample SHAP values: {shap_one.values[0][:5]}")
except Exception as e:
    print(f"   FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 8: Try calculating SHAP for 10 samples
print("\n[8] Testing SHAP calculation (10 samples)...")
try:
    shap_ten = explainer(X_tiny)
    print(f"   SUCCESS: SHAP values shape {shap_ten.values.shape}")
except Exception as e:
    print(f"   FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 9: Try calculating SHAP for 100 samples
print("\n[9] Testing SHAP calculation (100 samples)...")
try:
    X_hundred = X_test.head(100)
    shap_hundred = explainer(X_hundred)
    print(f"   SUCCESS: SHAP values shape {shap_hundred.values.shape}")
except Exception as e:
    print(f"   FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("DEBUG COMPLETE - SHAP APPEARS TO BE WORKING")
print("="*60)
