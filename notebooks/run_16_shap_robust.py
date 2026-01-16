"""
SHAP Explainability - Final Robust Attempt
Saves results to CSV/pickle to avoid terminal display issues
"""

import pandas as pd
import numpy as np
import joblib
import shap
from sklearn.model_selection import train_test_split
import pickle
import warnings
warnings.filterwarnings('ignore')

print("SHAP EXPLAINABILITY ANALYSIS - ROBUST VERSION")
print("=" * 60)

# Load data and model
print("\n[1/6] Loading data and model...")
df = pd.read_csv('data/processed/aadhaar_with_indices.csv')
xgb_model = joblib.load('outputs/models/xgboost_v3.pkl')

# Prepare features (same as training)
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

print(f"   Features: {X.shape[1]}")
print(f"   Test samples: {len(X_test)}")

# Create SHAP explainer
print("\n[2/6] Creating SHAP TreeExplainer...")
try:
    explainer = shap.TreeExplainer(xgb_model)
    print("   SUCCESS: Explainer created")
except Exception as e:
    print(f"   FAILED: {e}")
    exit(1)

# Calculate SHAP values for small sample
print("\n[3/6] Calculating SHAP values (100 samples)...")
sample_size = 100
X_sample = X_test.sample(n=sample_size, random_state=42)

try:
    shap_values = explainer(X_sample)
    print(f"   SUCCESS: SHAP values shape: {shap_values.values.shape}")
except Exception as e:
    print(f"   FAILED: {e}")
    exit(1)

# Save SHAP values to pickle
print("\n[4/6] Saving SHAP values to pickle...")
try:
    with open('outputs/models/shap_values.pkl', 'wb') as f:
        pickle.dump({
            'shap_values': shap_values.values,
            'base_values': shap_values.base_values,
            'data': X_sample,
            'feature_names': X_sample.columns.tolist()
        }, f)
    print("   SAVED: outputs/models/shap_values.pkl")
except Exception as e:
    print(f"   WARNING: Could not save pickle: {e}")

# Calculate feature importance
print("\n[5/6] Computing SHAP feature importance...")
mean_abs_shap = np.abs(shap_values.values).mean(0)
shap_importance = pd.DataFrame({
    'Feature': X_sample.columns,
    'Mean_Abs_SHAP': mean_abs_shap,
    'Mean_SHAP': shap_values.values.mean(0),
    'Std_SHAP': shap_values.values.std(0),
    'Max_SHAP': shap_values.values.max(0),
    'Min_SHAP': shap_values.values.min(0)
})
shap_importance = shap_importance.sort_values('Mean_Abs_SHAP', ascending=False)

# Save to CSV
shap_importance.to_csv('outputs/tables/shap_feature_importance.csv', index=False)
print("   SAVED: outputs/tables/shap_feature_importance.csv")

# Get predictions and save example explanations
print("\n[6/6] Saving example prediction explanations...")
y_sample = y_test.loc[X_sample.index]
probas = xgb_model.predict_proba(X_sample)[:, 1]

# Find interesting cases
high_conf_positive = np.where(probas > 0.9)[0]
high_conf_negative = np.where(probas < 0.1)[0]
uncertain = np.where((probas >= 0.45) & (probas <= 0.55))[0]

examples = []

if len(high_conf_positive) > 0:
    idx = high_conf_positive[0]
    examples.append({
        'Type': 'High Confidence Positive',
        'Probability': probas[idx],
        'Actual': y_sample.iloc[idx],
        'SHAP_Values': dict(zip(X_sample.columns, shap_values.values[idx])),
        'Top_5_Contributors': dict(sorted(
            zip(X_sample.columns, shap_values.values[idx]), 
            key=lambda x: abs(x[1]), reverse=True)[:5])
    })

if len(high_conf_negative) > 0:
    idx = high_conf_negative[0]
    examples.append({
        'Type': 'High Confidence Negative',
        'Probability': probas[idx],
        'Actual': y_sample.iloc[idx],
        'SHAP_Values': dict(zip(X_sample.columns, shap_values.values[idx])),
        'Top_5_Contributors': dict(sorted(
            zip(X_sample.columns, shap_values.values[idx]), 
            key=lambda x: abs(x[1]), reverse=True)[:5])
    })

if len(uncertain) > 0:
    idx = uncertain[0]
    examples.append({
        'Type': 'Uncertain',
        'Probability': probas[idx],
        'Actual': y_sample.iloc[idx],
        'SHAP_Values': dict(zip(X_sample.columns, shap_values.values[idx])),
        'Top_5_Contributors': dict(sorted(
            zip(X_sample.columns, shap_values.values[idx]), 
            key=lambda x: abs(x[1]), reverse=True)[:5])
    })

# Save examples
with open('outputs/tables/shap_example_explanations.pkl', 'wb') as f:
    pickle.dump(examples, f)
print("   SAVED: outputs/tables/shap_example_explanations.pkl")

# Create readable summary
with open('outputs/tables/shap_examples_summary.txt', 'w') as f:
    for ex in examples:
        f.write(f"\n{'='*60}\n")
        f.write(f"{ex['Type']}\n")
        f.write(f"{'='*60}\n")
        f.write(f"Predicted Probability: {ex['Probability']:.4f}\n")
        f.write(f"Actual Label: {ex['Actual']}\n\n")
        f.write(f"Top 5 Contributing Features:\n")
        for feat, val in ex['Top_5_Contributors'].items():
            direction = "INCREASES" if val > 0 else "DECREASES"
            f.write(f"  - {feat}: {val:+.4f} ({direction} probability)\n")
        f.write(f"\n")

print("   SAVED: outputs/tables/shap_examples_summary.txt")

# Print summary
print("\n" + "="*60)
print("SHAP ANALYSIS COMPLETE!")
print("="*60)
print(f"\nAnalyzed {sample_size} predictions")
print(f"Top 5 features by SHAP importance:")
for i, row in shap_importance.head(5).iterrows():
    print(f"  {i+1}. {row['Feature']}: {row['Mean_Abs_SHAP']:.4f}")

print(f"\nOutputs saved:")
print(f"  - outputs/tables/shap_feature_importance.csv")
print(f"  - outputs/tables/shap_examples_summary.txt")
print(f"  - outputs/models/shap_values.pkl")
print(f"  - outputs/tables/shap_example_explanations.pkl")

print("\nTo view example explanations:")
print("  cat outputs/tables/shap_examples_summary.txt")
