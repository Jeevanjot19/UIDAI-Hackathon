"""
SHAP Explainability Analysis - Simplified Version
Focuses on core outputs without complex visualizations
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import joblib
import shap
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("SHAP EXPLAINABILITY ANALYSIS (SIMPLIFIED)")
print("="*80)

# Load data and model
print("\n[1/7] Loading data and model...")
df = pd.read_csv('data/processed/aadhaar_with_indices.csv')
xgb_model = joblib.load('outputs/models/xgboost_v3.pkl')

# Prepare features
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
print(f"   Features: {X.shape[1]}, Test samples: {len(X_test)}")

# Create SHAP explainer
print("\n[2/7] Creating SHAP explainer...")
explainer = shap.TreeExplainer(xgb_model)
print("   Done")

# Calculate SHAP values (500 samples for speed)
print("\n[3/7] Calculating SHAP values for 500 samples...")
sample_size = 500
X_sample = X_test.sample(n=sample_size, random_state=42)
shap_values = explainer(X_sample)
print(f"   Calculated SHAP values: {shap_values.values.shape}")

# Calculate feature importance
print("\n[4/7] Computing feature importance...")
mean_abs_shap = np.abs(shap_values.values).mean(0)
feature_importance = pd.DataFrame({
    'Feature': X_sample.columns,
    'Mean_Abs_SHAP': mean_abs_shap,
    'Mean_SHAP': shap_values.values.mean(0),
    'Std_SHAP': shap_values.values.std(0)
})
feature_importance = feature_importance.sort_values('Mean_Abs_SHAP', ascending=False)
feature_importance.to_csv('outputs/tables/shap_feature_importance.csv', index=False)
print("   Saved: outputs/tables/shap_feature_importance.csv")

# Create summary plot
print("\n[5/7] Creating SHAP summary plot...")
try:
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
    plt.title('SHAP Feature Importance Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/figures/shap_summary_plot.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("   Saved: shap_summary_plot.png")
except Exception as e:
    print(f"   Skipped: {e}")

# Create bar plot
print("\n[6/7] Creating SHAP bar plot...")
try:
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=20)
    plt.title('Mean |SHAP Value| by Feature', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/figures/shap_bar_plot.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("   Saved: shap_bar_plot.png")
except Exception as e:
    print(f"   Skipped: {e}")

# Create simple feature importance chart
print("\n[7/7] Creating feature importance chart...")
top20 = feature_importance.head(20)
plt.figure(figsize=(12, 8))
plt.barh(range(len(top20)), top20['Mean_Abs_SHAP'].values, color='steelblue')
plt.yticks(range(len(top20)), top20['Feature'].values)
plt.xlabel('Mean |SHAP Value|', fontsize=12, fontweight='bold')
plt.ylabel('Feature', fontsize=12, fontweight='bold')
plt.title('Top 20 Features by SHAP Importance', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('outputs/figures/shap_feature_importance_chart.png', dpi=200, bbox_inches='tight')
plt.close()
print("   Saved: shap_feature_importance_chart.png")

# Print results
print("\n" + "="*80)
print("RESULTS")
print("="*80)
print("\nTop 10 Features by SHAP Importance:")
print(feature_importance.head(10)[['Feature', 'Mean_Abs_SHAP']].to_string(index=False))

print("\n" + "="*80)
print("SHAP ANALYSIS COMPLETE")
print("="*80)
print("\nOutputs:")
print("  - outputs/tables/shap_feature_importance.csv")
print("  - outputs/figures/shap_summary_plot.png")
print("  - outputs/figures/shap_bar_plot.png")
print("  - outputs/figures/shap_feature_importance_chart.png")
