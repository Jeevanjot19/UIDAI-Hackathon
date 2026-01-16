"""
Feature Importance Analysis - Alternative to SHAP
Uses XGBoost native feature importance + permutation importance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*80)

# Load data and model
print("\n[1/5] Loading data and model...")
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
print(f"   Features: {X.shape[1]}")
print(f"   Test samples: {len(X_test)}")

# Method 1: XGBoost built-in feature importance (gain-based)
print("\n[2/5] Extracting XGBoost feature importance (gain)...")
importance_gain = xgb_model.feature_importances_

# Get feature names from the model (in case order differs)
if hasattr(xgb_model, 'get_booster'):
    model_feature_names = xgb_model.get_booster().feature_names
else:
    model_feature_names = feature_cols

print(f"   Model has {len(importance_gain)} features")
print(f"   Feature list has {len(model_feature_names)} names")

feature_importance_gain = pd.DataFrame({
    'Feature': model_feature_names,
    'Importance_Gain': importance_gain
}).sort_values('Importance_Gain', ascending=False)

print("   Top 10 features by gain:")
for i, row in feature_importance_gain.head(10).iterrows():
    print(f"      {row['Feature']}: {row['Importance_Gain']:.4f}")

# Method 2: Permutation importance (more robust)
print("\n[3/5] Calculating permutation importance (this may take a minute)...")
# Use smaller sample for speed
X_perm = X_test.sample(n=min(5000, len(X_test)), random_state=42)
y_perm = y_test.loc[X_perm.index]

perm_importance = permutation_importance(
    xgb_model, X_perm, y_perm, 
    n_repeats=5, random_state=42, n_jobs=-1
)

feature_importance_perm = pd.DataFrame({
    'Feature': model_feature_names,
    'Importance_Perm_Mean': perm_importance.importances_mean,
    'Importance_Perm_Std': perm_importance.importances_std
}).sort_values('Importance_Perm_Mean', ascending=False)

print("   Top 10 features by permutation:")
for i, row in feature_importance_perm.head(10).iterrows():
    print(f"      {row['Feature']}: {row['Importance_Perm_Mean']:.4f}")

# Combine both methods
print("\n[4/5] Combining importance metrics...")
feature_importance = feature_importance_gain.merge(
    feature_importance_perm, on='Feature'
)

# Normalize and create composite score
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
feature_importance['Importance_Gain_Norm'] = scaler.fit_transform(
    feature_importance[['Importance_Gain']]
)
feature_importance['Importance_Perm_Norm'] = scaler.fit_transform(
    feature_importance[['Importance_Perm_Mean']]
)
feature_importance['Composite_Importance'] = (
    0.6 * feature_importance['Importance_Gain_Norm'] +
    0.4 * feature_importance['Importance_Perm_Norm']
)
feature_importance = feature_importance.sort_values('Composite_Importance', ascending=False)

# Save
feature_importance.to_csv('outputs/tables/feature_importance_detailed.csv', index=False)
print("   Saved: outputs/tables/feature_importance_detailed.csv")

# Create visualizations
print("\n[5/5] Creating visualizations...")

# 1. Top 20 features - composite importance
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

top20 = feature_importance.head(20)

# Gain importance
ax1 = axes[0]
ax1.barh(range(len(top20)), top20['Importance_Gain'].values, color='steelblue')
ax1.set_yticks(range(len(top20)))
ax1.set_yticklabels(top20['Feature'].values)
ax1.set_xlabel('Importance (Gain)', fontsize=12, fontweight='bold')
ax1.set_title('XGBoost Gain-Based Importance', fontsize=13, fontweight='bold')
ax1.invert_yaxis()
ax1.grid(alpha=0.3, axis='x')

# Permutation importance
ax2 = axes[1]
ax2.barh(range(len(top20)), top20['Importance_Perm_Mean'].values, color='coral')
ax2.set_yticks(range(len(top20)))
ax2.set_yticklabels(top20['Feature'].values)
ax2.set_xlabel('Importance (Permutation)', fontsize=12, fontweight='bold')
ax2.set_title('Permutation-Based Importance', fontsize=13, fontweight='bold')
ax2.invert_yaxis()
ax2.grid(alpha=0.3, axis='x')

plt.suptitle('Feature Importance Analysis - Top 20 Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/figures/feature_importance_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: feature_importance_comparison.png")

# 2. Composite importance
fig, ax = plt.subplots(figsize=(12, 10))
top25 = feature_importance.head(25)
colors = plt.cm.RdYlGn(top25['Composite_Importance'].values)
ax.barh(range(len(top25)), top25['Composite_Importance'].values, color=colors)
ax.set_yticks(range(len(top25)))
ax.set_yticklabels(top25['Feature'].values, fontsize=10)
ax.set_xlabel('Composite Importance Score', fontsize=12, fontweight='bold')
ax.set_title('Top 25 Features - Composite Importance\n(60% Gain + 40% Permutation)', 
             fontsize=14, fontweight='bold')
ax.invert_yaxis()
ax.grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('outputs/figures/feature_importance_composite.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: feature_importance_composite.png")

# 3. Importance correlation plot
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(feature_importance['Importance_Gain_Norm'], 
           feature_importance['Importance_Perm_Norm'],
           alpha=0.6, s=50, color='steelblue', edgecolors='black', linewidth=0.5)

# Annotate top 10
top10 = feature_importance.head(10)
for _, row in top10.iterrows():
    ax.annotate(row['Feature'], 
                (row['Importance_Gain_Norm'], row['Importance_Perm_Norm']),
                fontsize=8, alpha=0.7)

ax.set_xlabel('Gain-Based Importance (Normalized)', fontsize=12, fontweight='bold')
ax.set_ylabel('Permutation Importance (Normalized)', fontsize=12, fontweight='bold')
ax.set_title('Feature Importance: Gain vs Permutation', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3)
ax.plot([0, 1], [0, 1], 'r--', linewidth=2, alpha=0.5, label='y=x')
ax.legend()
plt.tight_layout()
plt.savefig('outputs/figures/feature_importance_correlation.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: feature_importance_correlation.png")

# Results
print("\n" + "="*80)
print("ANALYSIS RESULTS")
print("="*80)

print("\nTop 15 Features (Composite Importance):")
print(feature_importance.head(15)[['Feature', 'Composite_Importance', 'Importance_Gain', 'Importance_Perm_Mean']].to_string(index=False))

print("\n" + "="*80)
print("FEATURE IMPORTANCE ANALYSIS COMPLETE")
print("="*80)
print("\nOutputs:")
print("  - outputs/tables/feature_importance_detailed.csv")
print("  - outputs/figures/feature_importance_comparison.png")
print("  - outputs/figures/feature_importance_composite.png")
print("  - outputs/figures/feature_importance_correlation.png")

print("\nKey Insights:")
print("  1. Gain-based importance shows feature contribution to splits")
print("  2. Permutation importance shows prediction impact")
print("  3. Composite score combines both methods")
print("  4. Top features are most critical for model performance")
