"""
SHAP Explainability Analysis
Day 8 of Implementation Plan

Goals:
1. SHAP analysis on XGBoost model (best performer)
2. Feature importance with SHAP values
3. Waterfall plots for individual predictions
4. Dependence plots for top features
5. Feature interaction analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("SHAP EXPLAINABILITY ANALYSIS")
print("="*80)

# ============================================================================
# LOAD DATA & MODEL
# ============================================================================
print("\n[LOAD] Loading dataset and models...")
df = pd.read_csv('data/processed/aadhaar_with_indices.csv')
df['date'] = pd.to_datetime(df['date'])

# Load XGBoost model
xgb_model = joblib.load('outputs/models/xgboost_v3.pkl')
print("DONE: Loaded XGBoost model")

# Prepare features (same as training)
feature_cols = [col for col in df.columns if col not in [
    'date', 'state', 'district', 'pincode', 'stability_category',
    'high_updater_3m', 'high_updater_6m', 'future_updates_3m', 'future_updates_6m',
    'future_biometric_updates', 'will_need_biometric',
    # Composite index components (don't use as features)
    'mobile_digital_score', 'saturation_score', 'stability_score', 'online_update_score',
    'accessibility_score', 'burden_score', 'compliance_score', 'resilience_score',
    'maturity_saturation', 'maturity_stability', 'maturity_compliance', 'maturity_steady',
    'engagement_frequency', 'engagement_biometric', 'engagement_mobility', 'engagement_address',
    # Composite indices themselves
    'digital_inclusion_index', 'service_quality_score', 'aadhaar_maturity_index', 'citizen_engagement_index'
]]

X = df[feature_cols].fillna(0)
y = df['high_updater_3m']

print(f"Features: {X.shape[1]}")
print(f"Target: {y.value_counts().to_dict()}")

# Train-test split (same split as training)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nDONE: Data prepared:")
print(f"   Train: {X_train.shape}")
print(f"   Test: {X_test.shape}")

# ============================================================================
# 1. CREATE SHAP EXPLAINER
# ============================================================================
print("\n" + "="*80)
print("SHAP EXPLAINER CREATION")
print("="*80)

print("\n[SHAP] Creating TreeExplainer (optimized for XGBoost)...")
explainer = shap.TreeExplainer(xgb_model)
print("DONE: Explainer created")

# Calculate SHAP values (use sample for speed)
sample_size = min(1000, len(X_test))
X_sample = X_test.sample(n=sample_size, random_state=42)

print(f"\n[SHAP] Calculating SHAP values for {sample_size} samples...")
shap_values = explainer(X_sample)
print("DONE: SHAP values calculated")

# ============================================================================
# 2. SHAP SUMMARY PLOT (FEATURE IMPORTANCE)
# ============================================================================
print("\n" + "="*80)
print("SHAP FEATURE IMPORTANCE")
print("="*80)

print("\n[VIZ] Creating SHAP summary plot...")
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values, X_sample, show=False, max_display=25)
plt.title('SHAP Feature Importance - XGBoost Model', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('outputs/figures/shap_summary_plot.png', dpi=300, bbox_inches='tight')
print("DONE: Saved shap_summary_plot.png")
plt.close()

# ============================================================================
# 3. SHAP BAR PLOT (MEAN ABSOLUTE SHAP VALUES)
# ============================================================================
print("\n[VIZ] Creating SHAP bar plot...")
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=25)
plt.title('Mean |SHAP Value| - Feature Importance', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('outputs/figures/shap_bar_plot.png', dpi=300, bbox_inches='tight')
print("DONE: Saved shap_bar_plot.png")
plt.close()

# ============================================================================
# 4. SHAP WATERFALL PLOTS (INDIVIDUAL PREDICTIONS)
# ============================================================================
print("\n" + "="*80)
print("SHAP WATERFALL PLOTS (INDIVIDUAL PREDICTIONS)")
print("="*80)

# Select interesting cases
y_sample = y_test.loc[X_sample.index]
probas = xgb_model.predict_proba(X_sample)[:, 1]

# Case 1: High confidence positive (>0.9)
high_conf_idx = np.where(probas > 0.9)[0]
if len(high_conf_idx) > 0:
    idx_high = high_conf_idx[0]
    print(f"\n[WATERFALL] Case 1: High confidence positive (p={probas[idx_high]:.3f})")
    
    plt.figure(figsize=(12, 8))
    shap.waterfall_plot(shap_values[idx_high], show=False, max_display=20)
    plt.title(f'Waterfall Plot - High Confidence Positive (p={probas[idx_high]:.3f})', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/figures/shap_waterfall_high_positive.png', dpi=300, bbox_inches='tight')
    print("DONE: Saved shap_waterfall_high_positive.png")
    plt.close()

# Case 2: High confidence negative (<0.1)
low_conf_idx = np.where(probas < 0.1)[0]
if len(low_conf_idx) > 0:
    idx_low = low_conf_idx[0]
    print(f"\n[WATERFALL] Case 2: High confidence negative (p={probas[idx_low]:.3f})")
    
    plt.figure(figsize=(12, 8))
    shap.waterfall_plot(shap_values[idx_low], show=False, max_display=20)
    plt.title(f'Waterfall Plot - High Confidence Negative (p={probas[idx_low]:.3f})', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/figures/shap_waterfall_high_negative.png', dpi=300, bbox_inches='tight')
    print("DONE: Saved shap_waterfall_high_negative.png")
    plt.close()

# Case 3: Uncertain (0.4-0.6)
uncertain_idx = np.where((probas >= 0.4) & (probas <= 0.6))[0]
if len(uncertain_idx) > 0:
    idx_uncertain = uncertain_idx[0]
    print(f"\n[WATERFALL] Case 3: Uncertain prediction (p={probas[idx_uncertain]:.3f})")
    
    plt.figure(figsize=(12, 8))
    shap.waterfall_plot(shap_values[idx_uncertain], show=False, max_display=20)
    plt.title(f'Waterfall Plot - Uncertain Prediction (p={probas[idx_uncertain]:.3f})', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/figures/shap_waterfall_uncertain.png', dpi=300, bbox_inches='tight')
    print("DONE: Saved shap_waterfall_uncertain.png")
    plt.close()

# ============================================================================
# 5. SHAP DEPENDENCE PLOTS (TOP 5 FEATURES)
# ============================================================================
print("\n" + "="*80)
print("SHAP DEPENDENCE PLOTS")
print("="*80)

# Get top 5 features by mean |SHAP|
mean_abs_shap = np.abs(shap_values.values).mean(0)
top_indices = np.argsort(mean_abs_shap)[::-1][:5]
top_features = [X_sample.columns[i] for i in top_indices]

print(f"\nTop 5 features: {top_features}")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, (feature_idx, feature_name) in enumerate(zip(top_indices[:5], top_features)):
    ax = axes[idx]
    
    shap.dependence_plot(
        feature_idx, 
        shap_values.values, 
        X_sample,
        ax=ax,
        show=False
    )
    ax.set_title(f'Dependence: {feature_name}', fontsize=11, fontweight='bold')

# Remove extra subplot
axes[5].axis('off')

plt.suptitle('SHAP Dependence Plots - Top 5 Features', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('outputs/figures/shap_dependence_plots.png', dpi=300, bbox_inches='tight')
print("DONE: Saved shap_dependence_plots.png")
plt.close()

# ============================================================================
# 6. SHAP FORCE PLOT (SAMPLE)
# ============================================================================
print("\n[VIZ] Creating SHAP force plot...")

# Force plot for first 100 predictions
shap.initjs()
force_plot = shap.force_plot(
    explainer.expected_value, 
    shap_values.values[:100], 
    X_sample.iloc[:100],
    show=False
)

# Save as HTML
shap.save_html('outputs/figures/shap_force_plot.html', force_plot)
print("DONE: Saved shap_force_plot.html")

# ============================================================================
# 7. FEATURE IMPORTANCE TABLE
# ============================================================================
print("\n" + "="*80)
print("SHAP FEATURE IMPORTANCE TABLE")
print("="*80)

# Calculate importance metrics
feature_importance = pd.DataFrame({
    'Feature': X_sample.columns,
    'Mean_Abs_SHAP': np.abs(shap_values.values).mean(0),
    'Mean_SHAP': shap_values.values.mean(0),
    'Std_SHAP': shap_values.values.std(0)
})

feature_importance = feature_importance.sort_values('Mean_Abs_SHAP', ascending=False)
feature_importance.to_csv('outputs/tables/shap_feature_importance.csv', index=False)

print("\nTOP 20 FEATURES BY SHAP IMPORTANCE:")
print(feature_importance.head(20).to_string(index=False))

print("\nDONE: Saved shap_feature_importance.csv")

# ============================================================================
# 8. SHAP INTERACTION VALUES (TOP 3 FEATURES)
# ============================================================================
print("\n" + "="*80)
print("SHAP INTERACTION ANALYSIS")
print("="*80)

print("\n[SHAP] Calculating interaction values (this may take a while)...")

# Use smaller sample for interactions (computationally expensive)
interaction_sample_size = min(200, len(X_sample))
X_interaction = X_sample.iloc[:interaction_sample_size]

try:
    shap_interaction_values = explainer.shap_interaction_values(X_interaction)
    
    # Plot interaction for top 2 features
    top_2_features = top_features[:2]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, feature_name in enumerate(top_2_features):
        feature_idx = X_sample.columns.get_loc(feature_name)
        
        shap.dependence_plot(
            (feature_idx, feature_idx),
            shap_interaction_values,
            X_interaction,
            ax=axes[idx],
            show=False
        )
        axes[idx].set_title(f'Interaction: {feature_name}', fontsize=12, fontweight='bold')
    
    plt.suptitle('SHAP Interaction Plots', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/figures/shap_interaction_plots.png', dpi=300, bbox_inches='tight')
    print("DONE: Saved shap_interaction_plots.png")
    plt.close()
    
except Exception as e:
    print(f"WARNING: Interaction analysis skipped: {str(e)}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("SHAP EXPLAINABILITY SUMMARY")
print("="*80)

print(f"\nANALYSIS SUMMARY:")
print(f"   Model: XGBoost (72.48% ROC-AUC)")
print(f"   Features analyzed: {len(feature_cols)}")
print(f"   Predictions explained: {sample_size}")
print(f"   Top feature: {top_features[0]}")

print(f"\nTOP 5 MOST IMPORTANT FEATURES:")
for i, feat in enumerate(top_features, 1):
    importance = feature_importance[feature_importance['Feature'] == feat]['Mean_Abs_SHAP'].values[0]
    print(f"   {i}. {feat}: {importance:.4f}")

print(f"\nOUTPUTS:")
print(f"   - outputs/figures/shap_summary_plot.png")
print(f"   - outputs/figures/shap_bar_plot.png")
print(f"   - outputs/figures/shap_waterfall_*.png (3 cases)")
print(f"   - outputs/figures/shap_dependence_plots.png")
print(f"   - outputs/figures/shap_force_plot.html")
print(f"   - outputs/figures/shap_interaction_plots.png")
print(f"   - outputs/tables/shap_feature_importance.csv")

print("\nKEY INSIGHTS:")
print("   1. SHAP values explain individual predictions")
print("   2. Waterfall plots show contribution of each feature")
print("   3. Dependence plots reveal feature-target relationships")
print("   4. Interaction plots show feature synergies")
print("   5. Force plot visualizes prediction forces (HTML interactive)")

print("\n" + "="*80)
print("SHAP EXPLAINABILITY ANALYSIS COMPLETE!")
print("="*80)
