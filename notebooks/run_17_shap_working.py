"""
SHAP Explainability - Working Version with Correct Features
"""

import pandas as pd
import numpy as np
import joblib
import shap
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')

print("SHAP EXPLAINABILITY ANALYSIS - FIXED VERSION")
print("=" * 60)

# Load data and model
print("\n[1/7] Loading data and model...")
df = pd.read_csv('data/processed/aadhaar_with_indices.csv')
xgb_model = joblib.load('outputs/models/xgboost_v3.pkl')

# Get the ACTUAL features used during training
model_features = xgb_model.get_booster().feature_names
print(f"   Model was trained on {len(model_features)} features")

# Use only those features
X = df[model_features].fillna(0)
y = df['high_updater_3m']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   Test samples: {len(X_test):,}")
print(f"   Features aligned: {len(model_features)}")

# Create SHAP explainer
print("\n[2/7] Creating SHAP TreeExplainer...")
explainer = shap.TreeExplainer(xgb_model)
print("   SUCCESS: Explainer created")

# Calculate SHAP values for sample
print("\n[3/7] Calculating SHAP values (200 samples)...")
sample_size = 200
X_sample = X_test.sample(n=sample_size, random_state=42)
y_sample = y_test.loc[X_sample.index]

shap_values = explainer(X_sample)
print(f"   SUCCESS: SHAP values shape: {shap_values.values.shape}")

# Save SHAP values
print("\n[4/7] Saving SHAP values and data...")
with open('outputs/models/shap_values.pkl', 'wb') as f:
    pickle.dump({
        'shap_values': shap_values.values,
        'base_values': shap_values.base_values,
        'data': X_sample,
        'feature_names': model_features,
        'expected_value': explainer.expected_value
    }, f)
print("   SAVED: outputs/models/shap_values.pkl")

# Calculate feature importance
print("\n[5/7] Computing SHAP feature importance...")
mean_abs_shap = np.abs(shap_values.values).mean(0)
shap_importance = pd.DataFrame({
    'Feature': model_features,
    'Mean_Abs_SHAP': mean_abs_shap,
    'Mean_SHAP': shap_values.values.mean(0),
    'Std_SHAP': shap_values.values.std(0),
    'Max_SHAP': shap_values.values.max(0),
    'Min_SHAP': shap_values.values.min(0)
})
shap_importance = shap_importance.sort_values('Mean_Abs_SHAP', ascending=False)
shap_importance.to_csv('outputs/tables/shap_feature_importance.csv', index=False)
print("   SAVED: outputs/tables/shap_feature_importance.csv")

# Create visualizations
print("\n[6/7] Creating SHAP visualizations...")

# 1. Summary plot (bar)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values.values, X_sample, plot_type="bar", show=False, max_display=20)
plt.tight_layout()
plt.savefig('outputs/figures/shap_summary_bar.png', dpi=300, bbox_inches='tight')
plt.close()
print("   SAVED: shap_summary_bar.png")

# 2. Summary plot (beeswarm)
plt.figure(figsize=(10, 10))
shap.summary_plot(shap_values.values, X_sample, show=False, max_display=20)
plt.tight_layout()
plt.savefig('outputs/figures/shap_summary_beeswarm.png', dpi=300, bbox_inches='tight')
plt.close()
print("   SAVED: shap_summary_beeswarm.png")

# 3. Top dependence plots
top_features = shap_importance.head(6)['Feature'].tolist()
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, feature in enumerate(top_features):
    shap.dependence_plot(
        feature, 
        shap_values.values, 
        X_sample,
        ax=axes[idx],
        show=False
    )
    axes[idx].set_title(f'{feature}', fontsize=10)

plt.tight_layout()
plt.savefig('outputs/figures/shap_dependence_plots.png', dpi=300, bbox_inches='tight')
plt.close()
print("   SAVED: shap_dependence_plots.png")

# Get predictions
print("\n[7/7] Creating example explanations...")
probas = xgb_model.predict_proba(X_sample)[:, 1]

# Find interesting cases
high_conf_positive = np.where(probas > 0.9)[0]
high_conf_negative = np.where(probas < 0.1)[0]
uncertain = np.where((probas >= 0.45) & (probas <= 0.55))[0]

examples = []

# Create waterfall plots for interesting cases
if len(high_conf_positive) > 0:
    idx = high_conf_positive[0]
    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(shap.Explanation(
        values=shap_values.values[idx],
        base_values=shap_values.base_values[idx],
        data=X_sample.iloc[idx].values,
        feature_names=model_features
    ), show=False)
    plt.title(f'High Confidence Positive (Prob={probas[idx]:.3f}, Actual={y_sample.iloc[idx]})')
    plt.tight_layout()
    plt.savefig('outputs/figures/shap_waterfall_high_positive.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    examples.append({
        'Type': 'High Confidence Positive',
        'Probability': probas[idx],
        'Actual': y_sample.iloc[idx],
        'Top_5_Contributors': dict(sorted(
            zip(model_features, shap_values.values[idx]), 
            key=lambda x: abs(x[1]), reverse=True)[:5])
    })

if len(high_conf_negative) > 0:
    idx = high_conf_negative[0]
    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(shap.Explanation(
        values=shap_values.values[idx],
        base_values=shap_values.base_values[idx],
        data=X_sample.iloc[idx].values,
        feature_names=model_features
    ), show=False)
    plt.title(f'High Confidence Negative (Prob={probas[idx]:.3f}, Actual={y_sample.iloc[idx]})')
    plt.tight_layout()
    plt.savefig('outputs/figures/shap_waterfall_high_negative.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    examples.append({
        'Type': 'High Confidence Negative',
        'Probability': probas[idx],
        'Actual': y_sample.iloc[idx],
        'Top_5_Contributors': dict(sorted(
            zip(model_features, shap_values.values[idx]), 
            key=lambda x: abs(x[1]), reverse=True)[:5])
    })

if len(uncertain) > 0:
    idx = uncertain[0]
    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(shap.Explanation(
        values=shap_values.values[idx],
        base_values=shap_values.base_values[idx],
        data=X_sample.iloc[idx].values,
        feature_names=model_features
    ), show=False)
    plt.title(f'Uncertain Case (Prob={probas[idx]:.3f}, Actual={y_sample.iloc[idx]})')
    plt.tight_layout()
    plt.savefig('outputs/figures/shap_waterfall_uncertain.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    examples.append({
        'Type': 'Uncertain',
        'Probability': probas[idx],
        'Actual': y_sample.iloc[idx],
        'Top_5_Contributors': dict(sorted(
            zip(model_features, shap_values.values[idx]), 
            key=lambda x: abs(x[1]), reverse=True)[:5])
    })

print(f"   SAVED: {len(examples)} waterfall plots")

# Save examples summary
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

with open('outputs/tables/shap_example_explanations.pkl', 'wb') as f:
    pickle.dump(examples, f)

# Print summary
print("\n" + "="*60)
print("SHAP ANALYSIS COMPLETE!")
print("="*60)
print(f"\nAnalyzed {sample_size} predictions")
print(f"\nTop 10 features by SHAP importance:")
for idx, row in shap_importance.head(10).iterrows():
    print(f"  {idx+1:2d}. {row['Feature']:30s} {row['Mean_Abs_SHAP']:.4f}")

print(f"\nVisualizations created:")
print(f"  - shap_summary_bar.png")
print(f"  - shap_summary_beeswarm.png")
print(f"  - shap_dependence_plots.png")
print(f"  - shap_waterfall_high_positive.png")
print(f"  - shap_waterfall_high_negative.png")
print(f"  - shap_waterfall_uncertain.png")

print(f"\nData files saved:")
print(f"  - outputs/tables/shap_feature_importance.csv")
print(f"  - outputs/tables/shap_examples_summary.txt")
print(f"  - outputs/models/shap_values.pkl")
print(f"  - outputs/tables/shap_example_explanations.pkl")

print("\nSHAP EXPLAINABILITY: COMPLETE SUCCESS!")
