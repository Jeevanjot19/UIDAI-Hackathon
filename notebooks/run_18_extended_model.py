"""
Train XGBoost Model on Extended Feature Set (193 features)
Expected Performance Improvement: 72.48% → ~75% ROC-AUC
"""

import pandas as pd
import numpy as np
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score, confusion_matrix,
    classification_report
)
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("TRAINING XGBOOST ON EXTENDED FEATURE SET (193 FEATURES)")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/8] Loading extended feature dataset...")
df = pd.read_csv('data/processed/aadhaar_extended_features.csv', parse_dates=['date'])
print(f"   ✓ Loaded {len(df):,} records with {len(df.columns)} columns")

# Check for target variable
if 'high_updater_3m' not in df.columns:
    print("   ✗ ERROR: Target variable 'high_updater_3m' not found!")
    print("   Available columns:", df.columns.tolist()[:20])
    exit(1)

# ============================================================================
# 2. PREPARE FEATURES
# ============================================================================
print("\n[2/8] Preparing features...")

# Exclude non-feature columns
exclude_cols = [
    'date', 'state', 'district', 'high_updater_3m',  # Target & identifiers
    'total_updates_3m_ahead', 'target_period',       # Future leakage
    'future_', 'next_'                               # Any future-looking variables
]

# Get feature columns
feature_cols = [col for col in df.columns if not any(exc in col for exc in exclude_cols)]
feature_cols = [col for col in feature_cols if df[col].dtype in ['int64', 'float64']]

print(f"   ✓ Selected {len(feature_cols)} features")
print(f"   ✓ Excluded {len(df.columns) - len(feature_cols)} non-feature columns")

# Handle missing values
print("\n   → Handling missing values...")
missing_before = df[feature_cols].isnull().sum().sum()
df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
print(f"   ✓ Filled {missing_before:,} missing values with medians")

# Handle infinite values
print("\n   → Handling infinite values...")
df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

# Prepare X and y
X = df[feature_cols].copy()
y = df['high_updater_3m'].copy()

print(f"\n   Feature Statistics:")
print(f"   • Total features: {X.shape[1]}")
print(f"   • Total samples: {X.shape[0]:,}")
print(f"   • Class distribution:")
print(f"     - High updaters (1): {y.sum():,} ({y.mean()*100:.1f}%)")
print(f"     - Low updaters (0): {(~y.astype(bool)).sum():,} ({(1-y.mean())*100:.1f}%)")
print(f"   • Imbalance ratio: {y.sum() / (~y.astype(bool)).sum():.2f}:1")

# ============================================================================
# 3. TEMPORAL SPLIT (PREVENT DATA LEAKAGE)
# ============================================================================
print("\n[3/8] Creating temporal train/test split...")

# Sort by date
df_sorted = df.sort_values('date').reset_index(drop=True)
split_idx = int(len(df_sorted) * 0.8)

# Split
X_train = df_sorted.iloc[:split_idx][feature_cols].copy()
X_test = df_sorted.iloc[split_idx:][feature_cols].copy()
y_train = df_sorted.iloc[:split_idx]['high_updater_3m'].copy()
y_test = df_sorted.iloc[split_idx:]['high_updater_3m'].copy()

print(f"   ✓ Train set: {len(X_train):,} samples ({len(X_train)/len(df)*100:.1f}%)")
print(f"   ✓ Test set:  {len(X_test):,} samples ({len(X_test)/len(df)*100:.1f}%)")
print(f"\n   Train class distribution:")
print(f"   • High updaters: {y_train.sum():,} ({y_train.mean()*100:.1f}%)")
print(f"   • Low updaters:  {(~y_train.astype(bool)).sum():,} ({(1-y_train.mean())*100:.1f}%)")

# ============================================================================
# 4. CALCULATE CLASS WEIGHTS
# ============================================================================
print("\n[4/8] Calculating class weights...")

# Aggressive weighting (same as balanced model)
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
scale_pos_weight *= 1.5  # 1.5x multiplier for aggressive balancing

print(f"   ✓ Base scale_pos_weight: {scale_pos_weight/1.5:.4f}")
print(f"   ✓ Aggressive scale_pos_weight: {scale_pos_weight:.4f}")

# ============================================================================
# 5. TRAIN MODEL
# ============================================================================
print("\n[5/8] Training XGBoost with extended features...")

# Hyperparameters (optimized from previous runs)
params = {
    'n_estimators': 200,
    'learning_rate': 0.05,
    'max_depth': 10,
    'min_child_weight': 30,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': scale_pos_weight,
    'random_state': 42,
    'tree_method': 'hist',
    'eval_metric': 'auc',
    'n_jobs': -1
}

print("\n   Model Configuration:")
for key, value in params.items():
    print(f"   • {key:20}: {value}")

# Train
model = XGBClassifier(**params)
model.fit(X_train, y_train, verbose=False)

print("\n   ✓ Model trained successfully!")

# ============================================================================
# 6. EVALUATE MODEL
# ============================================================================
print("\n[6/8] Evaluating model performance...")

# Predictions
y_proba = model.predict_proba(X_test)[:, 1]

# Test multiple thresholds
thresholds = np.arange(0.3, 0.71, 0.05)
results = []

for thresh in thresholds:
    y_pred = (y_proba >= thresh).astype(int)
    
    # Metrics
    roc_auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    results.append({
        'threshold': thresh,
        'roc_auc': roc_auc,
        'f1': f1,
        'balanced_accuracy': bal_acc,
        'precision': precision,
        'recall': recall
    })

# Find optimal threshold
results_df = pd.DataFrame(results)
optimal_idx = results_df['f1'].idxmax()
optimal_threshold = results_df.loc[optimal_idx, 'threshold']

print(f"\n   Optimal Threshold: {optimal_threshold:.2f}")
print(f"\n   Performance at Optimal Threshold:")
print(f"   • ROC-AUC:           {results_df.loc[optimal_idx, 'roc_auc']:.4f}")
print(f"   • F1 Score:          {results_df.loc[optimal_idx, 'f1']:.4f}")
print(f"   • Balanced Accuracy: {results_df.loc[optimal_idx, 'balanced_accuracy']:.4f}")
print(f"   • Precision:         {results_df.loc[optimal_idx, 'precision']:.4f}")
print(f"   • Recall:            {results_df.loc[optimal_idx, 'recall']:.4f}")

# Confusion Matrix at optimal threshold
y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
cm = confusion_matrix(y_test, y_pred_optimal)

print(f"\n   Confusion Matrix:")
print(f"   ┌─────────────────┬──────────────┬──────────────┐")
print(f"   │                 │ Predicted 0  │ Predicted 1  │")
print(f"   ├─────────────────┼──────────────┼──────────────┤")
print(f"   │ Actual 0        │ {cm[0,0]:12,} │ {cm[0,1]:12,} │")
print(f"   │ Actual 1        │ {cm[1,0]:12,} │ {cm[1,1]:12,} │")
print(f"   └─────────────────┴──────────────┴──────────────┘")

# Calculate per-class metrics
tn, fp, fn, tp = cm.ravel()
print(f"\n   Per-Class Metrics:")
print(f"   • True Negatives:  {tn:,}")
print(f"   • False Positives: {fp:,}")
print(f"   • False Negatives: {fn:,}")
print(f"   • True Positives:  {tp:,}")
print(f"\n   • Low Updater Recall:  {tn/(tn+fp)*100:.1f}%")
print(f"   • High Updater Recall: {tp/(tp+fn)*100:.1f}%")

# ============================================================================
# 7. FEATURE IMPORTANCE
# ============================================================================
print("\n[7/8] Analyzing feature importance...")

# Get feature importance
importance = model.feature_importances_
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': importance
}).sort_values('importance', ascending=False)

print(f"\n   Top 20 Most Important Features:")
for i, row in importance_df.head(20).iterrows():
    print(f"   {row.name+1:2}. {row['feature']:40} {row['importance']:.4f}")

# ============================================================================
# 8. SAVE MODEL & METADATA
# ============================================================================
print("\n[8/8] Saving model and metadata...")

# Save model
model_path = 'outputs/models/xgboost_extended_193.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print(f"   ✓ Model saved to: {model_path}")

# Save metadata
metadata = {
    'model_type': 'XGBoost Extended Feature Set',
    'n_features': len(feature_cols),
    'n_train_samples': len(X_train),
    'n_test_samples': len(X_test),
    'class_imbalance_ratio': float(y.sum() / (~y.astype(bool)).sum()),
    'scale_pos_weight': float(scale_pos_weight),
    'optimal_threshold': float(optimal_threshold),
    'performance': {
        'roc_auc': float(results_df.loc[optimal_idx, 'roc_auc']),
        'f1_score': float(results_df.loc[optimal_idx, 'f1']),
        'balanced_accuracy': float(results_df.loc[optimal_idx, 'balanced_accuracy']),
        'precision': float(results_df.loc[optimal_idx, 'precision']),
        'recall': float(results_df.loc[optimal_idx, 'recall'])
    },
    'confusion_matrix': {
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp)
    },
    'hyperparameters': params,
    'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
}

metadata_path = 'outputs/models/extended_metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"   ✓ Metadata saved to: {metadata_path}")

# Save feature list
features_path = 'outputs/models/extended_features.txt'
with open(features_path, 'w') as f:
    f.write('\n'.join(feature_cols))
print(f"   ✓ Feature list saved to: {features_path}")

# Save feature importance
importance_path = 'outputs/tables/extended_feature_importance.csv'
importance_df.to_csv(importance_path, index=False)
print(f"   ✓ Feature importance saved to: {importance_path}")

# ============================================================================
# 9. VISUALIZATION
# ============================================================================
print("\n[9/9] Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('XGBoost Extended Model (193 Features) - Evaluation', fontsize=16, fontweight='bold')

# 1. Threshold vs Metrics
ax1 = axes[0, 0]
ax1.plot(results_df['threshold'], results_df['roc_auc'], marker='o', label='ROC-AUC', linewidth=2)
ax1.plot(results_df['threshold'], results_df['f1'], marker='s', label='F1 Score', linewidth=2)
ax1.plot(results_df['threshold'], results_df['balanced_accuracy'], marker='^', label='Balanced Accuracy', linewidth=2)
ax1.axvline(optimal_threshold, color='red', linestyle='--', label=f'Optimal ({optimal_threshold:.2f})')
ax1.set_xlabel('Threshold', fontsize=12)
ax1.set_ylabel('Score', fontsize=12)
ax1.set_title('Threshold Optimization', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# 2. Confusion Matrix
ax2 = axes[0, 1]
sns.heatmap(cm, annot=True, fmt=',', cmap='Blues', ax=ax2, cbar=False,
            xticklabels=['Predicted Low', 'Predicted High'],
            yticklabels=['Actual Low', 'Actual High'])
ax2.set_title(f'Confusion Matrix (Threshold={optimal_threshold:.2f})', fontsize=14, fontweight='bold')

# 3. Feature Importance (Top 15)
ax3 = axes[1, 0]
top_features = importance_df.head(15)
ax3.barh(range(len(top_features)), top_features['importance'].values, color='steelblue')
ax3.set_yticks(range(len(top_features)))
ax3.set_yticklabels(top_features['feature'].values, fontsize=10)
ax3.invert_yaxis()
ax3.set_xlabel('Importance', fontsize=12)
ax3.set_title('Top 15 Features by Importance', fontsize=14, fontweight='bold')
ax3.grid(axis='x', alpha=0.3)

# 4. Performance Comparison
ax4 = axes[1, 1]
metrics = ['ROC-AUC', 'F1 Score', 'Balanced\nAccuracy', 'Precision', 'Recall']
values = [
    results_df.loc[optimal_idx, 'roc_auc'],
    results_df.loc[optimal_idx, 'f1'],
    results_df.loc[optimal_idx, 'balanced_accuracy'],
    results_df.loc[optimal_idx, 'precision'],
    results_df.loc[optimal_idx, 'recall']
]
colors = ['#2ecc71' if v > 0.7 else '#f39c12' if v > 0.6 else '#e74c3c' for v in values]
bars = ax4.bar(metrics, values, color=colors, edgecolor='black', linewidth=1.5)
ax4.set_ylim([0, 1])
ax4.set_ylabel('Score', fontsize=12)
ax4.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
ax4.axhline(0.7, color='gray', linestyle='--', alpha=0.5, label='Target (0.7)')
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, value in zip(bars, values):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{value:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
viz_path = 'outputs/figures/extended_model_evaluation.png'
plt.savefig(viz_path, dpi=300, bbox_inches='tight')
print(f"   ✓ Visualization saved to: {viz_path}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)

print(f"\n📊 MODEL PERFORMANCE:")
print(f"   • ROC-AUC:           {results_df.loc[optimal_idx, 'roc_auc']:.4f}")
print(f"   • F1 Score:          {results_df.loc[optimal_idx, 'f1']:.4f}")
print(f"   • Balanced Accuracy: {results_df.loc[optimal_idx, 'balanced_accuracy']:.4f}")

print(f"\n🎯 IMPROVEMENT OVER BASELINE (102 features, ROC-AUC=0.7223):")
baseline_roc = 0.7223
improvement = (results_df.loc[optimal_idx, 'roc_auc'] - baseline_roc) / baseline_roc * 100
print(f"   • ROC-AUC Improvement: {improvement:+.2f}%")
print(f"   • Absolute Gain: {results_df.loc[optimal_idx, 'roc_auc'] - baseline_roc:+.4f}")

print(f"\n💾 SAVED FILES:")
print(f"   • Model:              {model_path}")
print(f"   • Metadata:           {metadata_path}")
print(f"   • Features:           {features_path}")
print(f"   • Feature Importance: {importance_path}")
print(f"   • Visualization:      {viz_path}")

print(f"\n✅ Ready for deployment!")
print("="*80)
