"""
Balanced Model Training with Class Imbalance Handling
Techniques: SMOTE, Class Weights, Threshold Tuning
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                             roc_curve, precision_recall_curve, f1_score, balanced_accuracy_score)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("BALANCED MODEL TRAINING - Handling Class Imbalance")
print("="*80)

# LOAD DATA
print("\n[LOAD] Loading dataset...")
df = pd.read_csv('data/processed/aadhaar_with_indices.csv')
df['date'] = pd.to_datetime(df['date'])

print(f"\n📊 ORIGINAL Class Distribution:")
print(df['high_updater_3m'].value_counts())
print(f"Imbalance Ratio: {(df['high_updater_3m'].value_counts()[1] / df['high_updater_3m'].value_counts()[0]):.2f}:1")

# FEATURE SELECTION
forbidden_features = [
    'date', 'state', 'district', 'pincode',
    'high_updater_3m', 'high_updater_6m', 'will_need_biometric',
    'future_updates_3m', 'future_updates_6m', 'future_biometric_updates',
]

available_features = [col for col in df.columns if col not in forbidden_features]
numeric_features = df[available_features].select_dtypes(include=[np.number]).columns.tolist()

# Remove high missing rate features
feature_completeness = df[numeric_features].isnull().sum() / len(df)
good_features = feature_completeness[feature_completeness < 0.5].index.tolist()

X = df[good_features].fillna(df[good_features].median())
y = df['high_updater_3m']

print(f"\n✅ Features: {len(good_features)}")

# TEMPORAL SPLIT
df_sorted = df.sort_values('date').reset_index(drop=True)
split_idx = int(len(df_sorted) * 0.8)
split_date = df_sorted['date'].iloc[split_idx]

train_mask = df['date'] < split_date
test_mask = df['date'] >= split_date

X_train_original, X_test = X[train_mask], X[test_mask]
y_train_original, y_test = y[train_mask], y[test_mask]

print(f"\nTrain: {len(X_train_original):,} | Test: {len(X_test):,}")
print(f"Train Class Distribution: {y_train_original.value_counts().to_dict()}")

# ===========================================================================
# TECHNIQUE 1: SMOTE (Synthetic Minority Over-sampling)
# ===========================================================================
print("\n" + "="*80)
print("TECHNIQUE 1: SMOTE - Synthetic Minority Over-sampling")
print("="*80)

# Apply SMOTE to balance classes
smote = SMOTE(sampling_strategy=0.5, random_state=42, k_neighbors=5)  # Balance to 50% minority
X_train_smote, y_train_smote = smote.fit_resample(X_train_original, y_train_original)

print(f"\n📊 After SMOTE:")
print(f"Class 0 (Low): {(y_train_smote==0).sum():,}")
print(f"Class 1 (High): {(y_train_smote==1).sum():,}")
print(f"New Ratio: {(y_train_smote==1).sum() / (y_train_smote==0).sum():.2f}:1")

# Train XGBoost on SMOTE data
xgb_smote = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=10,
    min_child_weight=30,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    tree_method='hist',
    eval_metric='auc',
    n_jobs=-1
)

print("\n🔧 Training XGBoost with SMOTE data...")
xgb_smote.fit(X_train_smote, y_train_smote, verbose=False)

y_proba_smote = xgb_smote.predict_proba(X_test)[:, 1]
roc_auc_smote = roc_auc_score(y_test, y_proba_smote)

print(f"\n✅ SMOTE Model ROC-AUC: {roc_auc_smote:.4f}")

# ===========================================================================
# TECHNIQUE 2: Increased Class Weights
# ===========================================================================
print("\n" + "="*80)
print("TECHNIQUE 2: Aggressive Class Weights")
print("="*80)

# More aggressive scale_pos_weight
scale_pos_weight = (y_train_original == 0).sum() / (y_train_original == 1).sum()
aggressive_weight = scale_pos_weight * 1.5  # 1.5x more penalty for minority class

print(f"\n⚖️ Original scale_pos_weight: {scale_pos_weight:.2f}")
print(f"⚖️ Aggressive weight: {aggressive_weight:.2f}")

xgb_weighted = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=10,
    min_child_weight=30,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=aggressive_weight,
    random_state=42,
    tree_method='hist',
    eval_metric='auc',
    n_jobs=-1
)

print("\n🔧 Training XGBoost with aggressive class weights...")
xgb_weighted.fit(X_train_original, y_train_original, verbose=False)

y_proba_weighted = xgb_weighted.predict_proba(X_test)[:, 1]
roc_auc_weighted = roc_auc_score(y_test, y_proba_weighted)

print(f"\n✅ Weighted Model ROC-AUC: {roc_auc_weighted:.4f}")

# ===========================================================================
# TECHNIQUE 3: Combined SMOTEENN (SMOTE + Edited Nearest Neighbors)
# ===========================================================================
print("\n" + "="*80)
print("TECHNIQUE 3: SMOTEENN - Hybrid Approach")
print("="*80)

smoteenn = SMOTEENN(sampling_strategy=0.5, random_state=42)
X_train_hybrid, y_train_hybrid = smoteenn.fit_resample(X_train_original, y_train_original)

print(f"\n📊 After SMOTEENN:")
print(f"Class 0 (Low): {(y_train_hybrid==0).sum():,}")
print(f"Class 1 (High): {(y_train_hybrid==1).sum():,}")
print(f"New Ratio: {(y_train_hybrid==1).sum() / (y_train_hybrid==0).sum():.2f}:1")

xgb_hybrid = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=10,
    min_child_weight=30,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    tree_method='hist',
    eval_metric='auc',
    n_jobs=-1
)

print("\n🔧 Training XGBoost with SMOTEENN data...")
xgb_hybrid.fit(X_train_hybrid, y_train_hybrid, verbose=False)

y_proba_hybrid = xgb_hybrid.predict_proba(X_test)[:, 1]
roc_auc_hybrid = roc_auc_score(y_test, y_proba_hybrid)

print(f"\n✅ SMOTEENN Model ROC-AUC: {roc_auc_hybrid:.4f}")

# ===========================================================================
# COMPARE ALL APPROACHES
# ===========================================================================
print("\n" + "="*80)
print("PERFORMANCE COMPARISON")
print("="*80)

results = {
    'SMOTE': roc_auc_smote,
    'Aggressive Weights': roc_auc_weighted,
    'SMOTEENN': roc_auc_hybrid
}

print("\n📊 ROC-AUC Scores:")
for name, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"  {name:20s}: {score:.4f}")

# Find best model
best_model_name = max(results, key=results.get)
best_score = results[best_model_name]

print(f"\n🏆 BEST: {best_model_name} ({best_score:.4f})")

# ===========================================================================
# THRESHOLD TUNING for Best Model
# ===========================================================================
print("\n" + "="*80)
print("THRESHOLD TUNING")
print("="*80)

# Use best model's probabilities
if best_model_name == 'SMOTE':
    best_proba = y_proba_smote
    best_model = xgb_smote
elif best_model_name == 'Aggressive Weights':
    best_proba = y_proba_weighted
    best_model = xgb_weighted
else:
    best_proba = y_proba_hybrid
    best_model = xgb_hybrid

# Test different thresholds
thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
threshold_results = []

for thresh in thresholds:
    y_pred_thresh = (best_proba >= thresh).astype(int)
    f1 = f1_score(y_test, y_pred_thresh)
    balanced_acc = balanced_accuracy_score(y_test, y_pred_thresh)
    
    threshold_results.append({
        'threshold': thresh,
        'f1_score': f1,
        'balanced_accuracy': balanced_acc
    })
    
    print(f"Threshold {thresh:.2f}: F1={f1:.4f}, Balanced Acc={balanced_acc:.4f}")

# Find best threshold based on F1 score
best_threshold_idx = max(range(len(threshold_results)), 
                         key=lambda i: threshold_results[i]['f1_score'])
best_threshold = threshold_results[best_threshold_idx]['threshold']

print(f"\n🎯 OPTIMAL THRESHOLD: {best_threshold}")

# ===========================================================================
# FINAL PREDICTIONS with Optimal Threshold
# ===========================================================================
y_pred_final = (best_proba >= best_threshold).astype(int)

print("\n" + "="*80)
print("FINAL CLASSIFICATION REPORT")
print("="*80)
print(classification_report(y_test, y_pred_final, target_names=['Low Updater', 'High Updater']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_final)
print("\nConfusion Matrix:")
print(f"              Predicted Low  Predicted High")
print(f"Actual Low         {cm[0,0]:6d}        {cm[0,1]:6d}")
print(f"Actual High        {cm[1,0]:6d}        {cm[1,1]:6d}")

# ===========================================================================
# SAVE BEST MODEL
# ===========================================================================
print("\n" + "="*80)
print("SAVING BALANCED MODEL")
print("="*80)

# Save model
joblib.dump(best_model, 'outputs/models/xgboost_balanced.pkl')
print("✅ Saved: outputs/models/xgboost_balanced.pkl")

# Save feature list
with open('outputs/models/balanced_features.txt', 'w') as f:
    f.write('\n'.join(good_features))
print("✅ Saved: outputs/models/balanced_features.txt")

# Save metadata
metadata = {
    'technique': best_model_name,
    'roc_auc': float(best_score),
    'optimal_threshold': float(best_threshold),
    'original_ratio': float((y_train_original==1).sum() / (y_train_original==0).sum()),
    'features': good_features,
    'test_size': len(X_test),
    'train_size': len(X_train_original)
}

import json
with open('outputs/models/balanced_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print("✅ Saved: outputs/models/balanced_metadata.json")

# ===========================================================================
# VISUALIZATIONS
# ===========================================================================
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. ROC Curves Comparison
fpr_smote, tpr_smote, _ = roc_curve(y_test, y_proba_smote)
fpr_weighted, tpr_weighted, _ = roc_curve(y_test, y_proba_weighted)
fpr_hybrid, tpr_hybrid, _ = roc_curve(y_test, y_proba_hybrid)

axes[0, 0].plot(fpr_smote, tpr_smote, label=f'SMOTE (AUC={roc_auc_smote:.3f})', linewidth=2)
axes[0, 0].plot(fpr_weighted, tpr_weighted, label=f'Weighted (AUC={roc_auc_weighted:.3f})', linewidth=2)
axes[0, 0].plot(fpr_hybrid, tpr_hybrid, label=f'SMOTEENN (AUC={roc_auc_hybrid:.3f})', linewidth=2)
axes[0, 0].plot([0, 1], [0, 1], 'k--', label='Random')
axes[0, 0].set_xlabel('False Positive Rate', fontsize=12)
axes[0, 0].set_ylabel('True Positive Rate', fontsize=12)
axes[0, 0].set_title('ROC Curves - All Techniques', fontsize=14, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# 2. Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, best_proba)
axes[0, 1].plot(recall, precision, linewidth=2, color='darkblue')
axes[0, 1].set_xlabel('Recall', fontsize=12)
axes[0, 1].set_ylabel('Precision', fontsize=12)
axes[0, 1].set_title(f'Precision-Recall Curve - {best_model_name}', fontsize=14, fontweight='bold')
axes[0, 1].grid(alpha=0.3)

# 3. Threshold Tuning Results
thresholds_plot = [r['threshold'] for r in threshold_results]
f1_scores = [r['f1_score'] for r in threshold_results]
balanced_accs = [r['balanced_accuracy'] for r in threshold_results]

axes[1, 0].plot(thresholds_plot, f1_scores, marker='o', label='F1 Score', linewidth=2)
axes[1, 0].plot(thresholds_plot, balanced_accs, marker='s', label='Balanced Accuracy', linewidth=2)
axes[1, 0].axvline(best_threshold, color='red', linestyle='--', label=f'Optimal ({best_threshold})')
axes[1, 0].set_xlabel('Classification Threshold', fontsize=12)
axes[1, 0].set_ylabel('Score', fontsize=12)
axes[1, 0].set_title('Threshold Optimization', fontsize=14, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# 4. Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1],
            xticklabels=['Low Updater', 'High Updater'],
            yticklabels=['Low Updater', 'High Updater'])
axes[1, 1].set_ylabel('Actual', fontsize=12)
axes[1, 1].set_xlabel('Predicted', fontsize=12)
axes[1, 1].set_title(f'Confusion Matrix (Threshold={best_threshold})', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/figures/balanced_model_evaluation.png', dpi=300, bbox_inches='tight')
print("✅ Saved: outputs/figures/balanced_model_evaluation.png")

print("\n" + "="*80)
print("✅ BALANCED MODEL TRAINING COMPLETE!")
print("="*80)
print(f"\nBest Technique: {best_model_name}")
print(f"ROC-AUC: {best_score:.4f}")
print(f"Optimal Threshold: {best_threshold}")
print(f"\nModel saved to: outputs/models/xgboost_balanced.pkl")
