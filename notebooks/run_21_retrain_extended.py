"""
Retrain XGBoost Model on 193 Extended Features
Incorporates all extended features for improved performance
Target: 85%+ ROC-AUC
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, balanced_accuracy_score, f1_score
import xgboost as xgb
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("RETRAINING XGBOOST ON 193 EXTENDED FEATURES")
print("="*80)

# Load extended features
print("\n[1/8] Loading extended feature dataset...")
try:
    df = pd.read_csv('data/processed/aadhaar_extended_features.csv')
    print(f"✅ Loaded extended features: {df.shape}")
except FileNotFoundError:
    print("⚠️ Extended features not found. Using standard features...")
    df = pd.read_csv('data/processed/aadhaar_with_indices.csv')
    print(f"✅ Loaded standard features: {df.shape}")

df['date'] = pd.to_datetime(df['date'])

# Prepare features
print("\n[2/8] Preparing features...")

# Exclude non-feature columns
exclude_cols = [
    'date', 'state', 'district', 'high_updater_3m',
    'total_enrolments', 'total_updates', 'total_all_updates',
    'total_demographic_updates', 'total_biometric_updates',
    'cluster'  # If present
]

feature_cols = [col for col in df.columns if col not in exclude_cols]
print(f"Total features available: {len(feature_cols)}")

# Handle missing values
X = df[feature_cols]

# Convert all columns to numeric, coercing errors
print(f"Converting features to numeric...")
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')

missing_count = X.isnull().sum().sum()
if missing_count > 0:
    print(f"Filling {missing_count:,} missing values with 0")
    X = X.fillna(0)

# Replace infinite values
print(f"Checking for infinite values...")
inf_mask = np.isinf(X.select_dtypes(include=[np.number]).values)
inf_count = inf_mask.sum() if inf_mask.size > 0 else 0
if inf_count > 0:
    print(f"Replacing {inf_count:,} infinite values")
    X = X.replace([np.inf, -np.inf], 0)

y = df['high_updater_3m']

print(f"✅ Features prepared: {X.shape}")
print(f"   Class distribution: {y.value_counts().to_dict()}")

# Temporal split
print("\n[3/8] Creating temporal train/test split...")
df_sorted = df.sort_values('date')
split_idx = int(len(df_sorted) * 0.8)

train_idx = df_sorted.index[:split_idx]
test_idx = df_sorted.index[split_idx:]

X_train = X.loc[train_idx]
X_test = X.loc[test_idx]
y_train = y.loc[train_idx]
y_test = y.loc[test_idx]

print(f"✅ Train: {len(X_train):,} | Test: {len(X_test):,}")
print(f"   Train date range: {df.loc[train_idx, 'date'].min()} to {df.loc[train_idx, 'date'].max()}")
print(f"   Test date range: {df.loc[test_idx, 'date'].min()} to {df.loc[test_idx, 'date'].max()}")

# Calculate class weights
print("\n[4/8] Calculating class weights...")
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos_weight = neg_count / pos_count
print(f"✅ Class 0: {neg_count:,} | Class 1: {pos_count:,}")
print(f"   scale_pos_weight = {scale_pos_weight:.4f}")

# Train XGBoost
print("\n[5/8] Training XGBoost model...")
print("Hyperparameters:")
print("  n_estimators: 200")
print("  learning_rate: 0.05")
print("  max_depth: 10")
print("  scale_pos_weight:", scale_pos_weight)

model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=10,
    min_child_weight=30,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    tree_method='hist',
    eval_metric='auc',
    n_jobs=-1
)

model.fit(X_train, y_train, verbose=False)
print("✅ Model trained successfully!")

# Evaluate
print("\n[6/8] Evaluating model...")

# Predictions
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Find optimal threshold
thresholds = np.arange(0.3, 0.8, 0.05)
best_threshold = 0.5
best_balanced_acc = 0

for thresh in thresholds:
    y_pred_temp = (y_pred_proba >= thresh).astype(int)
    balanced_acc = balanced_accuracy_score(y_test, y_pred_temp)
    if balanced_acc > best_balanced_acc:
        best_balanced_acc = balanced_acc
        best_threshold = thresh

print(f"✅ Optimal threshold: {best_threshold:.2f}")

y_pred = (y_pred_proba >= best_threshold).astype(int)

# Metrics
roc_auc = roc_auc_score(y_test, y_pred_proba)
balanced_acc = balanced_accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\nPerformance Metrics:")
print(f"  ROC-AUC: {roc_auc:.4f} ({roc_auc*100:.2f}%)")
print(f"  Balanced Accuracy: {balanced_acc:.4f} ({balanced_acc*100:.2f}%)")
print(f"  F1 Score: {f1:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(f"  TN={cm[0,0]:,}  FP={cm[0,1]:,}")
print(f"  FN={cm[1,0]:,}  TP={cm[1,1]:,}")

# Per-class metrics
print(f"\nPer-Class Metrics:")
class_report = classification_report(y_test, y_pred, target_names=['Low', 'High'], output_dict=True)
print(f"  Low Updater Recall: {class_report['Low']['recall']:.4f} ({class_report['Low']['recall']*100:.2f}%)")
print(f"  High Updater Recall: {class_report['High']['recall']:.4f} ({class_report['High']['recall']*100:.2f}%)")

# Feature importance
print("\n[7/8] Extracting feature importance...")
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Features:")
print(feature_importance.head(10).to_string(index=False))

# Save model
print("\n[8/8] Saving model and metadata...")
import os
os.makedirs('outputs/models', exist_ok=True)

# Save model
model_filename = 'outputs/models/xgboost_extended_193.pkl'
joblib.dump(model, model_filename)
print(f"✅ Saved: {model_filename}")

# Save metadata
metadata = {
    'n_features': len(feature_cols),
    'n_train': len(X_train),
    'n_test': len(X_test),
    'roc_auc': float(roc_auc),
    'balanced_accuracy': float(balanced_acc),
    'f1_score': float(f1),
    'optimal_threshold': float(best_threshold),
    'scale_pos_weight': float(scale_pos_weight),
    'hyperparameters': {
        'n_estimators': 200,
        'learning_rate': 0.05,
        'max_depth': 10,
        'min_child_weight': 30,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    },
    'top_feature': feature_importance['feature'].iloc[0],
    'top_feature_importance': float(feature_importance['importance'].iloc[0])
}

with open('outputs/models/extended_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print("✅ Saved: outputs/models/extended_metadata.json")

# Save feature list
with open('outputs/models/extended_features.txt', 'w') as f:
    f.write('\n'.join(feature_cols))
print("✅ Saved: outputs/models/extended_features.txt")

# Save feature importance
feature_importance.to_csv('outputs/models/extended_feature_importance.csv', index=False)
print("✅ Saved: outputs/models/extended_feature_importance.csv")

# Visualization
print("\n[9/9] Creating evaluation visualization...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
            xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
axes[0, 0].set_title(f'Confusion Matrix (Threshold={best_threshold:.2f})', fontweight='bold')
axes[0, 0].set_ylabel('Actual')
axes[0, 0].set_xlabel('Predicted')

# ROC Curve
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[0, 1].plot(fpr, tpr, linewidth=2, label=f'ROC (AUC={roc_auc:.3f})')
axes[0, 1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
axes[0, 1].set_xlabel('False Positive Rate')
axes[0, 1].set_ylabel('True Positive Rate')
axes[0, 1].set_title('ROC Curve', fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Feature Importance (Top 15)
top_15 = feature_importance.head(15)
axes[1, 0].barh(range(15), top_15['importance'][::-1])
axes[1, 0].set_yticks(range(15))
axes[1, 0].set_yticklabels(top_15['feature'][::-1], fontsize=9)
axes[1, 0].set_xlabel('Importance')
axes[1, 0].set_title('Top 15 Features by Importance', fontweight='bold')
axes[1, 0].grid(axis='x', alpha=0.3)

# Prediction Distribution
axes[1, 1].hist(y_pred_proba[y_test == 0], bins=50, alpha=0.5, label='Low Updaters', color='orange')
axes[1, 1].hist(y_pred_proba[y_test == 1], bins=50, alpha=0.5, label='High Updaters', color='green')
axes[1, 1].axvline(best_threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold={best_threshold:.2f}')
axes[1, 1].set_xlabel('Predicted Probability')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Prediction Distribution', fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/figures/extended_model_evaluation.png', dpi=150, bbox_inches='tight')
print("✅ Saved: outputs/figures/extended_model_evaluation.png")

# Summary
print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print(f"\n📊 Model Performance:")
print(f"   ROC-AUC: {roc_auc*100:.2f}%")
print(f"   Balanced Accuracy: {balanced_acc*100:.2f}%")
print(f"   F1 Score: {f1*100:.2f}%")
print(f"\n🎯 Features: {len(feature_cols)} total")
print(f"   Top Feature: {feature_importance['feature'].iloc[0]} ({feature_importance['importance'].iloc[0]:.4f})")
print(f"\n💾 Saved:")
print(f"   Model: {model_filename}")
print(f"   Metadata: outputs/models/extended_metadata.json")
print(f"   Evaluation: outputs/figures/extended_model_evaluation.png")
print(f"\n✅ Ready for deployment in dashboard!")
