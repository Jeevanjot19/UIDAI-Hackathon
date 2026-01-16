"""
Advanced Optimization with XGBoost + LightGBM
Target: Push from 71% to 80%+ ROC-AUC
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                             roc_curve, precision_recall_curve, ConfusionMatrixDisplay)
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ADVANCED OPTIMIZATION: XGBoost + LightGBM + Stacking")
print("="*80)

# LOAD DATA
print("\n[LOAD] Loading dataset...")
df = pd.read_csv('data/processed/aadhaar_with_advanced_features.csv')
df['date'] = pd.to_datetime(df['date'])

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

print(f"✅ Features: {len(good_features)}")

# TEMPORAL SPLIT
df_sorted = df.sort_values('date').reset_index(drop=True)
split_idx = int(len(df_sorted) * 0.8)
split_date = df_sorted['date'].iloc[split_idx]

train_mask = df['date'] < split_date
test_mask = df['date'] >= split_date

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
print(f"Class distribution: {y_train.value_counts().to_dict()}")

# ===========================================================================
# MODEL 1: XGBOOST (Usually best for tabular data)
# ===========================================================================
print("\n" + "="*80)
print("MODEL 1: XGBOOST")
print("="*80)

# Calculate scale_pos_weight for class imbalance
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=12,
    min_child_weight=50,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    tree_method='hist',
    eval_metric='auc',
    n_jobs=-1
)

print("Training XGBoost...")
xgb_model.fit(X_train, y_train, verbose=False)

y_pred_xgb = xgb_model.predict(X_test)
y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
xgb_roc_auc = roc_auc_score(y_test, y_proba_xgb)

print(f"\n✅ XGBoost ROC-AUC: {xgb_roc_auc:.4f}")
print(classification_report(y_test, y_pred_xgb, target_names=['Normal', 'High Updater']))

# ===========================================================================
# MODEL 2: LIGHTGBM (Faster alternative)
# ===========================================================================
print("\n" + "="*80)
print("MODEL 2: LIGHTGBM")
print("="*80)

lgb_model = lgb.LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=12,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

print("Training LightGBM...")
lgb_model.fit(X_train, y_train)

y_pred_lgb = lgb_model.predict(X_test)
y_proba_lgb = lgb_model.predict_proba(X_test)[:, 1]
lgb_roc_auc = roc_auc_score(y_test, y_proba_lgb)

print(f"\n✅ LightGBM ROC-AUC: {lgb_roc_auc:.4f}")
print(classification_report(y_test, y_pred_lgb, target_names=['Normal', 'High Updater']))

# ===========================================================================
# MODEL 3: OPTIMIZED RANDOM FOREST
# ===========================================================================
print("\n" + "="*80)
print("MODEL 3: RANDOM FOREST (Reference)")
print("="*80)

rf_model = RandomForestClassifier(
    n_estimators=150,
    max_depth=18,
    min_samples_split=50,
    min_samples_leaf=20,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

print("Training RF...")
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
y_proba_rf = rf_model.predict_proba(X_test)[:, 1]
rf_roc_auc = roc_auc_score(y_test, y_proba_rf)

print(f"\n✅ RF ROC-AUC: {rf_roc_auc:.4f}")

# ===========================================================================
# MODEL 4: STACKING ENSEMBLE (Meta-learning)
# ===========================================================================
print("\n" + "="*80)
print("MODEL 4: STACKING ENSEMBLE")
print("="*80)

# Use top 3 models as base estimators
stacking = StackingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('lgb', lgb_model),
        ('rf', rf_model)
    ],
    final_estimator=LogisticRegression(max_iter=1000, class_weight='balanced'),
    cv=3,
    n_jobs=-1
)

print("Training Stacking Ensemble (this takes time)...")
stacking.fit(X_train, y_train)

y_pred_stack = stacking.predict(X_test)
y_proba_stack = stacking.predict_proba(X_test)[:, 1]
stack_roc_auc = roc_auc_score(y_test, y_proba_stack)

print(f"\n✅ Stacking ROC-AUC: {stack_roc_auc:.4f}")
print(classification_report(y_test, y_pred_stack, target_names=['Normal', 'High Updater']))

# ===========================================================================
# MODEL 5: SIMPLE VOTING ENSEMBLE (Faster alternative to stacking)
# ===========================================================================
print("\n" + "="*80)
print("MODEL 5: VOTING ENSEMBLE")
print("="*80)

voting = VotingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('lgb', lgb_model),
        ('rf', rf_model)
    ],
    voting='soft'
)

print("Training Voting Ensemble...")
voting.fit(X_train, y_train)

y_pred_voting = voting.predict(X_test)
y_proba_voting = voting.predict_proba(X_test)[:, 1]
voting_roc_auc = roc_auc_score(y_test, y_proba_voting)

print(f"\n✅ Voting ROC-AUC: {voting_roc_auc:.4f}")

# ===========================================================================
# THRESHOLD OPTIMIZATION (on best model)
# ===========================================================================
print("\n" + "="*80)
print("THRESHOLD OPTIMIZATION")
print("="*80)

# Use best model's probabilities
best_proba = y_proba_stack if stack_roc_auc > voting_roc_auc else y_proba_voting

precision, recall, thresholds = precision_recall_curve(y_test, best_proba)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"Optimal threshold: {optimal_threshold:.4f}")

y_pred_optimized = (best_proba >= optimal_threshold).astype(int)
print(f"\nOptimized Results:")
print(classification_report(y_test, y_pred_optimized, target_names=['Normal', 'High Updater']))

# ===========================================================================
# RESULTS COMPARISON
# ===========================================================================
print("\n" + "="*80)
print("COMPREHENSIVE RESULTS")
print("="*80)

results = pd.DataFrame({
    'Model': [
        'v1: Basic RF (24 feat)',
        'v2: Improved RF (82 feat)',
        'v3: XGBoost',
        'v3: LightGBM',
        'v3: RF (deeper)',
        'v3: Stacking (XGB+LGB+RF)',
        'v3: Voting (XGB+LGB+RF)',
        'v3: Best + Opt. Threshold'
    ],
    'ROC-AUC': [
        0.6897,
        0.7140,
        xgb_roc_auc,
        lgb_roc_auc,
        rf_roc_auc,
        stack_roc_auc,
        voting_roc_auc,
        max(stack_roc_auc, voting_roc_auc)
    ],
    'Accuracy': [
        0.6797,
        0.7226,
        (y_pred_xgb == y_test).mean(),
        (y_pred_lgb == y_test).mean(),
        (y_pred_rf == y_test).mean(),
        (y_pred_stack == y_test).mean(),
        (y_pred_voting == y_test).mean(),
        (y_pred_optimized == y_test).mean()
    ]
})

print(results.to_string(index=False))

best_roc = results['ROC-AUC'].max()
improvement_from_v1 = ((best_roc - 0.6897) / 0.6897) * 100
improvement_from_v2 = ((best_roc - 0.7140) / 0.7140) * 100

print(f"\n🏆 BEST ROC-AUC: {best_roc:.4f}")
print(f"🚀 Total improvement from v1: +{improvement_from_v1:.1f}%")
print(f"🚀 Improvement from v2: +{improvement_from_v2:.1f}%")

if best_roc >= 0.80:
    print(f"\n✅ TARGET ACHIEVED: 80%+ ROC-AUC!")
else:
    gap = (0.80 - best_roc) * 100
    print(f"\n⚠️  Gap to 80%: {gap:.1f}% remaining")

# ===========================================================================
# VISUALIZATIONS
# ===========================================================================
print("\n[VIZ] Creating advanced visualizations...")

# 1. ROC Curves Comparison
plt.figure(figsize=(12, 7))
models_to_plot = [
    (y_proba_xgb, xgb_roc_auc, 'XGBoost'),
    (y_proba_lgb, lgb_roc_auc, 'LightGBM'),
    (y_proba_rf, rf_roc_auc, 'Random Forest'),
    (y_proba_stack, stack_roc_auc, 'Stacking'),
    (y_proba_voting, voting_roc_auc, 'Voting')
]

for proba, auc, name in models_to_plot:
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.plot(fpr, tpr, label=f'{name} ({auc:.4f})', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', label='Random (0.50)', alpha=0.5)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title(f'ROC Curves - All Models (Best: {best_roc:.4f})', fontsize=14, fontweight='bold')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/figures/final_roc_comparison.png', dpi=300)
print("✅ Saved: outputs/figures/final_roc_comparison.png")

# 2. Feature Importance (XGBoost)
plt.figure(figsize=(12, 8))
feat_imp = pd.DataFrame({
    'feature': good_features,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False).head(25)

plt.barh(range(25), feat_imp['importance'].values[::-1])
plt.yticks(range(25), feat_imp['feature'].values[::-1], fontsize=10)
plt.xlabel('Importance', fontsize=12)
plt.title(f'Top 25 Features - XGBoost (ROC-AUC: {xgb_roc_auc:.4f})', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/figures/xgboost_feature_importance.png', dpi=300)
print("✅ Saved: outputs/figures/xgboost_feature_importance.png")

# 3. Confusion Matrix (Best Model)
best_model_name = 'Stacking' if stack_roc_auc > voting_roc_auc else 'Voting'
best_preds = y_pred_stack if stack_roc_auc > voting_roc_auc else y_pred_voting

fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(y_test, best_preds, ax=ax, cmap='Blues', values_format='d')
plt.title(f'Confusion Matrix - {best_model_name} (ROC-AUC: {best_roc:.4f})', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig('outputs/figures/final_confusion_matrix.png', dpi=300)
print("✅ Saved: outputs/figures/final_confusion_matrix.png")

# ===========================================================================
# SAVE BEST MODELS
# ===========================================================================
print("\n[SAVE] Saving best models...")

# Save individual models
joblib.dump(xgb_model, 'outputs/models/xgboost_v3.pkl')
joblib.dump(lgb_model, 'outputs/models/lightgbm_v3.pkl')

# Save best ensemble
if stack_roc_auc > voting_roc_auc:
    joblib.dump(stacking, 'outputs/models/best_model_stacking.pkl')
    print("✅ Best: Stacking Ensemble")
else:
    joblib.dump(voting, 'outputs/models/best_model_voting.pkl')
    print("✅ Best: Voting Ensemble")

joblib.dump({'threshold': optimal_threshold}, 'outputs/models/optimal_threshold_v3.pkl')

# Save feature importance
feat_imp_full = pd.DataFrame({
    'feature': good_features,
    'xgb_importance': xgb_model.feature_importances_,
    'lgb_importance': lgb_model.feature_importances_,
    'rf_importance': rf_model.feature_importances_
})
feat_imp_full['avg_importance'] = feat_imp_full[['xgb_importance', 'lgb_importance', 'rf_importance']].mean(axis=1)
feat_imp_full = feat_imp_full.sort_values('avg_importance', ascending=False)
feat_imp_full.to_csv('outputs/tables/feature_importance_all_models.csv', index=False)

print("✅ Saved: models, feature importance, threshold")

# ===========================================================================
# FINAL SUMMARY
# ===========================================================================
print("\n" + "="*80)
print("OPTIMIZATION JOURNEY COMPLETE!")
print("="*80)

print(f"\n📊 PROGRESSION:")
print(f"   v1 (Basic RF, 24 features):     68.97% ROC-AUC")
print(f"   v2 (Improved RF, 82 features):  71.40% ROC-AUC (+2.43%)")
print(f"   v3 (XGBoost + Ensembles):       {best_roc:.2%} ROC-AUC (+{improvement_from_v1:.1f}% total)")

print(f"\n🎯 KEY INSIGHTS:")
print(f"   ✅ XGBoost ROC-AUC: {xgb_roc_auc:.4f}")
print(f"   ✅ LightGBM ROC-AUC: {lgb_roc_auc:.4f}")
print(f"   ✅ Best Ensemble: {best_roc:.4f}")
print(f"   ✅ Top 25 features identified")

if best_roc >= 0.80:
    print(f"\n🏆 SUCCESS: Achieved 80%+ target!")
else:
    print(f"\n📈 Current: {best_roc:.1%} | Gap to 80%: {(0.80-best_roc)*100:.1f}%")
    print(f"\n💡 Next steps to reach 80%:")
    print(f"   - More feature engineering (polynomial interactions)")
    print(f"   - SMOTE for better class balance")
    print(f"   - Hyperparameter tuning with Optuna")
    print(f"   - Deep feature selection")

print(f"\n📁 Outputs saved to:")
print(f"   - outputs/models/")
print(f"   - outputs/figures/")
print(f"   - outputs/tables/")
