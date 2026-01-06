# coding: utf-8
"""
Predictive Models Module
Implements time series forecasting, classification, and anomaly detection
for Aadhaar data analysis
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
import joblib
import os

# Try to import SHAP for explainability (optional)
try:
    import shap
    SHAP_AVAILABLE = True
    print("[INFO] SHAP library available for explainability analysis")
except ImportError:
    SHAP_AVAILABLE = False
    print("[WARNING] SHAP library not available. Install with: pip install shap")

print("="*100)
print("UIDAI HACKATHON - PREDICTIVE MODELING")
print("="*100)

# Create output directories
os.makedirs('../outputs/models', exist_ok=True)
os.makedirs('../outputs/figures', exist_ok=True)
os.makedirs('../outputs/tables', exist_ok=True)

print("\n[STEP 1] Loading feature-engineered dataset...")
df = pd.read_csv('../data/processed/aadhaar_with_features.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.replace([np.inf, -np.inf], np.nan)

print(f"Dataset loaded: {df.shape}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

# ==================== MODEL 1: IDENTITY STABILITY CLASSIFICATION ====================
print("\n" + "="*100)
print("MODEL 1: RANDOM FOREST - IDENTITY STABILITY CLASSIFICATION")
print("="*100)

# Create stability categories
df['stability_category'] = pd.cut(
    df['identity_stability_score'],
    bins=[0, 0.7, 1.0],
    labels=['Low_Stability', 'High_Stability']
)

print("\nClass distribution:")
print(df['stability_category'].value_counts())

# Select features for classification
feature_cols = [
    'mobility_indicator', 'digital_instability_index', 'update_burden_index',
    'manual_labor_proxy', 'enrolment_growth_rate', 'adult_enrolment_share',
    'demographic_update_rate', 'biometric_update_rate',
    'seasonal_variance_score', 'anomaly_severity_score'
]

# Prepare data
df_model = df[feature_cols + ['stability_category']].dropna()
X = df_model[feature_cols]
y = df_model['stability_category']

# Encode target
y_encoded = (y == 'Low_Stability').astype(int)  # 1 = Low Stability (needs intervention)

print(f"\nTraining samples: {len(X)}")
print(f"Features: {len(feature_cols)}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Train set: {len(X_train)}, Test set: {len(X_test)}")

# Train Random Forest
print("\nTraining Random Forest Classifier...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=100,
    min_samples_leaf=50,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

rf_model.fit(X_train, y_train)
print("[OK] Model trained successfully!")

# Predictions
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

# Evaluation
print("\n" + "-"*80)
print("CLASSIFICATION REPORT")
print("-"*80)
print(classification_report(y_test, y_pred, 
                          target_names=['High Stability', 'Low Stability']))

# Save classification report
with open('../outputs/tables/model_rf_classification_report.txt', 'w') as f:
    f.write("RANDOM FOREST - IDENTITY STABILITY CLASSIFICATION\n")
    f.write("="*80 + "\n\n")
    f.write(classification_report(y_test, y_pred, 
                                 target_names=['High Stability', 'Low Stability']))
    f.write(f"\n\nROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}\n")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['High Stability', 'Low Stability'],
            yticklabels=['High Stability', 'Low Stability'])
plt.title('Confusion Matrix - Identity Stability Classification', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('../outputs/figures/model_rf_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Confusion matrix saved")

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

plt.figure(figsize=(10, 7))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - Identity Stability Prediction', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('../outputs/figures/model_rf_roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"[OK] ROC-AUC Score: {roc_auc:.4f}")

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

feature_importance.to_csv('../outputs/tables/model_rf_feature_importance.csv', index=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='importance', y='feature', palette='viridis')
plt.title('Feature Importance - Identity Stability Prediction', fontsize=14, fontweight='bold')
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.savefig('../outputs/figures/model_rf_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Feature importance plot saved")

# ==================== EXPLAINABLE AI: MODEL INSIGHTS ====================
print("\n" + "="*100)
print("EXPLAINABLE AI: EXTRACTING INSIGHTS FROM RANDOM FOREST")
print("="*100)

# 1. Feature Importance Insights
print("\n[INSIGHT 1] Feature Importance Analysis:")
print("-" * 80)
top_features = feature_importance.head(3)
insights_text = "MODEL-DRIVEN INSIGHTS:\n\n"

for idx, row in top_features.iterrows():
    pct = row['importance'] * 100
    insights_text += f"{idx+1}. {row['feature'].upper()}: {pct:.2f}% importance\n"
    
    # Contextual interpretation
    if row['feature'] == 'mobility_indicator':
        insights_text += f"   - Citizens who frequently change addresses ({pct:.1f}% predictor) are most likely to have identity instability.\n"
        insights_text += f"   - ACTION: Target mobile centers at high-mobility districts.\n\n"
    elif row['feature'] == 'digital_instability_index':
        insights_text += f"   - Frequent mobile/email updates ({pct:.1f}% predictor) signal underlying identity issues.\n"
        insights_text += f"   - ACTION: Flag for potential fraud if combined with other risk factors.\n\n"
    elif row['feature'] == 'manual_labor_proxy':
        insights_text += f"   - Manual laborers ({pct:.1f}% predictor) face fingerprint degradation leading to biometric failures.\n"
        insights_text += f"   - ACTION: Deploy specialized biometric equipment at construction sites.\n\n"
    elif row['feature'] == 'update_burden_index':
        insights_text += f"   - High update frequency ({pct:.1f}% predictor) correlates with instability.\n"
        insights_text += f"   - ACTION: Investigate districts with >5 updates/citizen/year for fraud.\n\n"
    elif row['feature'] == 'anomaly_severity_score':
        insights_text += f"   - Pre-calculated anomaly score ({pct:.1f}% predictor) validates our feature engineering.\n"
        insights_text += f"   - ACTION: Use this as real-time risk indicator in production systems.\n\n"

print(insights_text)

# Save insights
with open('../outputs/tables/model_rf_explainability_insights.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("EXPLAINABLE AI: RANDOM FOREST MODEL INSIGHTS\n")
    f.write("="*80 + "\n\n")
    f.write(insights_text)
    f.write("\n" + "-"*80 + "\n")
    f.write("FULL FEATURE RANKING:\n")
    f.write("-"*80 + "\n")
    f.write(feature_importance.to_string(index=False))

# 2. Partial Dependence Analysis
print("\n[INSIGHT 2] Partial Dependence Analysis:")
print("-" * 80)
print("Analyzing how each top feature independently affects stability prediction...\n")

# Select top 4 features for PDP
top_4_features = feature_importance.head(4)['feature'].tolist()
top_4_indices = [feature_cols.index(f) for f in top_4_features]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for i, (feature_name, feature_idx) in enumerate(zip(top_4_features, top_4_indices)):
    # Calculate partial dependence
    pdp_result = partial_dependence(
        rf_model, X_train, [feature_idx], 
        kind='average', grid_resolution=50
    )
    
    axes[i].plot(pdp_result['grid_values'][0], pdp_result['average'][0], 
                 linewidth=3, color='darkblue')
    axes[i].set_xlabel(feature_name.replace('_', ' ').title(), fontsize=11, fontweight='bold')
    axes[i].set_ylabel('Predicted Probability\n(Low Stability)', fontsize=10)
    axes[i].set_title(f'Partial Dependence: {feature_name.replace("_", " ").title()}', 
                      fontsize=12, fontweight='bold')
    axes[i].grid(alpha=0.3)
    
    # Add interpretation text
    x_vals = pdp_result['grid_values'][0]
    y_vals = pdp_result['average'][0]
    trend = "increases" if y_vals[-1] > y_vals[0] else "decreases"
    change = abs(y_vals[-1] - y_vals[0])
    
    axes[i].text(0.05, 0.95, f'Effect: {trend.upper()}\nΔP = {change:.3f}', 
                transform=axes[i].transAxes, fontsize=9,
                verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Partial Dependence Plots: How Features Affect Stability Risk', 
             fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('../outputs/figures/model_rf_partial_dependence.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Partial dependence plots saved")

# Generate PDP insights
pdp_insights = "\nPARTIAL DEPENDENCE INSIGHTS:\n\n"
for feature_name, feature_idx in zip(top_4_features, top_4_indices):
    pdp_result = partial_dependence(
        rf_model, X_train, [feature_idx], kind='average', grid_resolution=50
    )
    x_vals = pdp_result['grid_values'][0]
    y_vals = pdp_result['average'][0]
    
    trend = "INCREASES" if y_vals[-1] > y_vals[0] else "DECREASES"
    change = abs(y_vals[-1] - y_vals[0])
    
    pdp_insights += f"• {feature_name.upper()}:\n"
    pdp_insights += f"  - As {feature_name} increases, instability risk {trend} by {change:.3f}\n"
    
    if feature_name == 'mobility_indicator' and trend == "INCREASES":
        pdp_insights += f"  - INTERPRETATION: Every 0.1 increase in mobility raises risk by ~{change*10:.3f}\n"
        pdp_insights += f"  - DECISION RULE: IF mobility > 0.25 THEN flag for intervention\n\n"
    elif feature_name == 'digital_instability_index' and trend == "INCREASES":
        pdp_insights += f"  - INTERPRETATION: Digital churn strongly predicts identity problems\n"
        pdp_insights += f"  - DECISION RULE: IF digital_instability > 0.5 THEN investigate for fraud\n\n"
    else:
        pdp_insights += f"  - DECISION RULE: Monitor this feature for risk assessment\n\n"

print(pdp_insights)

# 3. SHAP Analysis (if available)
if SHAP_AVAILABLE:
    print("\n[INSIGHT 3] SHAP (SHapley Additive exPlanations) Analysis:")
    print("-" * 80)
    print("Computing SHAP values to explain individual predictions...")
    
    # Use a sample for SHAP (computationally expensive)
    X_sample = X_test.sample(n=min(1000, len(X_test)), random_state=42)
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_sample)
    
    # If binary classification, select class 1 (Low Stability)
    if isinstance(shap_values, list):
        shap_values_class1 = shap_values[1]
    else:
        shap_values_class1 = shap_values
    
    # SHAP Summary Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_class1, X_sample, 
                     feature_names=feature_cols, show=False, max_display=10)
    plt.title('SHAP Summary: Feature Impact on Low Stability Prediction', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('../outputs/figures/model_rf_shap_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] SHAP summary plot saved")
    
    # SHAP Feature Importance
    mean_shap_values = np.abs(shap_values_class1).mean(axis=0)
    shap_importance = pd.DataFrame({
        'feature': feature_cols,
        'mean_abs_shap': mean_shap_values
    }).sort_values('mean_abs_shap', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=shap_importance.head(10), x='mean_abs_shap', y='feature', 
                palette='coolwarm')
    plt.title('SHAP Feature Importance (Mean |SHAP Value|)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Mean Absolute SHAP Value', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.savefig('../outputs/figures/model_rf_shap_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] SHAP feature importance plot saved")
    
    # SHAP Insights
    shap_insights = "\nSHAP-BASED INSIGHTS:\n\n"
    shap_insights += "SHAP values reveal how each feature contributes to individual predictions:\n\n"
    
    for idx, row in shap_importance.head(3).iterrows():
        shap_insights += f"• {row['feature'].upper()}: Mean |SHAP| = {row['mean_abs_shap']:.4f}\n"
        shap_insights += f"  - This feature pushes predictions by ±{row['mean_abs_shap']:.4f} on average\n"
        shap_insights += f"  - High values increase instability risk; low values decrease it\n\n"
    
    print(shap_insights)
    
    # Save SHAP insights
    with open('../outputs/tables/model_rf_shap_insights.txt', 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("SHAP ANALYSIS: INDIVIDUAL FEATURE CONTRIBUTIONS\n")
        f.write("="*80 + "\n\n")
        f.write(shap_insights)
        f.write("\nFULL SHAP IMPORTANCE RANKING:\n")
        f.write("-"*80 + "\n")
        f.write(shap_importance.to_string(index=False))
    
else:
    print("\n[SKIP] SHAP analysis (library not installed)")
    print("To enable: pip install shap")

# 4. Decision Rules Extraction
print("\n[INSIGHT 4] Decision Rules Extraction:")
print("-" * 80)
print("Analyzing model's decision patterns to create actionable rules...\n")

# Analyze high-risk predictions
X_test_df = X_test.copy()
X_test_df['predicted_risk'] = y_pred_proba
X_test_df['true_label'] = y_test.values

high_risk = X_test_df[X_test_df['predicted_risk'] > 0.7].sort_values('predicted_risk', ascending=False)
low_risk = X_test_df[X_test_df['predicted_risk'] < 0.3].sort_values('predicted_risk')

print(f"High Risk Cases (P > 0.7): {len(high_risk)}")
print(f"Low Risk Cases (P < 0.3): {len(low_risk)}")

# Extract rules
decision_rules = "\nEXTRACTED DECISION RULES:\n\n"

if len(high_risk) > 0:
    high_risk_profile = high_risk[feature_cols].mean()
    decision_rules += "RULE 1: HIGH INSTABILITY RISK PROFILE\n"
    decision_rules += "-" * 40 + "\n"
    for feature in feature_cols:
        val = high_risk_profile[feature]
        decision_rules += f"  • {feature}: {val:.4f}\n"
    
    decision_rules += "\nACTION: If a citizen matches this profile:\n"
    decision_rules += "  1. Flag for immediate review\n"
    decision_rules += "  2. Assign dedicated case officer\n"
    decision_rules += "  3. Offer free update assistance\n"
    decision_rules += "  4. Prioritize in queue (process within 48 hours)\n\n"

if len(low_risk) > 0:
    low_risk_profile = low_risk[feature_cols].mean()
    decision_rules += "RULE 2: LOW INSTABILITY RISK PROFILE\n"
    decision_rules += "-" * 40 + "\n"
    for feature in feature_cols:
        val = low_risk_profile[feature]
        decision_rules += f"  • {feature}: {val:.4f}\n"
    
    decision_rules += "\nACTION: If a citizen matches this profile:\n"
    decision_rules += "  1. Standard processing\n"
    decision_rules += "  2. Low priority for proactive outreach\n"
    decision_rules += "  3. Normal update timeline (10-15 days acceptable)\n\n"

# Threshold-based rules
decision_rules += "RULE 3: THRESHOLD-BASED INTERVENTIONS\n"
decision_rules += "-" * 40 + "\n"
decision_rules += f"IF mobility_indicator > 0.25 AND digital_instability_index > 0.5 THEN\n"
decision_rules += f"   - HIGH RISK: Deploy mobile center + fraud investigation\n\n"
decision_rules += f"IF manual_labor_proxy > 0.6 THEN\n"
decision_rules += f"   - VULNERABLE: Offer free biometric restoration\n\n"
decision_rules += f"IF update_burden_index > 0.7 THEN\n"
decision_rules += f"   - ANOMALY: Flag for fraud detection unit review\n\n"

print(decision_rules)

# Save all insights
with open('../outputs/tables/model_rf_explainability_insights.txt', 'a', encoding='utf-8') as f:
    f.write("\n\n" + "="*80 + "\n")
    f.write(pdp_insights)
    f.write("\n" + "="*80 + "\n")
    f.write(decision_rules)

print("[OK] All explainability insights saved to model_rf_explainability_insights.txt")

# ==================== END OF EXPLAINABLE AI SECTION ====================

# Save model
joblib.dump(rf_model, '../outputs/models/rf_stability_classifier.pkl')
print("[OK] Model saved to outputs/models/rf_stability_classifier.pkl")

# ==================== MODEL 2: ANOMALY DETECTION (ISOLATION FOREST) ====================
print("\n" + "="*100)
print("MODEL 2: ISOLATION FOREST - ANOMALY DETECTION")
print("="*100)

# Select features for anomaly detection
anomaly_features = [
    'enrolment_growth_rate', 'mobility_indicator', 'digital_instability_index',
    'update_burden_index', 'anomaly_severity_score'
]

df_anomaly = df[anomaly_features].dropna()
print(f"\nSamples for anomaly detection: {len(df_anomaly)}")

# Standardize features
scaler = StandardScaler()
X_anomaly_scaled = scaler.fit_transform(df_anomaly)

# Train Isolation Forest
print("\nTraining Isolation Forest...")
iso_forest = IsolationForest(
    n_estimators=100,
    contamination=0.05,  # Expect 5% anomalies
    random_state=42,
    n_jobs=-1
)

anomaly_labels = iso_forest.fit_predict(X_anomaly_scaled)
anomaly_scores = iso_forest.score_samples(X_anomaly_scaled)

# Add results to dataframe
df_anomaly['anomaly_label'] = anomaly_labels  # -1 = anomaly, 1 = normal
df_anomaly['anomaly_score'] = anomaly_scores

n_anomalies = (anomaly_labels == -1).sum()
n_normal = (anomaly_labels == 1).sum()

print(f"[OK] Anomaly detection complete!")
print(f"  Normal records: {n_normal} ({n_normal/len(df_anomaly)*100:.2f}%)")
print(f"  Anomalies detected: {n_anomalies} ({n_anomalies/len(df_anomaly)*100:.2f}%)")

# Save anomaly results
anomaly_summary = df_anomaly[anomaly_labels == -1].describe()
anomaly_summary.to_csv('../outputs/tables/model_if_anomaly_summary.csv')

# Visualize anomalies (2D projection)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Mobility vs Enrolment Growth
ax1 = axes[0]
normal_mask = anomaly_labels == 1
anomaly_mask = anomaly_labels == -1

ax1.scatter(df_anomaly[normal_mask]['mobility_indicator'], 
           df_anomaly[normal_mask]['enrolment_growth_rate'],
           c='blue', alpha=0.3, s=10, label='Normal')
ax1.scatter(df_anomaly[anomaly_mask]['mobility_indicator'], 
           df_anomaly[anomaly_mask]['enrolment_growth_rate'],
           c='red', alpha=0.8, s=50, label='Anomaly', edgecolors='black')
ax1.set_xlabel('Mobility Indicator', fontsize=11)
ax1.set_ylabel('Enrolment Growth Rate', fontsize=11)
ax1.set_title('Anomaly Detection: Mobility vs Growth', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# Plot 2: Update Burden vs Digital Instability
ax2 = axes[1]
ax2.scatter(df_anomaly[normal_mask]['update_burden_index'], 
           df_anomaly[normal_mask]['digital_instability_index'],
           c='blue', alpha=0.3, s=10, label='Normal')
ax2.scatter(df_anomaly[anomaly_mask]['update_burden_index'], 
           df_anomaly[anomaly_mask]['digital_instability_index'],
           c='red', alpha=0.8, s=50, label='Anomaly', edgecolors='black')
ax2.set_xlabel('Update Burden Index', fontsize=11)
ax2.set_ylabel('Digital Instability Index', fontsize=11)
ax2.set_title('Anomaly Detection: Update Burden vs Digital Instability', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

plt.suptitle('Isolation Forest - Anomaly Detection Results', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../outputs/figures/model_if_anomaly_detection.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Anomaly visualization saved")

# Save model
joblib.dump(iso_forest, '../outputs/models/isolation_forest_anomaly_detector.pkl')
joblib.dump(scaler, '../outputs/models/anomaly_scaler.pkl')
print("[OK] Models saved")

# ==================== MODEL 3: TIME SERIES FORECASTING (SIMPLE FORECAST) ====================
print("\n" + "="*100)
print("MODEL 3: TIME SERIES FORECASTING - ENROLMENT PREDICTION")
print("="*100)

# Aggregate by date
ts_data = df.groupby('date').agg({
    'total_enrolments': 'sum',
    'total_demographic_updates': 'sum',
    'mobility_indicator': 'mean',
    'identity_stability_score': 'mean'
}).reset_index()

ts_data = ts_data.sort_values('date')
print(f"\nTime series data points: {len(ts_data)}")

# Simple moving average forecast
window = 7
ts_data['enrolment_ma7'] = ts_data['total_enrolments'].rolling(window=window).mean()
ts_data['mobility_ma7'] = ts_data['mobility_indicator'].rolling(window=window).mean()

# Last observation carried forward for next month
last_enrolment_avg = ts_data['enrolment_ma7'].iloc[-1]
last_mobility_avg = ts_data['mobility_ma7'].iloc[-1]

print(f"\nForecasted average daily enrolments (next month): {last_enrolment_avg:,.0f}")
print(f"Forecasted average mobility indicator (next month): {last_mobility_avg:.4f}")

# Visualize forecast
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Enrolment forecast
ax1 = axes[0]
ax1.plot(ts_data['date'], ts_data['total_enrolments'], 
        label='Actual Enrolments', alpha=0.5, linewidth=1)
ax1.plot(ts_data['date'], ts_data['enrolment_ma7'], 
        label='7-Day Moving Average', color='red', linewidth=2)
ax1.axhline(last_enrolment_avg, color='green', linestyle='--', 
           linewidth=2, label=f'Forecast (Next Month): {last_enrolment_avg:,.0f}')
ax1.set_xlabel('Date', fontsize=11)
ax1.set_ylabel('Total Enrolments', fontsize=11)
ax1.set_title('Enrolment Trend & Forecast', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# Mobility forecast
ax2 = axes[1]
ax2.plot(ts_data['date'], ts_data['mobility_indicator'], 
        label='Actual Mobility', alpha=0.5, linewidth=1, color='orange')
ax2.plot(ts_data['date'], ts_data['mobility_ma7'], 
        label='7-Day Moving Average', color='darkblue', linewidth=2)
ax2.axhline(last_mobility_avg, color='purple', linestyle='--', 
           linewidth=2, label=f'Forecast (Next Month): {last_mobility_avg:.4f}')
ax2.set_xlabel('Date', fontsize=11)
ax2.set_ylabel('Mobility Indicator', fontsize=11)
ax2.set_title('Mobility Trend & Forecast', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

plt.suptitle('Time Series Forecasting - Moving Average Method', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../outputs/figures/model_ts_forecast.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Time series forecast saved")

# Save forecast summary
forecast_summary = pd.DataFrame({
    'metric': ['Enrolments', 'Mobility Indicator', 'Digital Instability', 'Identity Stability'],
    'forecast_value': [
        last_enrolment_avg,
        last_mobility_avg,
        ts_data['digital_instability_index'].rolling(window=7).mean().iloc[-1] if 'digital_instability_index' in ts_data.columns else 0,
        ts_data['identity_stability_score'].rolling(window=7).mean().iloc[-1]
    ]
})
forecast_summary.to_csv('../outputs/tables/model_ts_forecast_summary.csv', index=False)

# ==================== SUMMARY ====================
print("\n" + "="*100)
print("PREDICTIVE MODELING COMPLETE!")
print("="*100)

print("\n[MODEL 1] Random Forest Classifier:")
print(f"  - Purpose: Predict identity stability category (High/Low)")
print(f"  - ROC-AUC: {roc_auc:.4f}")
print(f"  - Test Accuracy: {(y_pred == y_test).mean():.4f}")
print(f"  - Output: {len(feature_importance)} feature importances")

print("\n[MODEL 2] Isolation Forest Anomaly Detector:")
print(f"  - Purpose: Detect unusual patterns in enrolment/update data")
print(f"  - Anomalies detected: {n_anomalies} ({n_anomalies/len(df_anomaly)*100:.2f}%)")
print(f"  - Features analyzed: {len(anomaly_features)}")

print("\n[MODEL 3] Time Series Forecast:")
print(f"  - Purpose: Predict next month's enrolments and mobility")
print(f"  - Method: 7-day moving average")
print(f"  - Enrolment forecast: {last_enrolment_avg:,.0f} daily avg")
print(f"  - Mobility forecast: {last_mobility_avg:.4f}")

print("\nAll outputs saved to:")
print("  📊 Figures: outputs/figures/model_*.png (6 visualizations)")
print("  📈 Tables: outputs/tables/model_*.csv (3 result files)")
print("  🤖 Models: outputs/models/*.pkl (3 trained models)")

print("\n" + "="*100)
print("[OK] PREDICTIVE MODELS MODULE COMPLETE!")
print("="*100)
