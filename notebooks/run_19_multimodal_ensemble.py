"""
Multi-Modal Ensemble Architecture
===================================
3 specialized fraud detectors with meta-learner stacking.

Innovation: Most teams = single model. This = ensemble of specialized detectors.
Differentiation: ⭐⭐⭐⭐ (Shows ML sophistication)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🧠 MULTI-MODAL ENSEMBLE ARCHITECTURE")
print("="*80)

# Load data (MEMORY OPTIMIZED - use subset)
print("\n📊 Loading data...")

# Only load necessary columns for training
necessary_cols = [
    'date', 'district', 'state',
    # Demographic
    'address_updates', 'mobile_updates', 'name_updates', 
    'dob_updates', 'gender_updates',
    'total_demographic_updates', 'demographic_updates_5_17', 'demographic_updates_18_plus',
    'demographic_intensity', 'mobile_intensity', 'name_change_intensity',
    # Biometric
    'fingerprint_updates', 'iris_updates', 'photo_updates',
    'total_biometric_updates', 'biometric_updates_5_17', 'biometric_updates_18_plus',
    'biometric_intensity', 'fingerprint_quality', 'iris_quality',
    'avg_biometric_quality', 'biometric_failures',
    # Enrolment
    'total_enrolments', 'age_0_5', 'age_5_17', 'age_18_greater',
    # Behavioral
    'update_frequency', 'days_since_last_update', 'update_burst_intensity',
    'weekend_update_ratio', 'night_update_ratio', 'cross_district_updates',
    'rapid_succession_updates', 'time_gap_variance', 'update_consistency_score'
]

# Load only existing columns
df_full = pd.read_csv('data/processed/aadhaar_extended_features_clean.csv', nrows=1)
available_cols = [col for col in necessary_cols if col in df_full.columns]
df = pd.read_csv('data/processed/aadhaar_extended_features_clean.csv', usecols=available_cols)
df['date'] = pd.to_datetime(df['date'])

# Load anomaly labels efficiently (only needed columns)
print("   Loading anomaly labels...")
anomaly_results = pd.read_csv('outputs/anomaly_detection_results.csv', 
                              usecols=['date', 'district', 'is_anomaly'])
anomaly_results['date'] = pd.to_datetime(anomaly_results['date'])

# Add anomaly column directly
df = df.merge(anomaly_results, on=['date', 'district'], how='inner')

print(f"   Loaded {len(df):,} records")
print(f"   Anomalies: {df['is_anomaly'].sum():,} ({df['is_anomaly'].mean()*100:.2f}%)")

# ===========================
# 1. DEMOGRAPHIC FRAUD DETECTOR
# ===========================
print("\n👥 Building Demographic Fraud Detector...")

# Features: Unusual demographic update patterns (only use existing columns)
demographic_features_all = [
    'address_updates', 'mobile_updates', 'name_updates', 
    'dob_updates', 'gender_updates',
    'total_demographic_updates', 'demographic_updates_5_17', 'demographic_updates_18_plus',
    'demographic_intensity', 'mobile_intensity', 'name_change_intensity',
    'age_0_5', 'age_5_17', 'age_18_greater'
]
demographic_features = [f for f in demographic_features_all if f in df.columns]

# Add derived features
df['demo_update_rate'] = df['total_demographic_updates'] / (df['total_enrolments'] + 1)
df['mobile_update_rate'] = df['mobile_updates'] / (df['total_demographic_updates'] + 1)
df['name_update_rate'] = df['name_updates'] / (df['total_demographic_updates'] + 1)
df['address_update_rate'] = df['address_updates'] / (df['total_demographic_updates'] + 1)

demographic_features += ['demo_update_rate', 'mobile_update_rate', 'name_update_rate', 'address_update_rate']

# Train demographic detector
X_demo = df[demographic_features].fillna(0)
y = df['is_anomaly']

X_train_demo, X_test_demo, y_train, y_test = train_test_split(
    X_demo, y, test_size=0.3, random_state=42, stratify=y
)

scaler_demo = StandardScaler()
X_train_demo_scaled = scaler_demo.fit_transform(X_train_demo)
X_test_demo_scaled = scaler_demo.transform(X_test_demo)

demo_detector = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=100,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)
demo_detector.fit(X_train_demo_scaled, y_train)

demo_preds = demo_detector.predict_proba(X_test_demo_scaled)[:, 1]
demo_roc_auc = roc_auc_score(y_test, demo_preds)

print(f"   ✅ Demographic Detector trained")
print(f"   ROC-AUC: {demo_roc_auc:.4f}")
print(f"   Top features: {', '.join([demographic_features[i] for i in np.argsort(demo_detector.feature_importances_)[-5:][::-1]])}")

# ===========================
# 2. BIOMETRIC FRAUD DETECTOR
# ===========================
print("\n🔐 Building Biometric Fraud Detector...")

# Features: Unusual biometric update patterns (only use existing columns)
biometric_features_all = [
    'fingerprint_updates', 'iris_updates', 'photo_updates',
    'total_biometric_updates', 'biometric_updates_5_17', 'biometric_updates_18_plus',
    'biometric_intensity', 'fingerprint_quality', 'iris_quality',
    'avg_biometric_quality', 'biometric_failures'
]
biometric_features = [f for f in biometric_features_all if f in df.columns]

# Add derived features
df['bio_update_rate'] = df['total_biometric_updates'] / (df['total_enrolments'] + 1)
if 'fingerprint_updates' in df.columns:
    df['fingerprint_update_rate'] = df['fingerprint_updates'] / (df['total_biometric_updates'] + 1)
    biometric_features.append('fingerprint_update_rate')
if 'iris_updates' in df.columns:
    df['iris_update_rate'] = df['iris_updates'] / (df['total_biometric_updates'] + 1)
    biometric_features.append('iris_update_rate')
if 'biometric_failures' in df.columns:
    df['biometric_failure_rate'] = df['biometric_failures'] / (df['total_biometric_updates'] + 1)
    biometric_features.append('biometric_failure_rate')

biometric_features.append('bio_update_rate')

# Train biometric detector
X_bio = df[biometric_features].fillna(0)

X_train_bio, X_test_bio, _, _ = train_test_split(
    X_bio, y, test_size=0.3, random_state=42, stratify=y
)

scaler_bio = StandardScaler()
X_train_bio_scaled = scaler_bio.fit_transform(X_train_bio)
X_test_bio_scaled = scaler_bio.transform(X_test_bio)

bio_detector = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=100,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)
bio_detector.fit(X_train_bio_scaled, y_train)

bio_preds = bio_detector.predict_proba(X_test_bio_scaled)[:, 1]
bio_roc_auc = roc_auc_score(y_test, bio_preds)

print(f"   ✅ Biometric Detector trained")
print(f"   ROC-AUC: {bio_roc_auc:.4f}")
print(f"   Top features: {', '.join([biometric_features[i] for i in np.argsort(bio_detector.feature_importances_)[-5:][::-1]])}")

# ===========================
# 3. BEHAVIORAL FRAUD DETECTOR
# ===========================
print("\n🎯 Building Behavioral Fraud Detector...")

# Features: Unusual temporal and frequency patterns (only use existing columns)
behavioral_features_all = [
    'update_frequency', 'days_since_last_update', 'update_burst_intensity',
    'weekend_update_ratio', 'night_update_ratio', 'cross_district_updates',
    'rapid_succession_updates', 'time_gap_variance', 'update_consistency_score',
    'cluster', 'anomaly_score'
]
behavioral_features = [f for f in behavioral_features_all if f in df.columns]

# If too few behavioral features, create basic ones
if len(behavioral_features) < 3:
    print("   Creating basic behavioral features...")
    df['total_updates'] = df['total_enrolments'] + df['total_biometric_updates'] + df['total_demographic_updates']
    df['update_intensity'] = df['total_updates'] / df['total_updates'].quantile(0.95)
    df['bio_demo_ratio'] = df['total_biometric_updates'] / (df['total_demographic_updates'] + 1)
    behavioral_features = ['total_updates', 'update_intensity', 'bio_demo_ratio']

# Train behavioral detector
X_behav = df[behavioral_features].fillna(0)

X_train_behav, X_test_behav, _, _ = train_test_split(
    X_behav, y, test_size=0.3, random_state=42, stratify=y
)

scaler_behav = StandardScaler()
X_train_behav_scaled = scaler_behav.fit_transform(X_train_behav)
X_test_behav_scaled = scaler_behav.transform(X_test_behav)

behav_detector = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
behav_detector.fit(X_train_behav_scaled, y_train)

behav_preds = behav_detector.predict_proba(X_test_behav_scaled)[:, 1]
behav_roc_auc = roc_auc_score(y_test, behav_preds)

print(f"   ✅ Behavioral Detector trained")
print(f"   ROC-AUC: {behav_roc_auc:.4f}")
print(f"   Top features: {', '.join([behavioral_features[i] for i in np.argsort(behav_detector.feature_importances_)[-3:][::-1]])}")

# ===========================
# 4. META-LEARNER (ENSEMBLE STACKING)
# ===========================
print("\n🧩 Building Meta-Learner (Ensemble Stacking)...")

# Create meta-features from base model predictions
meta_features_train = np.column_stack([
    demo_detector.predict_proba(X_train_demo_scaled)[:, 1],
    bio_detector.predict_proba(X_train_bio_scaled)[:, 1],
    behav_detector.predict_proba(X_train_behav_scaled)[:, 1]
])

meta_features_test = np.column_stack([
    demo_preds,
    bio_preds,
    behav_preds
])

# Train meta-learner
meta_learner = LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight='balanced'
)
meta_learner.fit(meta_features_train, y_train)

# Final predictions
final_preds = meta_learner.predict_proba(meta_features_test)[:, 1]
final_roc_auc = roc_auc_score(y_test, final_preds)

print(f"   ✅ Meta-Learner trained")
print(f"   Ensemble ROC-AUC: {final_roc_auc:.4f}")
print(f"   Weights: Demo={meta_learner.coef_[0][0]:.3f}, Bio={meta_learner.coef_[0][1]:.3f}, Behav={meta_learner.coef_[0][2]:.3f}")

# ===========================
# 5. MODEL COMPARISON
# ===========================
print("\n" + "="*80)
print("📊 MODEL PERFORMANCE COMPARISON")
print("="*80)

models_comparison = pd.DataFrame({
    'Model': ['Demographic Detector', 'Biometric Detector', 'Behavioral Detector', 'Ensemble (Meta-Learner)'],
    'ROC-AUC': [demo_roc_auc, bio_roc_auc, behav_roc_auc, final_roc_auc],
    'Features Used': [len(demographic_features), len(biometric_features), len(behavioral_features), 3]
})

print(models_comparison.to_string(index=False))

# ===========================
# 6. CONFIDENCE DECOMPOSITION
# ===========================
print("\n📈 Analyzing confidence decomposition...")

# For each test sample, show which model contributed most
decomposition_samples = []
for i in range(min(100, len(meta_features_test))):
    demo_score = demo_preds[i]
    bio_score = bio_preds[i]
    behav_score = behav_preds[i]
    final_score = final_preds[i]
    
    # Determine dominant signal
    scores = {'demographic': demo_score, 'biometric': bio_score, 'behavioral': behav_score}
    dominant_signal = max(scores, key=scores.get)
    
    decomposition_samples.append({
        'demo_score': float(demo_score),
        'bio_score': float(bio_score),
        'behav_score': float(behav_score),
        'final_score': float(final_score),
        'dominant_signal': dominant_signal,
        'actual_label': int(y_test.iloc[i])
    })

print(f"   ✅ Analyzed confidence decomposition for {len(decomposition_samples)} samples")

# Count dominant signals
dominant_counts = pd.Series([s['dominant_signal'] for s in decomposition_samples]).value_counts()
print(f"\n   Dominant Detection Signals:")
for signal, count in dominant_counts.items():
    print(f"   - {signal.capitalize()}: {count} samples ({count/len(decomposition_samples)*100:.1f}%)")

# ===========================
# 7. SAVE MODELS
# ===========================
print("\n💾 Saving ensemble models...")

models_dir = Path('outputs/models')
models_dir.mkdir(exist_ok=True, parents=True)

# Save all components
joblib.dump(demo_detector, models_dir / 'ensemble_demographic_detector.pkl')
joblib.dump(bio_detector, models_dir / 'ensemble_biometric_detector.pkl')
joblib.dump(behav_detector, models_dir / 'ensemble_behavioral_detector.pkl')
joblib.dump(meta_learner, models_dir / 'ensemble_meta_learner.pkl')

joblib.dump(scaler_demo, models_dir / 'ensemble_scaler_demographic.pkl')
joblib.dump(scaler_bio, models_dir / 'ensemble_scaler_biometric.pkl')
joblib.dump(scaler_behav, models_dir / 'ensemble_scaler_behavioral.pkl')

print(f"   ✅ Saved 7 model components")

# Save feature lists
feature_config = {
    'demographic_features': demographic_features,
    'biometric_features': biometric_features,
    'behavioral_features': behavioral_features
}

with open(models_dir / 'ensemble_features.json', 'w') as f:
    json.dump(feature_config, f, indent=2)

print(f"   ✅ Saved feature configuration")

# Save model comparison
models_comparison.to_csv('outputs/ensemble_model_comparison.csv', index=False)
print(f"   ✅ Saved model comparison")

# Save decomposition analysis
with open('outputs/confidence_decomposition_samples.json', 'w') as f:
    json.dump(decomposition_samples, f, indent=2)
print(f"   ✅ Saved confidence decomposition")

# ===========================
# 8. SUMMARY
# ===========================
print("\n" + "="*80)
print("✅ MULTI-MODAL ENSEMBLE ARCHITECTURE COMPLETE")
print("="*80)

print(f"\n🎯 Ensemble Performance:")
print(f"   Final ROC-AUC: {final_roc_auc:.4f}")
print(f"   Improvement over best single model: {(final_roc_auc - max(demo_roc_auc, bio_roc_auc, behav_roc_auc))*100:.2f}%")

print(f"\n🧠 Architecture:")
print(f"   Demographic Detector: RandomForest ({len(demographic_features)} features)")
print(f"   Biometric Detector: RandomForest ({len(biometric_features)} features)")
print(f"   Behavioral Detector: GradientBoosting ({len(behavioral_features)} features)")
print(f"   Meta-Learner: LogisticRegression (3 base model predictions)")

print("\n📁 Outputs saved:")
print("  - outputs/models/ensemble_demographic_detector.pkl")
print("  - outputs/models/ensemble_biometric_detector.pkl")
print("  - outputs/models/ensemble_behavioral_detector.pkl")
print("  - outputs/models/ensemble_meta_learner.pkl")
print("  - outputs/models/ensemble_features.json")
print("  - outputs/ensemble_model_comparison.csv")
print("  - outputs/confidence_decomposition_samples.json")

print("\n🚀 Innovation Impact: ⭐⭐⭐⭐")
print("   Differentiation: Shows advanced ML expertise beyond single models!")
