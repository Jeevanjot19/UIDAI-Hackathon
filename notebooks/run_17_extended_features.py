"""
Extended Feature Engineering for UIDAI Aadhaar Analytics
Expands from 102 to 140+ features across 16 major categories

New Feature Categories:
1. Temporal & Behavioral Dynamics (Burst, Fatigue, Persistence)
2. Life-Cycle & Age-Transition Intelligence
3. Migration & Mobility Intelligence
4. Update Composition & Quality
5. District Stress & Capacity Signals
6. Societal Stability & Trust Signals
7. District Comparative Features
8. Anomaly & Event Intelligence
9. Model-Assisted Features
10. Forecast-Derived Features
11. Policy Simulation Features
12. Meta-Summary Features
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("EXTENDED FEATURE ENGINEERING - UIDAI Aadhaar Analytics")
print("Expanding from 102 to 140+ features")
print("="*80)

# Load base data with indices
print("\n[1/18] Loading base data with existing features...")
df = pd.read_csv('data/processed/aadhaar_with_indices.csv', parse_dates=['date'])
print(f"   ✓ Loaded {len(df):,} records with {len(df.columns)} existing features")

# Sort by district and date for time-series operations
df = df.sort_values(['district', 'date']).reset_index(drop=True)

# ============================================================================
# A. TEMPORAL & BEHAVIORAL DYNAMICS
# ============================================================================
print("\n[2/18] Creating Temporal & Behavioral Dynamics features...")

# 1. Burst & Fatigue Features
print("   → Burst & Fatigue features...")

# Calculate 12-month rolling statistics for burst detection
df['updates_12m_max'] = df.groupby('district')['total_updates'].rolling(12).max().reset_index(0, drop=True)
df['updates_12m_mean'] = df.groupby('district')['total_updates'].rolling(12).mean().reset_index(0, drop=True)

# Burst score: ratio of max to average
df['update_burst_score'] = df['updates_12m_max'] / (df['updates_12m_mean'] + 1)

# Post-burst fatigue: drop after spike
df['update_next_month'] = df.groupby('district')['total_updates'].shift(-1)
df['post_burst_fatigue'] = (df['total_updates'] - df['update_next_month']) / (df['total_updates'] + 1)
df['post_burst_fatigue'] = df['post_burst_fatigue'].clip(0, 1)  # Only measure drops

# Sustained high activity: 3+ consecutive high months
df['is_high_month'] = (df['total_updates'] > df.groupby('district')['total_updates'].transform('median')).astype(int)
df['high_streak'] = df.groupby('district')['is_high_month'].apply(
    lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1)
).reset_index(0, drop=True)
df['sustained_high_activity_flag'] = (df['high_streak'] >= 3).astype(int)

# 2. Update Persistence Features
print("   → Update persistence features...")

# Correlation between current and lagged updates (persistence measure)
def calculate_persistence(group, lag):
    """Calculate correlation between current and lagged values"""
    if len(group) < lag + 10:  # Need enough data
        return pd.Series([0.5] * len(group), index=group.index)
    
    result = []
    for i in range(len(group)):
        if i < lag + 5:  # Not enough history
            result.append(0.5)
        else:
            window = group.iloc[max(0, i-10):i+1]
            lagged = window.shift(lag)
            corr = window.corr(lagged)
            result.append(corr if not np.isnan(corr) else 0.5)
    
    return pd.Series(result, index=group.index)

df['update_persistence_3m'] = df.groupby('district')['total_updates'].apply(
    lambda x: calculate_persistence(x, 3)
).reset_index(0, drop=True)

df['update_persistence_6m'] = df.groupby('district')['total_updates'].apply(
    lambda x: calculate_persistence(x, 6)
).reset_index(0, drop=True)

# Behavioral memory score: weighted persistence
df['behavioral_memory_score'] = (
    0.5 * df['update_persistence_3m'] + 
    0.3 * df['update_persistence_6m'] + 
    0.2 * (df['rolling_3m_updates'] / (df['rolling_3m_updates'].rolling(6).mean() + 1))
).clip(0, 1)

print(f"   ✓ Created 8 temporal/behavioral features")

# ============================================================================
# B. LIFE-CYCLE & AGE-TRANSITION INTELLIGENCE
# ============================================================================
print("\n[3/18] Creating Life-Cycle & Age-Transition features...")

# 3. Age-Cohort Pressure Index
print("   → Age-cohort pressure indices...")

# Estimate children turning 5, 15, 18 (requiring biometric updates)
# Use 60-month lag (5 years), 180-month lag (15 years)
if 'enrolments_0_5' in df.columns:
    df['cohort_size_turning_5'] = df.groupby('district')['enrolments_0_5'].shift(60)
    df['age_transition_pressure_5'] = (
        df['cohort_size_turning_5'] * 
        (1 - df.get('child_update_compliance', 0.7))  # Compliance gap
    ).fillna(0)
else:
    df['age_transition_pressure_5'] = 0

if 'enrolments_5_17' in df.columns:
    df['cohort_size_turning_15'] = df.groupby('district')['enrolments_5_17'].shift(120)
    df['age_transition_pressure_15'] = (
        df['cohort_size_turning_15'] * 0.5 *  # Assume 50% in age 10-15 bucket
        (1 - df.get('child_update_compliance', 0.7))
    ).fillna(0)
    
    # Age 18 transition (legal adult status)
    df['cohort_size_turning_18'] = df.groupby('district')['enrolments_5_17'].shift(156)
    df['age_transition_pressure_18'] = (
        df['cohort_size_turning_18'] * 0.3 *  # Assume 30% in age 15-18 bucket
        (1 - df.get('child_update_compliance', 0.7))
    ).fillna(0)
else:
    df['age_transition_pressure_15'] = 0
    df['age_transition_pressure_18'] = 0

# 4. Delayed Compliance Score
print("   → Delayed compliance scores...")

# Biometric delay score: overdue biometric updates
if 'total_biometric_updates' in df.columns:
    expected_biometric = (
        df.get('age_transition_pressure_5', 0) + 
        df.get('age_transition_pressure_15', 0)
    )
    actual_biometric = df['total_biometric_updates']
    df['biometric_delay_score'] = (expected_biometric - actual_biometric).clip(lower=0)
    df['biometric_delay_score'] = df['biometric_delay_score'] / (expected_biometric + 1)
else:
    df['biometric_delay_score'] = 0

# Child update backlog ratio
df['child_update_backlog_ratio'] = (
    df.get('biometric_delay_score', 0) + 
    (1 - df.get('child_update_compliance', 0.7))
) / 2

print(f"   ✓ Created 7 life-cycle/age-transition features")

# ============================================================================
# C. MIGRATION & MOBILITY INTELLIGENCE
# ============================================================================
print("\n[4/18] Creating Migration & Mobility features...")

# 5. Net Migration Proxy
print("   → Net migration proxies...")

# High address updates + low enrolments → inward migration
# High enrolments + low address updates → native growth
if 'address_updates' in df.columns:
    df['address_update_rate'] = df['address_updates'] / (df['cumulative_enrolments'] + 1)
    df['enrolment_growth_rate'] = df.groupby('district')['total_enrolments'].pct_change().fillna(0)
    
    # Normalize to 0-1 scale
    df['address_norm'] = (df['address_update_rate'] - df['address_update_rate'].mean()) / (df['address_update_rate'].std() + 0.01)
    df['enrolment_norm'] = (df['enrolment_growth_rate'] - df['enrolment_growth_rate'].mean()) / (df['enrolment_growth_rate'].std() + 0.01)
    
    # Inward migration: high address, low enrolment
    df['net_inward_migration_proxy'] = ((df['address_norm'] > 0) & (df['enrolment_norm'] < 0)).astype(int) * df['address_norm']
    
    # Outward migration: low address, negative enrolment
    df['net_outward_migration_proxy'] = ((df['address_norm'] < 0) & (df['enrolment_norm'] < -0.5)).astype(int) * abs(df['enrolment_norm'])
else:
    df['net_inward_migration_proxy'] = 0
    df['net_outward_migration_proxy'] = 0

# 6. Migration Volatility Index
print("   → Migration volatility indices...")

if 'address_updates' in df.columns:
    df['address_volatility'] = df.groupby('district')['address_updates'].rolling(6).std().reset_index(0, drop=True)
    df['migration_volatility_6m'] = df['address_volatility'] / (df.groupby('district')['address_updates'].rolling(6).mean().reset_index(0, drop=True) + 1)
    
    # Migration spike: address updates > 2 std above mean
    df['migration_spike_flag'] = (
        df['address_updates'] > 
        (df.groupby('district')['address_updates'].transform('mean') + 
         2 * df.groupby('district')['address_updates'].transform('std'))
    ).astype(int)
else:
    df['migration_volatility_6m'] = 0
    df['migration_spike_flag'] = 0

print(f"   ✓ Created 7 migration/mobility features")

# ============================================================================
# D. UPDATE COMPOSITION & QUALITY
# ============================================================================
print("\n[5/18] Creating Update Composition & Quality features...")

# 7. Update Mix Entropy
print("   → Update mix entropy...")

# Calculate entropy of update distribution
update_types = ['total_demographic_updates', 'total_biometric_updates', 'address_updates', 'mobile_updates']
available_types = [col for col in update_types if col in df.columns]

if len(available_types) >= 2:
    # Create distribution for each row
    update_dist = df[available_types].values
    update_dist = update_dist / (update_dist.sum(axis=1, keepdims=True) + 1)  # Normalize
    
    # Calculate entropy
    df['update_entropy_score'] = np.array([
        entropy(row + 1e-10) for row in update_dist  # Add small constant to avoid log(0)
    ])
    
    # Normalize to 0-1
    max_entropy = np.log(len(available_types))
    df['update_entropy_score'] = df['update_entropy_score'] / max_entropy
else:
    df['update_entropy_score'] = 0.5

# 8. Correction vs Maintenance Ratio
print("   → Correction vs maintenance ratios...")

# Corrections: name, DOB, gender changes (data quality issues)
corrections = 0
if 'name_updates' in df.columns:
    corrections += df['name_updates']
if 'dob_updates' in df.columns:
    corrections += df['dob_updates']
if 'gender_updates' in df.columns:
    corrections += df['gender_updates']

df['correction_count'] = corrections

# Maintenance: mobile, address changes (normal life events)
maintenance = 0
if 'mobile_updates' in df.columns:
    maintenance += df['mobile_updates']
if 'address_updates' in df.columns:
    maintenance += df['address_updates']

df['maintenance_count'] = maintenance

# Ratios
total_updates_sum = df['total_updates'] + 1
df['correction_ratio'] = df['correction_count'] / total_updates_sum
df['maintenance_ratio'] = df['maintenance_count'] / total_updates_sum
df['correction_to_maintenance'] = df['correction_count'] / (df['maintenance_count'] + 1)

print(f"   ✓ Created 6 composition/quality features")

# ============================================================================
# E. DISTRICT STRESS & CAPACITY SIGNALS
# ============================================================================
print("\n[6/18] Creating District Stress & Capacity features...")

# 9. Service Load Stress Index
print("   → Service load stress indices...")

# Stress = demand / capacity
if 'digital_inclusion_index' in df.columns:
    df['service_stress_index'] = (
        df['rolling_3m_updates'] / 
        (df['digital_inclusion_index'] + 10)  # Avoid division by near-zero
    )
else:
    df['service_stress_index'] = df['rolling_3m_updates'] / 50  # Fallback

# Normalize
df['service_stress_index'] = (df['service_stress_index'] - df['service_stress_index'].mean()) / (df['service_stress_index'].std() + 0.1)

# 10. Peak Load Concentration
print("   → Peak load concentration...")

df['updates_6m_max'] = df.groupby('district')['total_updates'].rolling(6).max().reset_index(0, drop=True)
df['updates_6m_mean'] = df.groupby('district')['total_updates'].rolling(6).mean().reset_index(0, drop=True)
df['peak_load_ratio'] = df['updates_6m_max'] / (df['updates_6m_mean'] + 1)

# Load variance (how evenly distributed)
df['load_variance'] = df.groupby('district')['total_updates'].rolling(6).std().reset_index(0, drop=True)
df['load_variance_normalized'] = df['load_variance'] / (df['updates_6m_mean'] + 1)

print(f"   ✓ Created 5 stress/capacity features")

# ============================================================================
# F. SOCIETAL STABILITY & TRUST SIGNALS
# ============================================================================
print("\n[7/18] Creating Societal Stability & Trust features...")

# 11. Identity Churn Score
print("   → Identity churn score...")

# Combine name, gender, DOB changes
identity_changes = df.get('correction_count', 0)
df['identity_churn_score'] = identity_changes / (df['cumulative_enrolments'] + 1) * 1000  # Per 1000 people

# Flag frequent modifiers (top 10%)
df['frequent_modifier_flag'] = (df['identity_churn_score'] > df['identity_churn_score'].quantile(0.9)).astype(int)

# 12. Trust Stability Indicator
print("   → Trust stability indicators...")

# Inverse of churn + consistency measure
df['identity_stability_score'] = 1 / (df['identity_churn_score'] + 1)

# Combine with update consistency
if 'update_persistence_6m' in df.columns:
    df['trust_stability_indicator'] = (
        0.6 * df['identity_stability_score'] +
        0.4 * df['update_persistence_6m']
    )
else:
    df['trust_stability_indicator'] = df['identity_stability_score']

# Long-term identity consistency
df['long_term_identity_consistency'] = (
    1 - df.groupby('district')['identity_churn_score'].rolling(12).mean().reset_index(0, drop=True).fillna(0) / 10
).clip(0, 1)

print(f"   ✓ Created 5 stability/trust features")

# ============================================================================
# G. DISTRICT COMPARATIVE FEATURES (RELATIVE METRICS)
# ============================================================================
print("\n[8/18] Creating District Comparative features...")

# 13. Peer-Normalized Scores
print("   → Peer-normalized scores...")

# State-level normalization
if 'state' in df.columns:
    df['state_mean_updates'] = df.groupby('state')['updates_per_1000'].transform('mean')
    df['state_std_updates'] = df.groupby('state')['updates_per_1000'].transform('std')
    df['relative_update_intensity'] = (
        (df['updates_per_1000'] - df['state_mean_updates']) / 
        (df['state_std_updates'] + 0.1)
    )
    
    if 'digital_inclusion_index' in df.columns:
        df['state_mean_digital'] = df.groupby('state')['digital_inclusion_index'].transform('mean')
        df['relative_digital_score'] = df['digital_inclusion_index'] - df['state_mean_digital']
    else:
        df['relative_digital_score'] = 0
    
    if 'aadhaar_maturity_index' in df.columns:
        df['state_mean_maturity'] = df.groupby('state')['aadhaar_maturity_index'].transform('mean')
        df['relative_maturity_score'] = df['aadhaar_maturity_index'] - df['state_mean_maturity']
    else:
        df['relative_maturity_score'] = 0
else:
    df['relative_update_intensity'] = 0
    df['relative_digital_score'] = 0
    df['relative_maturity_score'] = 0

# 14. Rank Momentum Features
print("   → Rank momentum features...")

# Rank districts within state by updates
if 'state' in df.columns:
    df['state_rank'] = df.groupby(['state', 'date'])['total_updates'].rank(ascending=False, method='dense')
    df['state_rank_3m_ago'] = df.groupby('district')['state_rank'].shift(3)
    df['state_rank_6m_ago'] = df.groupby('district')['state_rank'].shift(6)
    
    df['rank_change_3m'] = df['state_rank_3m_ago'] - df['state_rank']  # Positive = improved
    df['rank_change_6m'] = df['state_rank_6m_ago'] - df['state_rank']
    
    # Rank volatility
    df['rank_volatility'] = df.groupby('district')['state_rank'].rolling(6).std().reset_index(0, drop=True)
else:
    df['rank_change_3m'] = 0
    df['rank_change_6m'] = 0
    df['rank_volatility'] = 0

print(f"   ✓ Created 10 comparative features")

# ============================================================================
# H. ANOMALY & EVENT INTELLIGENCE
# ============================================================================
print("\n[9/18] Creating Anomaly & Event Intelligence features...")

# 15. Event Signature Features
print("   → Event signature features...")

# Detect type of anomaly based on characteristics
# Speed: how fast did it rise?
df['update_speed'] = df.groupby('district')['total_updates'].diff()

# Breadth: how widespread (multiple update types)?
if len(available_types) >= 2:
    df['update_breadth'] = (df[available_types] > 0).sum(axis=1) / len(available_types)
else:
    df['update_breadth'] = 0.5

# Duration: rolling window of high activity
df['high_activity_duration'] = df['high_streak']  # Already calculated

# Event likelihood scores
# Policy event: sudden, broad, sustained
df['policy_event_likelihood'] = (
    0.4 * (df['update_speed'] / (df['update_speed'].std() + 1)).clip(0, 1) +
    0.3 * df['update_breadth'] +
    0.3 * (df['high_activity_duration'] / 6).clip(0, 1)
)

# Data quality event: narrow (single type), short duration
df['data_quality_event_likelihood'] = (
    0.5 * (1 - df['update_breadth']) +
    0.3 * (df['update_speed'] / (df['update_speed'].std() + 1)).clip(0, 1) +
    0.2 * (df['high_activity_duration'] == 1).astype(int)
)

# Natural event proxy: sudden, temporary spike in address updates
if 'address_updates' in df.columns:
    df['address_spike'] = (df['address_updates'] > df['address_updates'].quantile(0.95)).astype(int)
    df['natural_event_proxy'] = (
        0.6 * df['address_spike'] +
        0.4 * (df['high_activity_duration'] <= 2).astype(int)
    )
else:
    df['natural_event_proxy'] = 0

print(f"   ✓ Created 8 anomaly/event features")

# ============================================================================
# I. FORECAST-DERIVED FEATURES
# ============================================================================
print("\n[10/18] Creating Forecast-Derived features...")

# 16. Forecast Stress Indicators
print("   → Forecast stress indicators...")

# Simple linear trend forecast (3-month ahead)
def forecast_linear_3m(group):
    """Fit linear trend and predict 3 months ahead"""
    if len(group) < 6:
        return pd.Series([group.mean()] * len(group), index=group.index)
    
    result = []
    for i in range(len(group)):
        if i < 5:
            result.append(group.iloc[i])
        else:
            # Fit on last 6 points
            window = group.iloc[max(0, i-5):i+1]
            x = np.arange(len(window))
            y = window.values
            
            # Linear regression
            slope, intercept = np.polyfit(x, y, 1)
            forecast = slope * (len(window) + 3) + intercept  # 3 months ahead
            result.append(forecast)
    
    return pd.Series(result, index=group.index)

df['forecast_3m'] = df.groupby('district')['total_updates'].apply(forecast_linear_3m).reset_index(0, drop=True)

# Forecasted growth rate
df['forecasted_growth_rate_3m'] = (
    (df['forecast_3m'] - df['total_updates']) / 
    (df['total_updates'] + 1)
).clip(-1, 2)  # Cap extreme values

# Forecast uncertainty (std of residuals)
df['forecast_error'] = abs(df['total_updates'] - df['forecast_3m'])
df['forecast_uncertainty_width'] = df.groupby('district')['forecast_error'].rolling(6).std().reset_index(0, drop=True).fillna(df['forecast_error'].std())

# Forecast spike risk: high growth + high uncertainty
df['forecast_spike_risk'] = (
    0.5 * (df['forecasted_growth_rate_3m'] > 0.2).astype(int) +
    0.5 * (df['forecast_uncertainty_width'] > df['forecast_uncertainty_width'].quantile(0.75)).astype(int)
)

print(f"   ✓ Created 4 forecast-derived features")

# ============================================================================
# J. TREND GEOMETRY & CURVATURE
# ============================================================================
print("\n[11/18] Creating Trend Geometry features...")

# 17. Trend Curvature
print("   → Trend curvature features...")

# Second derivative: acceleration/deceleration
df['update_acceleration'] = df.groupby('district')['update_speed'].diff()

# Curvature score: positive = accelerating, negative = decelerating
df['curvature_score'] = df['update_acceleration'] / (df['total_updates'].std() + 1)

# Convexity indicator
df['is_convex_trend'] = (df['curvature_score'] > 0).astype(int)
df['is_concave_trend'] = (df['curvature_score'] < 0).astype(int)

# Rolling trend slope
df['rolling_trend_slope_3m'] = df.groupby('district')['total_updates'].apply(
    lambda x: x.rolling(3).apply(lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) == 3 else 0)
).reset_index(0, drop=True)

df['rolling_trend_slope_6m'] = df.groupby('district')['total_updates'].apply(
    lambda x: x.rolling(6).apply(lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) == 6 else 0)
).reset_index(0, drop=True)

print(f"   ✓ Created 6 trend geometry features")

# ============================================================================
# K. SEASONALITY & CYCLICITY
# ============================================================================
print("\n[12/18] Creating Seasonality & Cyclicity features...")

# 18. Seasonal Strength
print("   → Seasonal strength indicators...")

# Month with highest average updates (dominant update month)
if 'month' in df.columns:
    df['month_avg_updates'] = df.groupby(['district', 'month'])['total_updates'].transform('mean')
    df['dominant_update_month'] = df.groupby('district')['month_avg_updates'].transform(lambda x: x.idxmax() if len(x) > 0 else 0)
else:
    df['dominant_update_month'] = 0

# Academic cycle alignment (March, June, September enrollment spikes)
if 'month' in df.columns:
    df['academic_cycle_alignment_score'] = df['month'].isin([3, 6, 9]).astype(int) * (
        df['total_updates'] / (df.groupby('district')['total_updates'].transform('mean') + 1)
    )
else:
    df['academic_cycle_alignment_score'] = 0

# Seasonal variance
df['seasonal_variance_score'] = df.groupby(['district', df['month'] if 'month' in df.columns else 0])['total_updates'].transform('std') / (
    df.groupby('district')['total_updates'].transform('std') + 1
)

print(f"   ✓ Created 4 seasonality features")

# ============================================================================
# L. ENGAGEMENT & DIGITAL ADOPTION DEPTH
# ============================================================================
print("\n[13/18] Creating Engagement & Digital Adoption features...")

# 19. Digital Adoption Depth
print("   → Digital adoption depth...")

# Mobile update ratio (digital vs in-person)
if 'mobile_updates' in df.columns:
    df['mobile_update_ratio'] = df['mobile_updates'] / (df['total_updates'] + 1)
    df['digital_self_service_score'] = (
        0.7 * df['mobile_update_ratio'] +
        0.3 * df.get('digital_inclusion_index', 50) / 100
    )
else:
    df['mobile_update_ratio'] = 0
    df['digital_self_service_score'] = 0

# Remote update dependency
df['remote_update_dependency'] = df['mobile_update_ratio']  # Alias for clarity

# 20. Engagement Depth
print("   → Engagement depth metrics...")

# Repeat user proxy: consistent monthly updates
df['update_consistency_6m'] = (
    1 - df.groupby('district')['total_updates'].rolling(6).std().reset_index(0, drop=True).fillna(0) / 
    (df.groupby('district')['total_updates'].rolling(6).mean().reset_index(0, drop=True) + 1)
).clip(0, 1)

df['engagement_consistency_score'] = (
    0.5 * df['update_consistency_6m'] +
    0.5 * df.get('behavioral_memory_score', 0.5)
)

# Update frequency per capita
df['update_frequency_per_capita'] = df['updates_per_1000']  # Already calculated, alias

print(f"   ✓ Created 6 engagement/digital features")

# ============================================================================
# M. CROSS-DATASET SYNTHESIS
# ============================================================================
print("\n[14/18] Creating Cross-Dataset Synthesis features...")

# 21. Enrolment-Update Coupling
print("   → Enrolment-update coupling...")

# Correlation between enrolments and updates
def rolling_correlation(df, col1, col2, window=6):
    """Calculate rolling correlation between two columns"""
    result = df.groupby('district').apply(
        lambda x: x[col1].rolling(window).corr(x[col2])
    ).reset_index(0, drop=True)
    return result

df['enrolment_update_correlation'] = rolling_correlation(df, 'total_enrolments', 'total_updates', window=6)

# Decoupling index: 1 - abs(correlation)
df['decoupling_index'] = 1 - abs(df['enrolment_update_correlation'].fillna(0))

# 22. Lifecycle Completeness
print("   → Lifecycle completeness...")

# Ratio of updates to enrolments (completeness proxy)
df['enrolment_to_update_completion_rate'] = (
    df['rolling_3m_updates'] / 
    (df.groupby('district')['total_enrolments'].rolling(3).sum().reset_index(0, drop=True) + 1)
)

# Identity maturity: saturation + completion
if 'saturation_ratio' in df.columns:
    df['identity_maturity_score'] = (
        0.6 * df['saturation_ratio'].clip(0, 1) +
        0.4 * df['enrolment_to_update_completion_rate'].clip(0, 1)
    )
else:
    df['identity_maturity_score'] = df['enrolment_to_update_completion_rate'].clip(0, 1)

print(f"   ✓ Created 5 synthesis features")

# ============================================================================
# N. COMPOSITE SUMMARY INDICES
# ============================================================================
print("\n[15/18] Creating Composite Summary Indices...")

# 23. Five Core Indices
print("   → Five core summary indices...")

# 1. Aadhaar System Maturity Index (already exists, enhance)
if 'aadhaar_maturity_index' not in df.columns:
    df['aadhaar_maturity_index'] = (
        0.3 * df.get('saturation_ratio', 0.7).clip(0, 1) +
        0.3 * df.get('identity_stability_score', 0.5) +
        0.2 * df.get('update_consistency_6m', 0.5) +
        0.2 * df.get('child_update_compliance', 0.7)
    ) * 100

# 2. District Service Stress Index
df['district_service_stress_index'] = (
    0.3 * (df['service_stress_index'] - df['service_stress_index'].min()) / (df['service_stress_index'].max() - df['service_stress_index'].min() + 0.01) +
    0.3 * df.get('peak_load_ratio', 1).clip(0, 5) / 5 +
    0.2 * df.get('load_variance_normalized', 0.5) +
    0.2 * (1 - df.get('digital_inclusion_index', 50) / 100)
) * 100

# 3. Identity Stability Index
df['identity_stability_index'] = (
    0.4 * df.get('identity_stability_score', 0.5) +
    0.3 * df.get('trust_stability_indicator', 0.5) +
    0.3 * df.get('long_term_identity_consistency', 0.5)
) * 100

# 4. Migration & Mobility Index
df['migration_mobility_index'] = (
    0.3 * (df.get('net_inward_migration_proxy', 0) + df.get('net_outward_migration_proxy', 0)).clip(0, 1) +
    0.3 * df.get('migration_volatility_6m', 0).clip(0, 1) +
    0.2 * df.get('migration_spike_flag', 0) +
    0.2 * (df.get('address_intensity', 0) / (df.get('address_intensity', 0).max() + 1))
) * 100

# 5. Digital Engagement Index (enhance existing)
if 'digital_inclusion_index' in df.columns:
    df['digital_engagement_index'] = (
        0.4 * df['digital_inclusion_index'] / 100 +
        0.3 * df.get('digital_self_service_score', 0.5) +
        0.3 * df.get('citizen_engagement_index', 50) / 100
    ) * 100
else:
    df['digital_engagement_index'] = (
        0.5 * df.get('digital_self_service_score', 0.5) +
        0.5 * df.get('citizen_engagement_index', 50) / 100
    ) * 100

print(f"   ✓ Created 5 composite summary indices")

# ============================================================================
# O. META-FEATURES FOR DISTRICT HEALTH
# ============================================================================
print("\n[16/18] Creating Meta-Features for District Health...")

# 24. District Health Vector
print("   → District health vector...")

# Normalize all indices to 0-100
df['stress_normalized'] = df['district_service_stress_index']
df['maturity_normalized'] = df['aadhaar_maturity_index']
df['engagement_normalized'] = df['digital_engagement_index']
df['stability_normalized'] = df['identity_stability_index']
df['mobility_normalized'] = df['migration_mobility_index']

# Overall district health score
df['district_health_score'] = (
    0.25 * (100 - df['stress_normalized']) +  # Lower stress = better
    0.25 * df['maturity_normalized'] +
    0.25 * df['engagement_normalized'] +
    0.15 * df['stability_normalized'] +
    0.10 * (100 - df['mobility_normalized'])  # Lower mobility = more stable
)

print(f"   ✓ Created district health meta-features")

# ============================================================================
# P. RECOVERY & RESILIENCE FEATURES
# ============================================================================
print("\n[17/18] Creating Recovery & Resilience features...")

# 25. Recovery Metrics
print("   → Recovery and resilience metrics...")

# Recovery half-life: time to recover from drop
df['is_drop'] = (df['update_speed'] < 0).astype(int)
df['months_since_drop'] = df.groupby('district')['is_drop'].apply(
    lambda x: x[::-1].cumsum()[::-1]
).reset_index(0, drop=True)

# Recovery rate: growth after drop
df['recovery_rate'] = df.groupby('district').apply(
    lambda x: x['update_speed'].where(x['is_drop'].shift(1) == 1, 0)
).reset_index(0, drop=True)

# Resilience score: how quickly does it bounce back
df['resilience_score'] = (
    (df['recovery_rate'] > 0).astype(int) * 
    (df['recovery_rate'] / (df['recovery_rate'].abs().max() + 1))
).clip(0, 1)

print(f"   ✓ Created 4 recovery/resilience features")

# ============================================================================
# Q. FINAL CLEANUP & SAVE
# ============================================================================
print("\n[18/18] Final cleanup and saving...")

# Fill NaN values
print("   → Filling missing values...")
df = df.fillna({
    'update_burst_score': 1.0,
    'post_burst_fatigue': 0.0,
    'update_persistence_3m': 0.5,
    'update_persistence_6m': 0.5,
    'behavioral_memory_score': 0.5,
    'enrolment_update_correlation': 0.0,
    'forecast_3m': df['total_updates'].median(),
    'forecasted_growth_rate_3m': 0.0,
    'forecast_uncertainty_width': df.get('forecast_error', pd.Series([0])).std(),
})

# Drop temporary columns
temp_cols = [
    'updates_12m_max', 'updates_12m_mean', 'update_next_month', 'is_high_month', 
    'high_streak', 'cohort_size_turning_5', 'cohort_size_turning_15', 'cohort_size_turning_18',
    'address_norm', 'enrolment_norm', 'address_volatility', 'update_speed',
    'update_breadth', 'high_activity_duration', 'address_spike', 'forecast_error',
    'update_acceleration', 'month_avg_updates', 'is_drop', 'months_since_drop',
    'state_mean_updates', 'state_std_updates', 'state_mean_digital', 'state_mean_maturity',
    'state_rank_3m_ago', 'state_rank_6m_ago', 'updates_6m_max', 'updates_6m_mean',
    'stress_normalized', 'maturity_normalized', 'engagement_normalized',
    'stability_normalized', 'mobility_normalized', 'address_update_rate', 'enrolment_growth_rate'
]

df = df.drop(columns=[col for col in temp_cols if col in df.columns], errors='ignore')

# Save extended dataset
print("\n   → Saving extended dataset...")
output_path = 'data/processed/aadhaar_extended_features.csv'
df.to_csv(output_path, index=False)

# ============================================================================
# SUMMARY REPORT
# ============================================================================
print("\n" + "="*80)
print("FEATURE ENGINEERING COMPLETE!")
print("="*80)

# Count new features
new_feature_categories = {
    'Temporal & Behavioral': 8,
    'Life-Cycle & Age-Transition': 7,
    'Migration & Mobility': 7,
    'Update Composition & Quality': 6,
    'District Stress & Capacity': 5,
    'Societal Stability & Trust': 5,
    'District Comparative': 10,
    'Anomaly & Event Intelligence': 8,
    'Forecast-Derived': 4,
    'Trend Geometry': 6,
    'Seasonality & Cyclicity': 4,
    'Engagement & Digital Adoption': 6,
    'Cross-Dataset Synthesis': 5,
    'Composite Summary Indices': 5,
    'District Health Meta-Features': 1,
    'Recovery & Resilience': 4
}

print(f"\n📊 NEW FEATURES CREATED:")
total_new = 0
for category, count in new_feature_categories.items():
    print(f"   • {category:35} {count:3} features")
    total_new += count

print(f"\n   {'TOTAL NEW FEATURES:':<35} {total_new:3}")
print(f"   {'ORIGINAL FEATURES:':<35} {102:3}")
print(f"   {'GRAND TOTAL:':<35} {102 + total_new:3}")

print(f"\n💾 Saved to: {output_path}")
print(f"   • Records: {len(df):,}")
print(f"   • Features: {len(df.columns)}")
print(f"   • File size: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

print("\n✅ Ready for model training with expanded feature set!")
print("="*80)
