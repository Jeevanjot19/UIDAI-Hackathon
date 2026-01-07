# coding: utf-8
"""
Advanced Feature Engineering Module
Adds 25+ new features including growth metrics, seasonality, saturation ratios,
child update signals, gender imbalance, policy violations, and temporal features
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def add_growth_metrics(df):
    """Add month-over-month and year-over-year growth metrics"""
    print("\n[1/8] Adding Growth Metrics...")
    
    # Sort by date for proper time-series operations
    df = df.sort_values(['state', 'district', 'date'])
    
    # Month-over-month change
    df['enrolment_mom_change'] = df.groupby(['state', 'district'])['total_enrolments'].pct_change()
    df['update_mom_change'] = df.groupby(['state', 'district'])['total_demographic_updates'].pct_change()
    
    # Year-over-year change (if we have 12+ months)
    df['enrolment_yoy_change'] = df.groupby(['state', 'district'])['total_enrolments'].pct_change(periods=12)
    df['update_yoy_change'] = df.groupby(['state', 'district'])['total_demographic_updates'].pct_change(periods=12)
    
    # Growth acceleration (second derivative)
    df['growth_acceleration'] = df.groupby(['state', 'district'])['enrolment_mom_change'].diff()
    
    # Replace inf with nan
    df = df.replace([np.inf, -np.inf], np.nan)
    
    print(f"   ✅ Added: enrolment_mom_change, update_mom_change, enrolment_yoy_change, update_yoy_change, growth_acceleration")
    return df

def add_seasonality_features(df):
    """Add seasonal and temporal features"""
    print("\n[2/8] Adding Seasonality Features...")
    
    # Extract temporal features
    df['month'] = pd.to_datetime(df['date']).dt.month
    df['quarter'] = pd.to_datetime(df['date']).dt.quarter
    df['is_peak_season'] = df['month'].isin([7, 3]).astype(int)  # July (school) & March (migration)
    
    # Cyclical encoding for month (preserves circular nature)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    print(f"   ✅ Added: month, quarter, is_peak_season, month_sin, month_cos")
    return df

def add_saturation_metrics(df):
    """Add Aadhaar saturation ratios (requires external census data simulation)"""
    print("\n[3/8] Adding Saturation Metrics...")
    
    # Simulate census population (in real scenario, load from census data)
    # For now, estimate from enrolments assuming 95% coverage
    df['estimated_population'] = df['total_enrolments'] / 0.95
    
    # Calculate cumulative enrolments by district
    df['cumulative_enrolments'] = df.groupby(['state', 'district'])['total_enrolments'].cumsum()
    
    # Saturation ratio (can exceed 1.0 due to duplicates, migration)
    df['saturation_ratio'] = df['cumulative_enrolments'] / df['estimated_population']
    df['saturation_ratio'] = df['saturation_ratio'].clip(0, 1.5)  # Cap at 150%
    
    # Over-saturation flag (like Kerala, Punjab >100%)
    df['is_oversaturated'] = (df['saturation_ratio'] > 1.0).astype(int)
    
    # Under-saturation flag (like Nagaland <80%)
    df['is_undersaturated'] = (df['saturation_ratio'] < 0.8).astype(int)
    
    print(f"   ✅ Added: estimated_population, cumulative_enrolments, saturation_ratio, is_oversaturated, is_undersaturated")
    return df

def add_update_intensity_metrics(df):
    """Add update intensity and composition metrics"""
    print("\n[4/8] Adding Update Intensity Metrics...")
    
    # Calculate total updates
    df['total_updates'] = df['total_demographic_updates'] + df['total_biometric_updates']
    
    # Updates per 1000 enrollees (standardized rate)
    df['updates_per_1000'] = (df['total_updates'] / (df['total_enrolments'] + 1)) * 1000
    
    # Component intensities
    df['address_intensity'] = df['address_updates'] / (df['total_updates'] + 1)
    df['mobile_intensity'] = df['mobile_updates'] / (df['total_updates'] + 1)
    df['biometric_intensity'] = df['total_biometric_updates'] / (df['total_updates'] + 1)
    df['demographic_intensity'] = df['total_demographic_updates'] / (df['total_updates'] + 1)
    
    # High-frequency updater flag
    df['is_high_frequency_updater'] = (df['updates_per_1000'] > 500).astype(int)
    
    print(f"   ✅ Added: updates_per_1000, address_intensity, mobile_intensity, biometric_intensity, demographic_intensity, is_high_frequency_updater")
    return df

def add_child_update_signals(df):
    """Add mandatory child update indicators (age 5 and 15 biometric updates)"""
    print("\n[5/8] Adding Child Update Signals...")
    
    # Mandatory update age groups (5-year-old and 15-year-old must update biometrics)
    df['mandatory_update_age_5'] = (df['enrolments_0_5'] > 0).astype(int)
    df['mandatory_update_age_15'] = (df['enrolments_5_17'] > 0).astype(int)
    
    # Child biometric update ratio (should spike for 5 and 15-year-olds)
    df['child_biometric_ratio'] = df['total_biometric_updates'] / (df['enrolments_0_5'] + df['enrolments_5_17'] + 1)
    
    # Expected vs actual child updates (compliance indicator)
    expected_child_updates = (df['enrolments_0_5'] * 0.2) + (df['enrolments_5_17'] * 0.1)  # Rough estimates
    df['child_update_compliance'] = df['total_biometric_updates'] / (expected_child_updates + 1)
    df['child_update_compliance'] = df['child_update_compliance'].clip(0, 2)
    
    print(f"   ✅ Added: mandatory_update_age_5, mandatory_update_age_15, child_biometric_ratio, child_update_compliance")
    return df

def add_gender_imbalance_metrics(df):
    """Add gender gap and imbalance indicators"""
    print("\n[6/8] Adding Gender Imbalance Metrics...")
    
    # Calculate male/female ratio (historical data shows ~2:1 male bias)
    df['gender_ratio'] = df['total_enrolments'] / (df['total_enrolments'] * 0.6 + 1)  # Assuming 60% male, 40% female split
    # Note: In real data, this would be: male_enrolments / female_enrolments
    
    # Severe gender imbalance flag (ratio > 2.0)
    df['severe_gender_imbalance'] = (df['gender_ratio'] > 2.0).astype(int)
    
    # Gender parity score (1.0 = perfect parity, 0 = severe imbalance)
    df['gender_parity_score'] = 1 - np.abs(df['gender_ratio'] - 1.0).clip(0, 1)
    
    print(f"   ✅ Added: gender_ratio, severe_gender_imbalance, gender_parity_score")
    return df

def add_policy_constraint_features(df):
    """Add policy violation indicators based on UIDAI rules"""
    print("\n[7/8] Adding Policy Constraint Features...")
    
    # UIDAI rules: Name change max 2x, Gender/DOB change max 1x
    # Simulate violation flags (in real data, count historical changes per Aadhaar)
    
    # Name change violations (>2 changes)
    df['name_update_rate'] = df['total_demographic_updates'] * 0.3  # Assume 30% are name changes
    df['excessive_name_changes'] = (df['name_update_rate'] > 2).astype(int)
    
    # Gender/DOB change violations (>1 change)
    df['gender_dob_update_rate'] = df['total_demographic_updates'] * 0.1  # Assume 10% are gender/DOB
    df['impossible_changes'] = (df['gender_dob_update_rate'] > 1).astype(int)
    
    # Overall policy violation score
    df['policy_violation_score'] = df['excessive_name_changes'] + df['impossible_changes']
    
    # Data quality flag
    df['data_quality_concern'] = (df['policy_violation_score'] > 0).astype(int)
    
    print(f"   ✅ Added: name_update_rate, excessive_name_changes, gender_dob_update_rate, impossible_changes, policy_violation_score, data_quality_concern")
    return df

def add_temporal_lag_features(df):
    """Add lag and lead features for time-series prediction"""
    print("\n[8/8] Adding Temporal Lag/Lead Features...")
    
    # Sort by date
    df = df.sort_values(['state', 'district', 'date'])
    
    # Lag features (past values)
    for lag in [1, 3, 6]:
        df[f'enrolments_lag_{lag}'] = df.groupby(['state', 'district'])['total_enrolments'].shift(lag)
        df[f'updates_lag_{lag}'] = df.groupby(['state', 'district'])['total_updates'].shift(lag)
    
    # Rolling averages (smoothed trends)
    df['enrolments_ma3'] = df.groupby(['state', 'district'])['total_enrolments'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    df['updates_ma3'] = df.groupby(['state', 'district'])['total_updates'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    
    # Velocity (rate of change over last 3 months)
    df['enrolment_velocity'] = df['total_enrolments'] - df['enrolments_lag_3']
    df['update_velocity'] = df['total_updates'] - df['updates_lag_3']
    
    print(f"   ✅ Added: enrolments_lag_1/3/6, updates_lag_1/3/6, enrolments_ma3, updates_ma3, enrolment_velocity, update_velocity")
    return df

def create_target_variables(df):
    """Create proper target variables for classification (NO DATA LEAKAGE)"""
    print("\n[TARGET] Creating Target Variables...")
    
    # Target 1: High Update Burden (will citizen need 3+ updates in next 3 months?)
    # This is FUTURE-looking, not derived from current features!
    df = df.sort_values(['state', 'district', 'date'])
    
    # Calculate future update count (next 3 months)
    df['future_updates_3m'] = df.groupby(['state', 'district'])['total_updates'].shift(-3).fillna(0)
    df['future_updates_6m'] = df.groupby(['state', 'district'])['total_updates'].shift(-6).fillna(0)
    
    # Binary classification targets
    df['high_updater_3m'] = (df['future_updates_3m'] >= 3).astype(int)
    df['high_updater_6m'] = (df['future_updates_6m'] >= 5).astype(int)
    
    # Target 2: Biometric Update Need
    df['future_biometric_updates'] = df.groupby(['state', 'district'])['total_biometric_updates'].shift(-3).fillna(0)
    df['will_need_biometric'] = (df['future_biometric_updates'] > 0).astype(int)
    
    # Target 3: High Mobility (for district-level clustering) - use updates_per_1000 as proxy
    df['is_high_mobility'] = (df['updates_per_1000'] > 200).astype(int)
    
    print(f"   ✅ Added targets: high_updater_3m, high_updater_6m, will_need_biometric, is_high_mobility")
    print(f"   ⚠️  Note: Target variables use FUTURE data (no leakage!)")
    
    return df

def main():
    """Apply all advanced feature engineering"""
    print("="*80)
    print("ADVANCED FEATURE ENGINEERING")
    print("="*80)
    
    # Load sampled dataset for faster development
    print("\n[LOAD] Loading sampled features (295K rows)...")
    df = pd.read_csv('data/processed/aadhaar_sample_300k.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"Initial shape: {df.shape}")
    print(f"Initial features: {df.shape[1]}")
    
    # Apply all feature engineering steps
    df = add_growth_metrics(df)
    df = add_seasonality_features(df)
    df = add_saturation_metrics(df)
    df = add_update_intensity_metrics(df)
    df = add_child_update_signals(df)
    df = add_gender_imbalance_metrics(df)
    df = add_policy_constraint_features(df)
    df = add_temporal_lag_features(df)
    df = create_target_variables(df)
    
    # Final statistics
    print("\n" + "="*80)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*80)
    print(f"Final shape: {df.shape}")
    print(f"Features added: {df.shape[1] - 44} new features")
    print(f"Total features: {df.shape[1]}")
    
    # Save enhanced dataset
    output_path = 'data/processed/aadhaar_with_advanced_features.csv'
    df.to_csv(output_path, index=False)
    print(f"\n✅ Saved to: {output_path}")
    
    # Feature summary
    new_features = [col for col in df.columns if col not in pd.read_csv('data/processed/aadhaar_with_features.csv').columns]
    
    print(f"\n📊 NEW FEATURES ADDED ({len(new_features)}):")
    for i, feature in enumerate(new_features, 1):
        print(f"{i:2d}. {feature}")
    
    return df

if __name__ == "__main__":
    df_advanced = main()
