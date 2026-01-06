# Script to run Feature Engineering Notebook as Python code
# Import libraries
import sys
sys.path.append('../src')

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from feature_engineering import AadhaarFeatureEngineer
import warnings
import time
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

print("Libraries imported successfully")

# Step 1: Load Merged Dataset
print("\n" + "="*80)
print("STEP 1: Loading merged dataset...")
print("="*80)
df = pd.read_csv('../data/processed/merged_aadhaar_data.csv')

# Convert date column
df['date'] = pd.to_datetime(df['date'])

print(f"Dataset loaded: {df.shape}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"\nDate range: {df['date'].min()} to {df['date'].max()}")
print(f"States: {df['state'].nunique()}")
print(f"Districts: {df['district'].nunique()}")

# Step 2: Initialize Feature Engineer
print("\n" + "="*80)
print("STEP 2: Initialize Feature Engineer")
print("="*80)
engineer = AadhaarFeatureEngineer()

print("Feature Engineer initialized")
print("\nThis will create 20+ new features across 5 layers")

# Step 3: Apply Feature Engineering
print("\n" + "="*80)
print("STEP 3: Apply Feature Engineering")
print("="*80)
print("Applying feature engineering...")
start_time = time.time()

df_features = engineer.engineer_all_features(df)

elapsed = time.time() - start_time
print(f"\nFeature engineering completed in {elapsed:.2f} seconds")
print(f"Total columns: {len(df_features.columns)}")
print(f"New features created: {len(df_features.columns) - len(df.columns)}")

# Step 4: Feature Summary Statistics
print("\n" + "="*80)
print("STEP 4: Feature Summary Statistics")
print("="*80)
feature_summary = engineer.get_feature_summary(df_features)

print("FEATURE SUMMARY STATISTICS")
print("="*80)
print(feature_summary.round(4))

# Create outputs directory if it doesn't exist
import os
os.makedirs('../outputs/tables', exist_ok=True)
os.makedirs('../outputs/figures', exist_ok=True)

# Save summary
feature_summary.to_csv('../outputs/tables/feature_summary.csv', index=False)
print("\nSummary saved to outputs/tables/feature_summary.csv")

# Step 5: Visualize Key Societal Indicators (Layer 3)
print("\n" + "="*80)
print("STEP 5: Visualize Key Societal Indicators (Layer 3)")
print("="*80)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

indicators = [
    'mobility_indicator',
    'digital_instability_index',
    'identity_stability_score',
    'update_burden_index',
    'manual_labor_proxy',
    'lifecycle_transition_spike'
]

titles = [
    'Mobility Indicator\n(Migration Proxy)',
    'Digital Instability Index\n(Mobile Churn)',
    'Identity Stability Score\n(KEY METRIC)',
    'Update Burden Index\n(Service Load)',
    'Manual Labor Proxy\n(Fingerprint Degradation)',
    'Lifecycle Transition Spike\n(Age Stress)'
]

for idx, (indicator, title) in enumerate(zip(indicators, titles)):
    ax = axes[idx // 3, idx % 3]
    
    # Histogram
    df_features[indicator].hist(bins=50, ax=ax, edgecolor='black', alpha=0.7)
    ax.set_title(title, fontweight='bold', fontsize=11)
    ax.set_xlabel(indicator, fontsize=9)
    ax.set_ylabel('Frequency', fontsize=9)
    ax.grid(alpha=0.3)
    
    # Add mean line
    mean_val = df_features[indicator].mean()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
    ax.legend()

plt.suptitle('Layer 3: Societal Indicators - Distribution Analysis', 
             fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('../outputs/figures/layer3_societal_indicators.png', dpi=300, bbox_inches='tight')
plt.close()

print("Visualization saved to outputs/figures/layer3_societal_indicators.png")

# Step 6: Identity Stability Score Analysis (KEY FEATURE)
print("\n" + "="*80)
print("STEP 6: Identity Stability Score Analysis (KEY FEATURE)")
print("="*80)

print("IDENTITY STABILITY SCORE ANALYSIS")
print("="*80)

# Overall statistics
print("\nOverall Statistics:")
print(df_features['identity_stability_score'].describe())

# Classification into stability levels
df_features['stability_category'] = pd.cut(
    df_features['identity_stability_score'],
    bins=[0, 0.4, 0.7, 1.0],
    labels=['Low Stability', 'Medium Stability', 'High Stability']
)

print("\nStability Distribution:")
print(df_features['stability_category'].value_counts())
print("\nPercentages:")
print(df_features['stability_category'].value_counts(normalize=True) * 100)

# Top 10 most stable states
print("\nTop 10 Most Stable States:")
top_stable_states = df_features.groupby('state')['identity_stability_score'].mean().nlargest(10)
print(top_stable_states)

# Bottom 10 least stable states
print("\nBottom 10 Least Stable States (Need Intervention):")
bottom_stable_states = df_features.groupby('state')['identity_stability_score'].mean().nsmallest(10)
print(bottom_stable_states)

# Step 7: Mobility Indicator Analysis (Migration Proxy)
print("\n" + "="*80)
print("STEP 7: Mobility Indicator Analysis (Migration Proxy)")
print("="*80)

print("MOBILITY INDICATOR ANALYSIS")
print("="*80)

# High mobility states (potential migration hotspots)
print("\nTop 15 High Mobility States (Migration Hotspots):")
high_mobility = df_features.groupby('state')['mobility_indicator'].mean().nlargest(15)
print(high_mobility)

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Top mobility states
high_mobility.plot(kind='barh', ax=ax1, color='coral')
ax1.set_title('Top 15 High Mobility States\n(Migration Hotspots)', fontweight='bold', fontsize=12)
ax1.set_xlabel('Average Mobility Indicator', fontsize=10)
ax1.grid(axis='x', alpha=0.3)

# Mobility over time
mobility_time = df_features.groupby('date')['mobility_indicator'].mean()
ax2.plot(mobility_time.index, mobility_time.values, linewidth=2, color='darkblue')
ax2.set_title('National Mobility Trend Over Time', fontweight='bold', fontsize=12)
ax2.set_xlabel('Date', fontsize=10)
ax2.set_ylabel('Average Mobility Indicator', fontsize=10)
ax2.grid(alpha=0.3)
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
plt.savefig('../outputs/figures/mobility_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nVisualization saved to outputs/figures/mobility_analysis.png")

# Step 8: Feature Correlation Analysis
print("\n" + "="*80)
print("STEP 8: Feature Correlation Analysis")
print("="*80)

key_features = [
    'enrolment_growth_rate',
    'adult_enrolment_share',
    'mobility_indicator',
    'digital_instability_index',
    'identity_stability_score',
    'update_burden_index',
    'manual_labor_proxy',
    'seasonal_variance_score',
    'anomaly_severity_score',
    'enrolment_volatility_index'
]

corr_matrix = df_features[key_features].corr()

# Visualize
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Matrix\nKey Societal Indicators', 
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('../outputs/figures/feature_correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

print("Correlation matrix saved to outputs/figures/feature_correlation_matrix.png")

# Step 9: Save Feature-Engineered Dataset
print("\n" + "="*80)
print("STEP 9: Save Feature-Engineered Dataset")
print("="*80)

output_path = '../data/processed/aadhaar_with_features.csv'
print(f"Saving feature-engineered dataset...")
df_features.to_csv(output_path, index=False)

file_size = df_features.memory_usage(deep=True).sum() / 1024**2
print(f"Saved to: {output_path}")
print(f"File size: ~{file_size:.2f} MB")
print(f"Total features: {len(df_features.columns)}")

# Also save column documentation
print("\nCreating column documentation...")
column_info = pd.DataFrame({
    'column_name': df_features.columns,
    'data_type': df_features.dtypes.values,
    'non_null_count': df_features.count().values,
    'null_count': df_features.isnull().sum().values
})
column_info.to_csv('../outputs/tables/column_documentation.csv', index=False)
print("Column documentation saved to outputs/tables/column_documentation.csv")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("✓ Successfully applied 20+ engineered features to 2.9M records")
print("✓ Created Layer 2-8 features (normalized, societal, temporal, equity, resilience)")
print("✓ Generated visualizations and statistical summaries")
print("✓ Saved feature-engineered dataset")
print("\nAll outputs saved to:")
print("  - data/processed/aadhaar_with_features.csv")
print("  - outputs/tables/feature_summary.csv")
print("  - outputs/tables/column_documentation.csv")
print("  - outputs/figures/")
print("\nReady for notebooks 03-05 (Univariate, Bivariate, Trivariate Analysis)")
