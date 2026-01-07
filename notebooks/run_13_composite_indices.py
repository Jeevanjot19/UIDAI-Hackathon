"""
Composite Indices Creation
Day 7 of Implementation Plan

Goals:
1. Digital Inclusion Index (0-100 scale)
2. Service Quality Score
3. Update Burden Index
4. Aadhaar Maturity Index
5. State/district rankings
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("COMPOSITE INDICES CREATION")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[LOAD] Loading dataset...")
df = pd.read_csv('data/processed/aadhaar_with_advanced_features.csv')
df['date'] = pd.to_datetime(df['date'])

print(f"Dataset: {df.shape}")

# ============================================================================
# 1. DIGITAL INCLUSION INDEX (0-100)
# ============================================================================
print("\n" + "="*80)
print("DIGITAL INCLUSION INDEX")
print("="*80)

print("\nCalculating sub-components...")

# Sub-components (normalized 0-100)
scaler = MinMaxScaler(feature_range=(0, 100))

# a) Mobile Update Intensity (higher = more digital)
df['mobile_digital_score'] = scaler.fit_transform(df[['mobile_intensity']])

# b) Saturation (higher = better coverage)
df['saturation_score'] = scaler.fit_transform(df[['saturation_ratio']])

# c) Digital Instability (INVERTED - lower instability = better)
df['stability_score'] = 100 - scaler.fit_transform(df[['digital_instability_index']])

# d) Online Update Rate (demographic updates are often online)
df['online_update_score'] = scaler.fit_transform(df[['demographic_update_rate']])

# Composite: Weighted average
df['digital_inclusion_index'] = (
    0.30 * df['mobile_digital_score'] +
    0.25 * df['saturation_score'] +
    0.25 * df['stability_score'] +
    0.20 * df['online_update_score']
)

print(f"✅ Digital Inclusion Index created (0-100 scale)")
print(f"   Mean: {df['digital_inclusion_index'].mean():.2f}")
print(f"   Median: {df['digital_inclusion_index'].median():.2f}")
print(f"   Std: {df['digital_inclusion_index'].std():.2f}")

# ============================================================================
# 2. SERVICE QUALITY SCORE (0-100)
# ============================================================================
print("\n" + "="*80)
print("SERVICE QUALITY SCORE")
print("="*80)

print("\nCalculating sub-components...")

# a) Service Accessibility (higher = better)
df['accessibility_score'] = scaler.fit_transform(df[['service_accessibility_score']])

# b) Update Burden (INVERTED - lower burden = better service)
df['burden_score'] = 100 - scaler.fit_transform(df[['update_burden_index']])

# c) Policy Violation (INVERTED - fewer violations = better quality)
df['compliance_score'] = 100 - scaler.fit_transform(df[['policy_violation_score']].clip(upper=df['policy_violation_score'].quantile(0.99)))

# d) Recovery Rate (higher = resilient service)
df['resilience_score'] = scaler.fit_transform(df[['recovery_rate']])

# Composite
df['service_quality_score'] = (
    0.35 * df['accessibility_score'] +
    0.30 * df['burden_score'] +
    0.20 * df['compliance_score'] +
    0.15 * df['resilience_score']
)

print(f"✅ Service Quality Score created (0-100 scale)")
print(f"   Mean: {df['service_quality_score'].mean():.2f}")
print(f"   Median: {df['service_quality_score'].median():.2f}")
print(f"   Std: {df['service_quality_score'].std():.2f}")

# ============================================================================
# 3. AADHAAR MATURITY INDEX (0-100)
# ============================================================================
print("\n" + "="*80)
print("AADHAAR MATURITY INDEX")
print("="*80)

print("\nCalculating sub-components...")

# a) Saturation (higher = mature)
df['maturity_saturation'] = scaler.fit_transform(df[['saturation_ratio']])

# b) Stability (higher = mature)
df['maturity_stability'] = scaler.fit_transform(df[['identity_stability_score']])

# c) Child Compliance (higher = mature)
df['maturity_compliance'] = scaler.fit_transform(df[['child_update_compliance']])

# d) Low Volatility (INVERTED - lower volatility = mature)
df['maturity_steady'] = 100 - scaler.fit_transform(df[['enrolment_volatility_index']])

# Composite
df['aadhaar_maturity_index'] = (
    0.30 * df['maturity_saturation'] +
    0.30 * df['maturity_stability'] +
    0.20 * df['maturity_compliance'] +
    0.20 * df['maturity_steady']
)

print(f"✅ Aadhaar Maturity Index created (0-100 scale)")
print(f"   Mean: {df['aadhaar_maturity_index'].mean():.2f}")
print(f"   Median: {df['aadhaar_maturity_index'].median():.2f}")
print(f"   Std: {df['aadhaar_maturity_index'].std():.2f}")

# ============================================================================
# 4. CITIZEN ENGAGEMENT INDEX (0-100)
# ============================================================================
print("\n" + "="*80)
print("CITIZEN ENGAGEMENT INDEX")
print("="*80)

print("\nCalculating sub-components...")

# a) Update Frequency (higher = engaged)
df['engagement_frequency'] = scaler.fit_transform(df[['updates_per_1000']])

# b) Biometric Updates (higher = proactive)
df['engagement_biometric'] = scaler.fit_transform(df[['biometric_intensity']])

# c) Mobility (higher = engaged/dynamic)
df['engagement_mobility'] = scaler.fit_transform(df[['mobility_indicator']])

# d) Address Updates (higher = engaged)
df['engagement_address'] = scaler.fit_transform(df[['address_intensity']])

# Composite
df['citizen_engagement_index'] = (
    0.35 * df['engagement_frequency'] +
    0.25 * df['engagement_biometric'] +
    0.20 * df['engagement_mobility'] +
    0.20 * df['engagement_address']
)

print(f"✅ Citizen Engagement Index created (0-100 scale)")
print(f"   Mean: {df['citizen_engagement_index'].mean():.2f}")
print(f"   Median: {df['citizen_engagement_index'].median():.2f}")
print(f"   Std: {df['citizen_engagement_index'].std():.2f}")

# ============================================================================
# SAVE ENRICHED DATASET
# ============================================================================
print("\n[SAVE] Saving enriched dataset with indices...")
df.to_csv('data/processed/aadhaar_with_indices.csv', index=False)
print(f"✅ Saved: data/processed/aadhaar_with_indices.csv ({df.shape})")

# ============================================================================
# DISTRICT-LEVEL RANKINGS
# ============================================================================
print("\n" + "="*80)
print("DISTRICT-LEVEL RANKINGS")
print("="*80)

district_indices = df.groupby(['state', 'district']).agg({
    'digital_inclusion_index': 'mean',
    'service_quality_score': 'mean',
    'aadhaar_maturity_index': 'mean',
    'citizen_engagement_index': 'mean',
    'total_enrolments': 'sum',
    'total_all_updates': 'sum'
}).reset_index()

# Overall Index (average of all 4)
district_indices['overall_index'] = (
    district_indices['digital_inclusion_index'] +
    district_indices['service_quality_score'] +
    district_indices['aadhaar_maturity_index'] +
    district_indices['citizen_engagement_index']
) / 4

# Rankings
district_indices['digital_rank'] = district_indices['digital_inclusion_index'].rank(ascending=False)
district_indices['service_rank'] = district_indices['service_quality_score'].rank(ascending=False)
district_indices['maturity_rank'] = district_indices['aadhaar_maturity_index'].rank(ascending=False)
district_indices['engagement_rank'] = district_indices['citizen_engagement_index'].rank(ascending=False)
district_indices['overall_rank'] = district_indices['overall_index'].rank(ascending=False)

# Save rankings
district_indices.to_csv('outputs/tables/district_index_rankings.csv', index=False)
print(f"✅ Saved: outputs/tables/district_index_rankings.csv")

# Top 10 districts
print(f"\n🏆 TOP 10 DISTRICTS (Overall Index):")
top10 = district_indices.nsmallest(10, 'overall_rank')[['state', 'district', 'overall_index', 'overall_rank']]
print(top10.to_string(index=False))

print(f"\n⚠️ BOTTOM 10 DISTRICTS (Overall Index):")
bottom10 = district_indices.nlargest(10, 'overall_rank')[['state', 'district', 'overall_index', 'overall_rank']]
print(bottom10.to_string(index=False))

# ============================================================================
# STATE-LEVEL RANKINGS
# ============================================================================
print("\n" + "="*80)
print("STATE-LEVEL RANKINGS")
print("="*80)

state_indices = df.groupby('state').agg({
    'digital_inclusion_index': 'mean',
    'service_quality_score': 'mean',
    'aadhaar_maturity_index': 'mean',
    'citizen_engagement_index': 'mean',
    'total_enrolments': 'sum',
    'total_all_updates': 'sum'
}).reset_index()

state_indices['overall_index'] = (
    state_indices['digital_inclusion_index'] +
    state_indices['service_quality_score'] +
    state_indices['aadhaar_maturity_index'] +
    state_indices['citizen_engagement_index']
) / 4

state_indices['overall_rank'] = state_indices['overall_index'].rank(ascending=False)

# Save rankings
state_indices.to_csv('outputs/tables/state_index_rankings.csv', index=False)
print(f"✅ Saved: outputs/tables/state_index_rankings.csv")

# Top 10 states
print(f"\n🏆 TOP 10 STATES (Overall Index):")
top10_states = state_indices.nsmallest(10, 'overall_rank')[['state', 'overall_index', 'overall_rank']]
print(top10_states.to_string(index=False))

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n[VIZ] Creating visualizations...")

# 1. Index Distributions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

indices = ['digital_inclusion_index', 'service_quality_score', 
           'aadhaar_maturity_index', 'citizen_engagement_index']
titles = ['Digital Inclusion Index', 'Service Quality Score',
          'Aadhaar Maturity Index', 'Citizen Engagement Index']
colors = ['steelblue', 'coral', 'seagreen', 'mediumpurple']

for idx, (index, title, color, ax) in enumerate(zip(indices, titles, colors, axes.flatten())):
    ax.hist(df[index], bins=50, color=color, edgecolor='black', alpha=0.7)
    ax.axvline(df[index].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df[index].mean():.1f}')
    ax.set_xlabel('Index Score', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/figures/composite_indices_distributions.png', dpi=300)
print("✅ Saved: outputs/figures/composite_indices_distributions.png")

# 2. State Comparison (Top 15 states)
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

for idx, (index, title, ax) in enumerate(zip(indices, titles, axes.flatten())):
    top15 = state_indices.nlargest(15, index)[['state', index]].sort_values(index)
    top15.plot(x='state', y=index, kind='barh', ax=ax, color=colors[idx], 
               edgecolor='black', legend=False)
    ax.set_xlabel('Index Score', fontsize=11, fontweight='bold')
    ax.set_ylabel('')
    ax.set_title(f'{title} - Top 15 States', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('outputs/figures/state_indices_top15.png', dpi=300)
print("✅ Saved: outputs/figures/state_indices_top15.png")

# 3. Correlation Heatmap
fig, ax = plt.subplots(figsize=(10, 8))

correlation_matrix = df[indices + ['updates_per_1000', 'saturation_ratio', 
                                    'mobility_indicator', 'enrolment_volatility_index']].corr()

sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
            ax=ax, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
ax.set_title('Composite Indices Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/figures/indices_correlation_heatmap.png', dpi=300)
print("✅ Saved: outputs/figures/indices_correlation_heatmap.png")

# 4. Scatter Matrix (Indices vs Overall)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, (index, title, color, ax) in enumerate(zip(indices, titles, colors, axes.flatten())):
    # District level
    ax.scatter(district_indices[index], district_indices['overall_index'],
               alpha=0.5, color=color, s=30, edgecolors='black', linewidth=0.5)
    
    # Simple trend line (correlation)
    corr = district_indices[[index, 'overall_index']].corr().iloc[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel(title, fontsize=11, fontweight='bold')
    ax.set_ylabel('Overall Index', fontsize=11, fontweight='bold')
    ax.set_title(f'{title} vs Overall', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/figures/indices_vs_overall.png', dpi=300)
print("✅ Saved: outputs/figures/indices_vs_overall.png")

# 5. District Performance Quadrants (Digital vs Service)
fig, ax = plt.subplots(figsize=(12, 8))

scatter = ax.scatter(district_indices['digital_inclusion_index'],
                     district_indices['service_quality_score'],
                     c=district_indices['overall_index'],
                     cmap='RdYlGn', s=100, alpha=0.6, edgecolors='black')

# Quadrant lines
median_digital = district_indices['digital_inclusion_index'].median()
median_service = district_indices['service_quality_score'].median()
ax.axvline(median_digital, color='gray', linestyle='--', linewidth=2)
ax.axhline(median_service, color='gray', linestyle='--', linewidth=2)

# Quadrant labels
ax.text(5, 95, 'Low Digital\nHigh Service', fontsize=10, ha='left', 
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
ax.text(95, 95, 'High Digital\nHigh Service', fontsize=10, ha='right',
        bbox=dict(boxstyle='round', facecolor='green', alpha=0.5))
ax.text(5, 5, 'Low Digital\nLow Service', fontsize=10, ha='left',
        bbox=dict(boxstyle='round', facecolor='red', alpha=0.5))
ax.text(95, 5, 'High Digital\nLow Service', fontsize=10, ha='right',
        bbox=dict(boxstyle='round', facecolor='orange', alpha=0.5))

ax.set_xlabel('Digital Inclusion Index', fontsize=12, fontweight='bold')
ax.set_ylabel('Service Quality Score', fontsize=12, fontweight='bold')
ax.set_title('District Performance Quadrants', fontsize=14, fontweight='bold')
plt.colorbar(scatter, label='Overall Index', ax=ax)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/figures/district_performance_quadrants.png', dpi=300)
print("✅ Saved: outputs/figures/district_performance_quadrants.png")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*80)
print("COMPOSITE INDICES SUMMARY")
print("="*80)

summary_stats = pd.DataFrame({
    'Index': titles,
    'Mean': [df[idx].mean() for idx in indices],
    'Median': [df[idx].median() for idx in indices],
    'Std': [df[idx].std() for idx in indices],
    'Min': [df[idx].min() for idx in indices],
    'Max': [df[idx].max() for idx in indices]
})

print("\n" + summary_stats.to_string(index=False))
summary_stats.to_csv('outputs/tables/indices_summary_statistics.csv', index=False)
print("\n✅ Saved: outputs/tables/indices_summary_statistics.csv")

# ============================================================================
# INSIGHTS
# ============================================================================
print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

print(f"\n📊 OVERALL STATISTICS:")
print(f"   ✅ Districts analyzed: {len(district_indices)}")
print(f"   ✅ States analyzed: {len(state_indices)}")
print(f"   ✅ Average Overall Index: {district_indices['overall_index'].mean():.2f}")

print(f"\n🏆 BEST PERFORMERS:")
print(f"   Digital Inclusion: {state_indices.nlargest(1, 'digital_inclusion_index')['state'].values[0]}")
print(f"   Service Quality: {state_indices.nlargest(1, 'service_quality_score')['state'].values[0]}")
print(f"   Aadhaar Maturity: {state_indices.nlargest(1, 'aadhaar_maturity_index')['state'].values[0]}")
print(f"   Citizen Engagement: {state_indices.nlargest(1, 'citizen_engagement_index')['state'].values[0]}")

print(f"\n⚠️ NEEDS IMPROVEMENT:")
print(f"   Digital Inclusion: {state_indices.nsmallest(1, 'digital_inclusion_index')['state'].values[0]}")
print(f"   Service Quality: {state_indices.nsmallest(1, 'service_quality_score')['state'].values[0]}")
print(f"   Aadhaar Maturity: {state_indices.nsmallest(1, 'aadhaar_maturity_index')['state'].values[0]}")
print(f"   Citizen Engagement: {state_indices.nsmallest(1, 'citizen_engagement_index')['state'].values[0]}")

print(f"\n📁 OUTPUTS:")
print(f"   - data/processed/aadhaar_with_indices.csv")
print(f"   - outputs/tables/district_index_rankings.csv")
print(f"   - outputs/tables/state_index_rankings.csv")
print(f"   - outputs/tables/indices_summary_statistics.csv")
print(f"   - outputs/figures/composite_indices_*.png (5 visualizations)")

print("\n" + "="*80)
print("COMPOSITE INDICES CREATION COMPLETE!")
print("="*80)
