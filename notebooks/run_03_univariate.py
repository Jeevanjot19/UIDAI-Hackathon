# Notebook 03 - Univariate Analysis (Standalone Script)
# This script performs comprehensive univariate analysis

import sys
sys.path.append('../src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create output directories
os.makedirs('../outputs/figures', exist_ok=True)
os.makedirs('../outputs/tables', exist_ok=True)

print("="*100)
print("NOTEBOOK 03: UNIVARIATE ANALYSIS")
print("="*100)

# STEP 1: Load Data
print("\\n[STEP 1] Loading feature-engineered dataset...")
df = pd.read_csv('../data/processed/aadhaar_with_features.csv')
df['date'] = pd.to_datetime(df['date'])

print(f"✓ Dataset shape: {df.shape}")
print(f"✓ Date range: {df['date'].min()} to {df['date'].max()}")
print(f"✓ States: {df['state'].nunique()}, Districts: {df['district'].nunique()}")

# STEP 2: Statistical Summary
print("\\n[STEP 2] Generating statistical summaries...")

# Replace inf values with NaN
df = df.replace([np.inf, -np.inf], np.nan)

key_features = [
    'enrolment_growth_rate', 'adult_enrolment_share', 'child_enrolment_share',
    'demographic_update_rate', 'biometric_update_rate', 'mobility_indicator',
    'digital_instability_index', 'identity_stability_score', 'update_burden_index',
    'manual_labor_proxy', 'lifecycle_transition_spike', 'seasonal_variance_score',
    'anomaly_severity_score', 'enrolment_volatility_index'
]

summary_stats = df[key_features].describe()
summary_stats.to_csv('../outputs/tables/03_univariate_summary_stats.csv')
print(f"✓ Statistical summary saved ({len(key_features)} features)")

# STEP 3: Distribution Plots
print("\\n[STEP 3] Creating distribution plots...")
fig, axes = plt.subplots(5, 3, figsize=(20, 18))
axes = axes.flatten()

for idx, feature in enumerate(key_features):
    ax = axes[idx]
    df[feature].hist(bins=50, ax=ax, alpha=0.6, edgecolor='black', density=True)
    df[feature].plot(kind='kde', ax=ax, color='red', linewidth=2)
    
    ax.set_title(f'{feature}\\n(μ={df[feature].mean():.4f}, σ={df[feature].std():.4f})', 
                 fontweight='bold', fontsize=10)
    ax.set_xlabel(feature, fontsize=8)
    ax.set_ylabel('Density', fontsize=8)
    ax.grid(alpha=0.3)

for idx in range(len(key_features), len(axes)):
    axes[idx].set_visible(False)

plt.suptitle('Distribution Analysis - All Key Features', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig('../outputs/figures/03_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Distribution plots saved")

# STEP 4: Time Series Analysis
print("\\n[STEP 4] Creating time series plots...")
societal_indicators = ['mobility_indicator', 'digital_instability_index', 
                      'identity_stability_score', 'update_burden_index']

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
axes = axes.flatten()

for idx, indicator in enumerate(societal_indicators):
    ax = axes[idx]
    national_trend = df.groupby('date')[indicator].mean()
    ax.plot(national_trend.index, national_trend.values, linewidth=2.5, color='darkblue')
    
    rolling_avg = national_trend.rolling(window=7).mean()
    ax.plot(rolling_avg.index, rolling_avg.values, linewidth=2, 
            color='red', linestyle='--', alpha=0.7, label='7-Day MA')
    
    ax.set_title(f'{indicator.replace("_", " ").title()}', fontweight='bold', fontsize=12)
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel(indicator, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

plt.suptitle('Time Series - Key Societal Indicators', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('../outputs/figures/03_timeseries.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Time series plots saved")

# STEP 5: State Rankings
print("\\n[STEP 5] Generating state rankings...")

# Identity Stability
state_stability = df.groupby('state')['identity_stability_score'].agg(['mean', 'std', 'count'])
state_stability = state_stability.sort_values('mean', ascending=False)
state_stability.to_csv('../outputs/tables/03_state_stability_rankings.csv')

# Mobility
state_mobility = df.groupby('state')['mobility_indicator'].agg(['mean', 'std', 'count'])
state_mobility = state_mobility.sort_values('mean', ascending=False)
state_mobility.to_csv('../outputs/tables/03_state_mobility_rankings.csv')

print(f"✓ Top 5 stable states: {', '.join(state_stability.head(5).index.tolist())}")
print(f"✓ Top 5 mobile states: {', '.join(state_mobility.head(5).index.tolist())}")

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

state_stability.head(15)['mean'].plot(kind='barh', ax=ax1, color='forestgreen', edgecolor='black')
ax1.set_title('Top 15 States - Highest Identity Stability', fontweight='bold', fontsize=13)
ax1.set_xlabel('Average Identity Stability Score')
ax1.invert_yaxis()
ax1.grid(axis='x', alpha=0.3)

state_mobility.head(20)['mean'].plot(kind='barh', ax=ax2, color='coral', edgecolor='black')
ax2.set_title('Top 20 States - Highest Mobility (Migration Hotspots)', fontweight='bold', fontsize=13)
ax2.set_xlabel('Average Mobility Indicator')
ax2.invert_yaxis()
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('../outputs/figures/03_state_rankings.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ State ranking visualizations saved")

# STEP 6: District Analysis
print("\\n[STEP 6] Analyzing district-level patterns...")
district_features = df.groupby(['state', 'district']).agg({
    'identity_stability_score': 'mean',
    'mobility_indicator': 'mean',
    'digital_instability_index': 'mean',
    'update_burden_index': 'mean'
}).reset_index()

district_features.to_csv('../outputs/tables/03_district_analysis.csv', index=False)
print(f"✓ District analysis saved ({len(district_features)} districts)")

# STEP 7: Age Group Analysis
print("\\n[STEP 7] Analyzing age group distributions...")
age_group_cols_enrol = [col for col in df.columns if 'enrolments_' in col and col != 'total_enrolments']
age_group_cols_demo = [col for col in df.columns if 'demographic_updates_' in col and col != 'total_demographic_updates']

total_enrol_by_age = df[age_group_cols_enrol].sum().sort_values(ascending=False)
total_demo_by_age = df[age_group_cols_demo].sum().sort_values(ascending=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

total_enrol_by_age.plot(kind='bar', ax=ax1, color='skyblue', edgecolor='black')
ax1.set_title('Total Enrolments by Age Group', fontweight='bold', fontsize=13)
ax1.set_xlabel('Age Group')
ax1.set_ylabel('Total Enrolments')
ax1.tick_params(axis='x', rotation=45)
ax1.grid(alpha=0.3, axis='y')

total_demo_by_age.plot(kind='bar', ax=ax2, color='lightcoral', edgecolor='black')
ax2.set_title('Total Demographic Updates by Age Group', fontweight='bold', fontsize=13)
ax2.set_xlabel('Age Group')
ax2.set_ylabel('Total Updates')
ax2.tick_params(axis='x', rotation=45)
ax2.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('../outputs/figures/03_age_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Age group analysis saved")

# STEP 8: Monthly Patterns
print("\\n[STEP 8] Analyzing monthly patterns...")
df['month'] = df['date'].dt.month
monthly_stats = df.groupby('month').agg({
    'total_enrolments': 'mean',
    'total_demographic_updates': 'mean',
    'mobility_indicator': 'mean',
    'identity_stability_score': 'mean'
}).reset_index()

month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
monthly_stats['month_name'] = monthly_stats['month'].apply(lambda x: month_names[x-1])
monthly_stats.to_csv('../outputs/tables/03_monthly_patterns.csv', index=False)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

axes[0,0].plot(monthly_stats['month_name'], monthly_stats['total_enrolments'], marker='o', linewidth=2.5)
axes[0,0].set_title('Avg Enrolments by Month', fontweight='bold')
axes[0,0].tick_params(axis='x', rotation=45)
axes[0,0].grid(alpha=0.3)

axes[0,1].plot(monthly_stats['month_name'], monthly_stats['total_demographic_updates'], marker='s', linewidth=2.5, color='red')
axes[0,1].set_title('Avg Demographic Updates by Month', fontweight='bold')
axes[0,1].tick_params(axis='x', rotation=45)
axes[0,1].grid(alpha=0.3)

axes[1,0].plot(monthly_stats['month_name'], monthly_stats['mobility_indicator'], marker='^', linewidth=2.5, color='orange')
axes[1,0].set_title('Avg Mobility by Month', fontweight='bold')
axes[1,0].tick_params(axis='x', rotation=45)
axes[1,0].grid(alpha=0.3)

axes[1,1].plot(monthly_stats['month_name'], monthly_stats['identity_stability_score'], marker='D', linewidth=2.5, color='green')
axes[1,1].set_title('Avg Identity Stability by Month', fontweight='bold')
axes[1,1].tick_params(axis='x', rotation=45)
axes[1,1].grid(alpha=0.3)

plt.suptitle('Monthly Patterns - Seasonal Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('../outputs/figures/03_monthly_patterns.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Monthly patterns saved")

# STEP 9: Key Insights
print("\\n[STEP 9] Extracting key insights...")
insights = []

low_stability_states = state_stability.tail(5).index.tolist()
insights.append(f"1. PRIORITY INTERVENTION: {', '.join(low_stability_states)} show lowest identity stability")

high_mobility_states = state_mobility.head(5).index.tolist()
insights.append(f"2. MIGRATION HOTSPOTS: {', '.join(high_mobility_states)} have highest mobility indicators")

dominant_age = total_enrol_by_age.idxmax()
insights.append(f"3. AGE PATTERN: {dominant_age} shows highest enrolment activity")

peak_month = monthly_stats.loc[monthly_stats['total_enrolments'].idxmax(), 'month_name']
insights.append(f"4. SEASONAL PEAK: {peak_month} has maximum enrolment activity")

with open('../outputs/tables/03_key_insights.txt', 'w') as f:
    f.write("KEY INSIGHTS - UNIVARIATE ANALYSIS\\n")
    f.write("="*100 + "\\n\\n")
    for insight in insights:
        f.write(insight + "\\n")

print("\\n" + "="*100)
print("UNIVARIATE ANALYSIS COMPLETE")
print("="*100)
for insight in insights:
    print(f"  • {insight}")

print("\\nOutputs Generated:")
print("  📊 6 visualization files → outputs/figures/03_*.png")
print("  📈 5 statistical tables → outputs/tables/03_*.csv")
print("  💡 1 insights document → outputs/tables/03_key_insights.txt")
print("\\n✅ Notebook 03 analysis complete!")
