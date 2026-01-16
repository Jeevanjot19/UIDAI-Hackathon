# Notebook 04 - Bivariate Analysis (Standalone Script)
# This script performs comprehensive bivariate analysis

import sys
sys.path.append('../src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from scipy import stats
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

os.makedirs('../outputs/figures', exist_ok=True)
os.makedirs('../outputs/tables', exist_ok=True)

print("="*100)
print("NOTEBOOK 04: BIVARIATE ANALYSIS")
print("="*100)

# STEP 1: Load Data
print("\\n[STEP 1] Loading feature-engineered dataset...")
df = pd.read_csv('../data/processed/aadhaar_with_features.csv')
df['date'] = pd.to_datetime(df['date'])
# Replace inf values
df = df.replace([np.inf, -np.inf], np.nan)
print(f"✓ Dataset loaded: {df.shape}")

# STEP 2: Correlation Analysis
print("\\n[STEP 2] Computing correlation matrix...")
key_features = [
    'enrolment_growth_rate', 'adult_enrolment_share', 'mobility_indicator',
    'digital_instability_index', 'identity_stability_score', 'update_burden_index',
    'manual_labor_proxy', 'seasonal_variance_score', 'anomaly_severity_score',
    'enrolment_volatility_index', 'demographic_update_rate', 'biometric_update_rate'
]

corr_matrix = df[key_features].corr()
corr_matrix.to_csv('../outputs/tables/04_correlation_matrix.csv')

# Correlation heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
            vmin=-1, vmax=1)
plt.title('Feature Correlation Matrix\\nBivariate Relationships', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('../outputs/figures/04_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Correlation matrix saved")

# STEP 3: Scatter Plots - Key Relationships
print("\\n[STEP 3] Creating scatter plots for key relationships...")

key_pairs = [
    ('mobility_indicator', 'identity_stability_score', 'Mobility vs Identity Stability'),
    ('digital_instability_index', 'update_burden_index', 'Digital Instability vs Update Burden'),
    ('manual_labor_proxy', 'biometric_update_rate', 'Manual Labor vs Biometric Updates'),
    ('enrolment_growth_rate', 'adult_enrolment_share', 'Growth Rate vs Adult Share'),
]

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
axes = axes.flatten()

for idx, (feat1, feat2, title) in enumerate(key_pairs):
    ax = axes[idx]
    
    # Remove rows with NaN in either column
    valid_mask = df[[feat1, feat2]].notna().all(axis=1)
    df_valid = df[valid_mask]
    
    # Scatter plot
    ax.scatter(df_valid[feat1], df_valid[feat2], alpha=0.3, s=10, c='steelblue')
    
    # Regression line
    if len(df_valid) > 0:
        z = np.polyfit(df_valid[feat1], df_valid[feat2], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df_valid[feat1].min(), df_valid[feat1].max(), 100)
        ax.plot(x_line, p(x_line), "r--", linewidth=2, label=f'y={z[0]:.4f}x+{z[1]:.4f}')
    
    # Calculate correlation
    if len(df_valid) > 0:
        corr, pval = stats.pearsonr(df_valid[feat1], df_valid[feat2])
    else:
        corr, pval = 0, 1
    ax.text(0.05, 0.95, f'r={corr:.3f}, p={pval:.2e}', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel(feat1, fontsize=11)
    ax.set_ylabel(feat2, fontsize=11)
    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)

plt.suptitle('Scatter Plots - Key Feature Relationships', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('../outputs/figures/04_scatter_plots.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Scatter plots saved")

# STEP 4: Cross-tabulation by Geography
print("\\n[STEP 4] Creating geographic cross-tabulations...")

# Mobility vs Stability by State
state_cross = df.groupby('state').agg({
    'mobility_indicator': 'mean',
    'identity_stability_score': 'mean',
    'digital_instability_index': 'mean'
}).reset_index()

state_cross.to_csv('../outputs/tables/04_state_cross_analysis.csv', index=False)

# Create quadrant plot
fig, ax = plt.subplots(figsize=(14, 10))

scatter = ax.scatter(state_cross['mobility_indicator'], 
                    state_cross['identity_stability_score'],
                    s=100, alpha=0.6, c=state_cross['digital_instability_index'],
                    cmap='YlOrRd', edgecolors='black', linewidth=0.5)

# Add state labels for extremes
top_mobile = state_cross.nlargest(5, 'mobility_indicator')
low_stable = state_cross.nsmallest(5, 'identity_stability_score')

for _, row in top_mobile.iterrows():
    ax.annotate(row['state'], (row['mobility_indicator'], row['identity_stability_score']),
                fontsize=8, alpha=0.7)

ax.set_xlabel('Mobility Indicator (Migration Proxy)', fontsize=12, fontweight='bold')
ax.set_ylabel('Identity Stability Score', fontsize=12, fontweight='bold')
ax.set_title('State-wise: Mobility vs Identity Stability\\n(Color = Digital Instability)', 
             fontsize=14, fontweight='bold')
ax.grid(alpha=0.3)

# Add quadrant lines
mob_median = state_cross['mobility_indicator'].median()
stab_median = state_cross['identity_stability_score'].median()
ax.axvline(mob_median, color='red', linestyle='--', alpha=0.5, label='Median Mobility')
ax.axhline(stab_median, color='blue', linestyle='--', alpha=0.5, label='Median Stability')
ax.legend()

plt.colorbar(scatter, ax=ax, label='Digital Instability Index')
plt.tight_layout()
plt.savefig('../outputs/figures/04_mobility_stability_quadrant.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Geographic cross-tabulation saved")

# STEP 5: Time-based Bivariate Analysis
print("\\n[STEP 5] Analyzing temporal relationships...")

# Monthly correlation trends
df['month'] = df['date'].dt.month
monthly_corr = []

for month in range(3, 13):  # March to December
    month_data = df[df['month'] == month]
    if len(month_data) > 100:
        corr, _ = stats.pearsonr(month_data['mobility_indicator'].dropna(), 
                                 month_data['identity_stability_score'].dropna())
        monthly_corr.append({'month': month, 'correlation': corr})

monthly_corr_df = pd.DataFrame(monthly_corr)
monthly_corr_df.to_csv('../outputs/tables/04_monthly_correlations.csv', index=False)

fig, ax = plt.subplots(figsize=(12, 6))
month_names = ['Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ax.plot(month_names[:len(monthly_corr)], monthly_corr_df['correlation'], 
        marker='o', linewidth=2.5, markersize=10, color='darkgreen')
ax.set_title('Mobility-Stability Correlation Over Months', fontsize=14, fontweight='bold')
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('Correlation Coefficient', fontsize=12)
ax.grid(alpha=0.3)
ax.axhline(0, color='red', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('../outputs/figures/04_temporal_correlation.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Temporal relationship analysis saved")

# STEP 6: Age Group Comparisons
print("\\n[STEP 6] Comparing age group patterns...")

age_enrol_cols = [col for col in df.columns if 'enrolments_' in col and col != 'total_enrolments']
age_demo_cols = [col for col in df.columns if 'demographic_updates_' in col and col != 'total_demographic_updates']

# Create pairwise comparison
age_comparison = pd.DataFrame({
    'age_group': [col.replace('enrolments_', '') for col in age_enrol_cols],
    'total_enrolments': [df[col].sum() for col in age_enrol_cols],
    'total_updates': [df[col.replace('enrolments_', 'demographic_updates_')].sum() 
                     if col.replace('enrolments_', 'demographic_updates_') in df.columns else 0 
                     for col in age_enrol_cols]
})

age_comparison['update_to_enrolment_ratio'] = age_comparison['total_updates'] / (age_comparison['total_enrolments'] + 1)
age_comparison.to_csv('../outputs/tables/04_age_group_comparison.csv', index=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Enrolments vs Updates
ax1.scatter(age_comparison['total_enrolments'], age_comparison['total_updates'], 
           s=200, alpha=0.6, c=range(len(age_comparison)), cmap='viridis', edgecolors='black')
for idx, row in age_comparison.iterrows():
    ax1.annotate(row['age_group'], (row['total_enrolments'], row['total_updates']), 
                fontsize=9, alpha=0.8)
ax1.set_xlabel('Total Enrolments', fontsize=11, fontweight='bold')
ax1.set_ylabel('Total Demographic Updates', fontsize=11, fontweight='bold')
ax1.set_title('Enrolments vs Updates by Age Group', fontsize=13, fontweight='bold')
ax1.grid(alpha=0.3)

# Update ratio
age_comparison.plot(x='age_group', y='update_to_enrolment_ratio', kind='bar', 
                   ax=ax2, color='teal', edgecolor='black', legend=False)
ax2.set_title('Update-to-Enrolment Ratio by Age', fontsize=13, fontweight='bold')
ax2.set_xlabel('Age Group', fontsize=11)
ax2.set_ylabel('Ratio', fontsize=11)
ax2.tick_params(axis='x', rotation=45)
ax2.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('../outputs/figures/04_age_group_bivariate.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Age group comparisons saved")

# STEP 7: Identify Strong Correlations
print("\\n[STEP 7] Identifying strong correlations...")

# Get correlations above threshold
strong_corrs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        corr_val = corr_matrix.iloc[i, j]
        if abs(corr_val) > 0.5:  # Strong correlation threshold
            strong_corrs.append({
                'feature_1': corr_matrix.columns[i],
                'feature_2': corr_matrix.columns[j],
                'correlation': corr_val,
                'strength': 'Strong Positive' if corr_val > 0 else 'Strong Negative'
            })

strong_corrs_df = pd.DataFrame(strong_corrs).sort_values('correlation', key=abs, ascending=False)
strong_corrs_df.to_csv('../outputs/tables/04_strong_correlations.csv', index=False)

print(f"✓ Found {len(strong_corrs)} strong correlations (|r| > 0.5)")
if len(strong_corrs) > 0:
    print("\\nTop 5 strongest correlations:")
    for idx, row in strong_corrs_df.head(5).iterrows():
        print(f"  • {row['feature_1']} ↔ {row['feature_2']}: r={row['correlation']:.3f}")

# STEP 8: Key Insights
print("\\n[STEP 8] Extracting key insights...")
insights = []

# Mobility-Stability relationship
mob_stab_corr = corr_matrix.loc['mobility_indicator', 'identity_stability_score']
insights.append(f"1. MOBILITY-STABILITY: r={mob_stab_corr:.3f} - {'Negative' if mob_stab_corr < 0 else 'Positive'} relationship")

# Digital instability patterns
dig_update_corr = corr_matrix.loc['digital_instability_index', 'update_burden_index']
insights.append(f"2. DIGITAL-UPDATE BURDEN: r={dig_update_corr:.3f} - Mobile churn correlates with update load")

# Age patterns
high_ratio_age = age_comparison.loc[age_comparison['update_to_enrolment_ratio'].idxmax(), 'age_group']
insights.append(f"3. AGE PATTERN: {high_ratio_age} has highest update-to-enrolment ratio")

# Geographic clusters
high_mob_low_stab = len(state_cross[(state_cross['mobility_indicator'] > mob_median) & 
                                    (state_cross['identity_stability_score'] < stab_median)])
insights.append(f"4. GEOGRAPHIC CLUSTERS: {high_mob_low_stab} states show high mobility + low stability")

with open('../outputs/tables/04_key_insights.txt', 'w') as f:
    f.write("KEY INSIGHTS - BIVARIATE ANALYSIS\\n")
    f.write("="*100 + "\\n\\n")
    for insight in insights:
        f.write(insight + "\\n")

print("\\n" + "="*100)
print("BIVARIATE ANALYSIS COMPLETE")
print("="*100)
for insight in insights:
    print(f"  • {insight}")

print("\\nOutputs Generated:")
print("  📊 6 visualization files → outputs/figures/04_*.png")
print("  📈 5 statistical tables → outputs/tables/04_*.csv")
print("  💡 1 insights document → outputs/tables/04_key_insights.txt")
print("\\n✅ Notebook 04 analysis complete!")
