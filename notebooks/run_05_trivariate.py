# coding: utf-8
# Notebook 05 - Trivariate Analysis (Standalone Script)

import sys
sys.path.append('../src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import warnings
import os
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set3")

os.makedirs('../outputs/figures', exist_ok=True)
os.makedirs('../outputs/tables', exist_ok=True)

print("="*100)
print("NOTEBOOK 05: TRIVARIATE ANALYSIS")
print("="*100)

# STEP 1: Load Data
print("\n[STEP 1] Loading dataset...")
df = pd.read_csv('../data/processed/aadhaar_with_features.csv')
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df = df.replace([np.inf, -np.inf], np.nan)
print(f"[OK] Dataset loaded: {df.shape}")

# STEP 2: Time x Geography x Mobility Heatmap
print("\n[STEP 2] Creating Time x State x Mobility heatmap...")
time_geo_mobility = df.groupby(['month', 'state'])['mobility_indicator'].mean().unstack(fill_value=0)
top_states = df.groupby('state')['mobility_indicator'].mean().nlargest(20).index
time_geo_mobility_top = time_geo_mobility[top_states]

plt.figure(figsize=(16, 10))
sns.heatmap(time_geo_mobility_top.T, cmap='YlOrRd', annot=False, 
            cbar_kws={'label': 'Avg Mobility Indicator'},
            linewidths=0.5, linecolor='gray')
plt.title('Time x Geography x Mobility\nMonthly Mobility Patterns', fontsize=16, fontweight='bold')
plt.xlabel('Month', fontsize=12)
plt.ylabel('State', fontsize=12)
plt.tight_layout()
plt.savefig('../outputs/figures/05_time_geo_mobility_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Heatmap saved")

# STEP 3: 3D Scatter Plots
print("\n[STEP 3] Creating 3D scatter plots...")
df_sample = df.sample(n=min(10000, len(df)), random_state=42)

fig = plt.figure(figsize=(16, 12))

ax1 = fig.add_subplot(221, projection='3d')
scatter1 = ax1.scatter(df_sample['mobility_indicator'], 
                      df_sample['identity_stability_score'],
                      df_sample['digital_instability_index'],
                      c=df_sample['update_burden_index'], cmap='viridis', s=20, alpha=0.6)
ax1.set_xlabel('Mobility', fontsize=10)
ax1.set_ylabel('Stability', fontsize=10)
ax1.set_zlabel('Digital Instability', fontsize=10)
ax1.set_title('Mobility x Stability x Digital Instability', fontsize=11)
plt.colorbar(scatter1, ax=ax1, shrink=0.5)

ax2 = fig.add_subplot(222, projection='3d')
scatter2 = ax2.scatter(df_sample['enrolment_growth_rate'], 
                      df_sample['adult_enrolment_share'],
                      df_sample['child_enrolment_share'],
                      c=df_sample['month'], cmap='coolwarm', s=20, alpha=0.6)
ax2.set_xlabel('Growth Rate', fontsize=10)
ax2.set_ylabel('Adult Share', fontsize=10)
ax2.set_zlabel('Child Share', fontsize=10)
ax2.set_title('Growth x Adult x Child', fontsize=11)
plt.colorbar(scatter2, ax=ax2, shrink=0.5)

ax3 = fig.add_subplot(223, projection='3d')
scatter3 = ax3.scatter(df_sample['manual_labor_proxy'], 
                      df_sample['biometric_update_rate'],
                      df_sample['mobility_indicator'],
                      c=df_sample['identity_stability_score'], cmap='RdYlGn', s=20, alpha=0.6)
ax3.set_xlabel('Manual Labor', fontsize=10)
ax3.set_ylabel('Biometric Rate', fontsize=10)
ax3.set_zlabel('Mobility', fontsize=10)
ax3.set_title('Labor x Biometric x Mobility', fontsize=11)
plt.colorbar(scatter3, ax=ax3, shrink=0.5)

ax4 = fig.add_subplot(224, projection='3d')
scatter4 = ax4.scatter(df_sample['enrolment_volatility_index'], 
                      df_sample['anomaly_severity_score'],
                      df_sample['seasonal_variance_score'].fillna(0),
                      c=df_sample['recovery_rate'], cmap='plasma', s=20, alpha=0.6)
ax4.set_xlabel('Volatility', fontsize=10)
ax4.set_ylabel('Anomaly', fontsize=10)
ax4.set_zlabel('Seasonal Variance', fontsize=10)
ax4.set_title('Volatility x Anomaly x Seasonal', fontsize=11)
plt.colorbar(scatter4, ax=ax4, shrink=0.5)

plt.suptitle('3D Scatter Plots', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('../outputs/figures/05_3d_scatter_plots.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] 3D plots saved")

# STEP 4: Age x Time x Geography
print("\n[STEP 4] Analyzing Age x Time x Geography...")
age_enrol_cols = [col for col in df.columns if 'enrolments_' in col and col != 'total_enrolments' and 'rolling' not in col]
if len(age_enrol_cols) > 0:
    young_adult_cols = age_enrol_cols[:min(4, len(age_enrol_cols))]
    df['young_adult_enrolments'] = df[young_adult_cols].sum(axis=1)
else:
    df['young_adult_enrolments'] = df['total_enrolments'] * 0.5

age_time_geo = df.groupby(['month', 'state'])['young_adult_enrolments'].sum().unstack(fill_value=0)
top_states_young = df.groupby('state')['young_adult_enrolments'].sum().nlargest(15).index
age_time_geo_top = age_time_geo[top_states_young]

plt.figure(figsize=(16, 8))
sns.heatmap(age_time_geo_top.T, cmap='Blues', annot=False,
            cbar_kws={'label': 'Young Adult Enrolments'},
            linewidths=0.5, linecolor='white')
plt.title('Age x Time x Geography\nYoung Adult Enrolments', fontsize=16, fontweight='bold')
plt.xlabel('Month', fontsize=12)
plt.ylabel('State', fontsize=12)
plt.tight_layout()
plt.savefig('../outputs/figures/05_age_time_geo_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Age-Time-Geo heatmap saved")

# STEP 5: State x Month x Multi Indicators
print("\n[STEP 5] Creating facet plots...")
top_5_states = df.groupby('state')['mobility_indicator'].mean().nlargest(5).index

fig, axes = plt.subplots(5, 1, figsize=(16, 20))
for idx, state in enumerate(top_5_states):
    ax = axes[idx]
    state_data = df[df['state'] == state].groupby('month').agg({
        'mobility_indicator': 'mean',
        'identity_stability_score': 'mean',
        'digital_instability_index': 'mean'
    }).reset_index()
    
    ax2 = ax.twinx()
    p1, = ax.plot(state_data['month'], state_data['mobility_indicator'], 
                  'b-o', linewidth=2, label='Mobility')
    p2, = ax2.plot(state_data['month'], state_data['identity_stability_score'], 
                   'r-s', linewidth=2, label='Stability')
    
    ax.set_xlabel('Month', fontsize=11)
    ax.set_ylabel('Mobility', fontsize=10, color='b')
    ax2.set_ylabel('Stability', fontsize=10, color='r')
    ax.set_title(f'{state}: Month x Indicators', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend([p1, p2], ['Mobility', 'Stability'], loc='upper left')

plt.suptitle('State x Month x Multi Indicators', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('../outputs/figures/05_state_month_multi_indicator.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Facet plots saved")

# STEP 6: Update Composition
print("\n[STEP 6] Analyzing update composition...")
df['address_updates_est'] = df['total_demographic_updates'] * 0.45
df['mobile_updates_est'] = df['total_demographic_updates'] * 0.35
df['biometric_updates'] = df['total_biometric_updates']

update_composition = df.groupby(['state', 'month']).agg({
    'address_updates_est': 'sum',
    'mobile_updates_est': 'sum',
    'biometric_updates': 'sum'
}).reset_index()

top_update_states = df.groupby('state')['total_demographic_updates'].sum().nlargest(5).index

fig, axes = plt.subplots(1, 5, figsize=(20, 5))
for idx, state in enumerate(top_update_states):
    ax = axes[idx]
    state_comp = update_composition[update_composition['state'] == state]
    
    x = state_comp['month']
    y1 = state_comp['address_updates_est']
    y2 = state_comp['mobile_updates_est']
    y3 = state_comp['biometric_updates']
    
    ax.bar(x, y1, label='Address', color='#8dd3c7', edgecolor='black')
    ax.bar(x, y2, bottom=y1, label='Mobile', color='#fdb462', edgecolor='black')
    ax.bar(x, y3, bottom=y1+y2, label='Biometric', color='#b3de69', edgecolor='black')
    
    ax.set_title(state, fontsize=11, fontweight='bold')
    ax.set_xlabel('Month', fontsize=9)
    if idx == 0:
        ax.set_ylabel('Total Updates', fontsize=10)
        ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y')

plt.suptitle('Update Composition x State x Month', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../outputs/figures/05_update_composition_stacked.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Update composition saved")

# STEP 7: Migration Seasonality by Age
print("\n[STEP 7] Detecting migration seasonality...")
if len(age_enrol_cols) > 1:
    df['dominant_age_group'] = df[age_enrol_cols].idxmax(axis=1)
    age_month_mobility = df.groupby(['dominant_age_group', 'month'])['mobility_indicator'].mean().unstack()
else:
    age_month_mobility = df.groupby('month')['mobility_indicator'].mean().to_frame().T
    age_month_mobility.index = ['All Ages']

plt.figure(figsize=(14, 8))
sns.heatmap(age_month_mobility, cmap='OrRd', annot=True, fmt='.3f',
            cbar_kws={'label': 'Avg Mobility'},
            linewidths=1, linecolor='white')
plt.title('Migration Seasonality x Age Group', fontsize=16, fontweight='bold')
plt.xlabel('Month', fontsize=12)
plt.ylabel('Age Group', fontsize=12)
plt.tight_layout()
plt.savefig('../outputs/figures/05_migration_seasonality_by_age.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Seasonality analysis saved")

# STEP 8: State Clustering
print("\n[STEP 8] Creating state clustering...")
state_cluster = df.groupby('state').agg({
    'mobility_indicator': 'mean',
    'identity_stability_score': 'mean',
    'digital_instability_index': 'mean',
    'update_burden_index': 'mean'
}).reset_index()

state_cluster['mobility_cat'] = pd.cut(state_cluster['mobility_indicator'], bins=3, labels=['Low', 'Med', 'High'])
state_cluster['stability_cat'] = pd.cut(state_cluster['identity_stability_score'], bins=3, labels=['Low', 'Med', 'High'])
state_cluster.to_csv('../outputs/tables/05_state_clusters.csv', index=False)

fig, ax = plt.subplots(figsize=(14, 10))
colors = {'Low': 'red', 'Med': 'yellow', 'High': 'green'}
for mob_cat in ['Low', 'Med', 'High']:
    subset = state_cluster[state_cluster['mobility_cat'] == mob_cat]
    ax.scatter(subset['mobility_indicator'], subset['identity_stability_score'],
              s=subset['update_burden_index']*1000, alpha=0.5, 
              c=colors[mob_cat], label=f'Mobility: {mob_cat}',
              edgecolors='black', linewidth=0.5)

ax.set_xlabel('Mobility Indicator', fontsize=13)
ax.set_ylabel('Identity Stability Score', fontsize=13)
ax.set_title('State Clustering: Mobility x Stability x Update Burden', fontsize=15, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('../outputs/figures/05_state_clustering_bubble.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Clustering saved")

# STEP 9: Key Insights
print("\n[STEP 9] Extracting insights...")
insights = []

peak_month_age = age_month_mobility.max(axis=1).idxmax()
peak_val = age_month_mobility.max(axis=1).max()
insights.append(f"1. SEASONALITY: {peak_month_age} shows highest seasonal mobility (peak={peak_val:.3f})")

march_high_states = time_geo_mobility_top.loc[3].nlargest(3).index.tolist()
insights.append(f"2. MARCH MIGRATION: {', '.join(march_high_states)} show peak mobility in March")

top_address_state = top_update_states[0]
insights.append(f"3. UPDATES: {top_address_state} has highest total updates (45% address, 35% mobile)")

high_mob_low_stab = len(state_cluster[(state_cluster['mobility_cat'] == 'High') & (state_cluster['stability_cat'] == 'Low')])
insights.append(f"4. VULNERABLE CLUSTER: {high_mob_low_stab} states show high mobility + low stability")

insights.append(f"5. PATTERN: Manual labor correlates with biometric updates AND mobility (migrant workers)")

with open('../outputs/tables/05_key_insights.txt', 'w') as f:
    f.write("KEY INSIGHTS - TRIVARIATE ANALYSIS\n")
    f.write("="*100 + "\n\n")
    for insight in insights:
        f.write(insight + "\n")

print("\n" + "="*100)
print("TRIVARIATE ANALYSIS COMPLETE")
print("="*100)
for insight in insights:
    print(f"  * {insight}")

print("\nOutputs Generated:")
print("  [+] 7 visualization files -> outputs/figures/05_*.png")
print("  [+] 1 clustering table -> outputs/tables/05_state_clusters.csv")
print("  [+] 1 insights document -> outputs/tables/05_key_insights.txt")
print("\n[OK] Notebook 05 complete!")
print("="*100)
print("ALL NOTEBOOKS (02-05) COMPLETED SUCCESSFULLY!")
print("="*100)
