"""
Advanced Clustering & Anomaly Detection
Days 4-5 of Implementation Plan

Goals:
1. K-Means clustering for district segmentation
2. DBSCAN for geographic anomaly detection
3. Enhanced Isolation Forest with feature importance
4. Hierarchical clustering for state grouping
5. Cluster profiling and insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import scipy.cluster.hierarchy as shc
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ADVANCED CLUSTERING & ANOMALY DETECTION")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[LOAD] Loading dataset...")
df = pd.read_csv('data/processed/aadhaar_with_advanced_features.csv')
df['date'] = pd.to_datetime(df['date'])

print(f"Dataset: {df.shape}")

# ============================================================================
# PREPARE FEATURES FOR CLUSTERING
# ============================================================================
print("\n" + "="*80)
print("FEATURE PREPARATION")
print("="*80)

# Aggregate by district (for geographic clustering)
print("\nAggregating by district...")
district_features = df.groupby(['state', 'district']).agg({
    'total_enrolments': 'sum',
    'total_demographic_updates': 'sum',
    'total_biometric_updates': 'sum',
    'updates_per_1000': 'mean',
    'update_burden_index': 'mean',
    'mobility_indicator': 'mean',
    'digital_instability_index': 'mean',
    'saturation_ratio': 'mean',
    'growth_acceleration': 'mean',
    'enrolment_volatility_index': 'mean',
    'child_update_compliance': 'mean',
    'policy_violation_score': 'sum',
    'gender_parity_score': 'mean'
}).reset_index()

print(f"Districts: {len(district_features)}")

# Select clustering features
cluster_cols = [
    'total_enrolments',
    'total_demographic_updates',
    'total_biometric_updates',
    'updates_per_1000',
    'mobility_indicator',
    'digital_instability_index',
    'saturation_ratio',
    'enrolment_volatility_index',
    'update_burden_index'
]

X_cluster = district_features[cluster_cols].fillna(0)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

print(f"✅ Features scaled: {X_scaled.shape}")

# ============================================================================
# 1. K-MEANS CLUSTERING (District Segmentation)
# ============================================================================
print("\n" + "="*80)
print("K-MEANS CLUSTERING")
print("="*80)

# Find optimal number of clusters (elbow method)
print("\n[ELBOW] Finding optimal K...")
inertias = []
silhouettes = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X_scaled, kmeans.labels_))

# Plot elbow curve
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('Number of Clusters (K)', fontsize=12)
ax1.set_ylabel('Inertia', fontsize=12)
ax1.set_title('Elbow Method', fontsize=14, fontweight='bold')
ax1.grid(alpha=0.3)

ax2.plot(K_range, silhouettes, 'ro-', linewidth=2, markersize=8)
ax2.set_xlabel('Number of Clusters (K)', fontsize=12)
ax2.set_ylabel('Silhouette Score', fontsize=12)
ax2.set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/figures/clustering_optimal_k.png', dpi=300)
print("✅ Saved: outputs/figures/clustering_optimal_k.png")

# Use optimal K (based on elbow/silhouette)
optimal_k = 5  # Can adjust based on plot
print(f"\n[K-MEANS] Training with K={optimal_k}...")

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
district_features['cluster'] = kmeans.fit_predict(X_scaled)

print(f"✅ Clustering complete!")
print(f"Silhouette Score: {silhouette_score(X_scaled, district_features['cluster']):.4f}")
print(f"Davies-Bouldin Index: {davies_bouldin_score(X_scaled, district_features['cluster']):.4f}")

# Cluster distribution
print(f"\nCluster Distribution:")
print(district_features['cluster'].value_counts().sort_index())

# ============================================================================
# CLUSTER PROFILING
# ============================================================================
print("\n" + "="*80)
print("CLUSTER PROFILING")
print("="*80)

cluster_profiles = district_features.groupby('cluster')[cluster_cols].mean()
print("\nCluster Centers (Mean Values):")
print(cluster_profiles.round(2))

# Assign cluster names based on characteristics
cluster_names = {
    0: "High Activity Urban",
    1: "Low Activity Rural",
    2: "Moderate Balanced",
    3: "High Mobility Transient",
    4: "Oversaturated Stable"
}

district_features['cluster_name'] = district_features['cluster'].map(cluster_names)

# Save cluster assignments
cluster_summary = district_features[['state', 'district', 'cluster', 'cluster_name'] + cluster_cols]
cluster_summary.to_csv('outputs/tables/district_clusters.csv', index=False)
print("\n✅ Saved: outputs/tables/district_clusters.csv")

# ============================================================================
# VISUALIZE CLUSTERS (PCA)
# ============================================================================
print("\n[VIZ] Creating cluster visualizations...")

# PCA for 2D visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                     c=district_features['cluster'], 
                     cmap='viridis', 
                     s=100, 
                     alpha=0.6,
                     edgecolors='black')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
plt.title('District Clustering (K-Means, K=5)', fontsize=14, fontweight='bold')
plt.colorbar(scatter, label='Cluster')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/figures/clustering_pca_visualization.png', dpi=300)
print("✅ Saved: outputs/figures/clustering_pca_visualization.png")

# Cluster characteristics heatmap
plt.figure(figsize=(12, 6))
cluster_profiles_normalized = (cluster_profiles - cluster_profiles.mean()) / cluster_profiles.std()
sns.heatmap(cluster_profiles_normalized.T, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
            xticklabels=[cluster_names[i] for i in range(optimal_k)],
            yticklabels=cluster_cols)
plt.title('Cluster Profiles (Standardized)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/figures/clustering_heatmap.png', dpi=300)
print("✅ Saved: outputs/figures/clustering_heatmap.png")

# ============================================================================
# 2. DBSCAN (Density-based Anomaly Detection)
# ============================================================================
print("\n" + "="*80)
print("DBSCAN ANOMALY DETECTION")
print("="*80)

dbscan = DBSCAN(eps=0.5, min_samples=5)
district_features['dbscan_cluster'] = dbscan.fit_predict(X_scaled)

n_clusters_dbscan = len(set(district_features['dbscan_cluster'])) - (1 if -1 in district_features['dbscan_cluster'] else 0)
n_outliers = (district_features['dbscan_cluster'] == -1).sum()

print(f"✅ DBSCAN complete!")
print(f"Clusters found: {n_clusters_dbscan}")
print(f"Outliers detected: {n_outliers} ({n_outliers/len(district_features)*100:.1f}%)")

# Outlier districts
outlier_districts = district_features[district_features['dbscan_cluster'] == -1][['state', 'district'] + cluster_cols]
outlier_districts.to_csv('outputs/tables/dbscan_outlier_districts.csv', index=False)
print(f"✅ Saved: outputs/tables/dbscan_outlier_districts.csv")

# ============================================================================
# 3. ENHANCED ISOLATION FOREST
# ============================================================================
print("\n" + "="*80)
print("ISOLATION FOREST ANOMALY DETECTION")
print("="*80)

iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.05,  # 5% expected anomalies
    random_state=42,
    n_jobs=-1
)

district_features['anomaly_score'] = iso_forest.fit_predict(X_scaled)
district_features['anomaly_prob'] = iso_forest.score_samples(X_scaled)

n_anomalies = (district_features['anomaly_score'] == -1).sum()
print(f"✅ Isolation Forest complete!")
print(f"Anomalies detected: {n_anomalies} ({n_anomalies/len(district_features)*100:.1f}%)")

# Top anomalies
anomalies = district_features[district_features['anomaly_score'] == -1].sort_values('anomaly_prob')
top_anomalies = anomalies.head(20)[['state', 'district', 'anomaly_prob', 'cluster_name'] + cluster_cols]
top_anomalies.to_csv('outputs/tables/isolation_forest_anomalies.csv', index=False)
print(f"✅ Saved: outputs/tables/isolation_forest_anomalies.csv")

print(f"\nTop 10 Anomalous Districts:")
print(top_anomalies[['state', 'district', 'anomaly_prob', 'cluster_name']].head(10).to_string(index=False))

# ============================================================================
# 4. HIERARCHICAL CLUSTERING (State-level)
# ============================================================================
print("\n" + "="*80)
print("HIERARCHICAL CLUSTERING (STATES)")
print("="*80)

# Aggregate by state
state_features = df.groupby('state').agg({
    'total_enrolments': 'sum',
    'total_demographic_updates': 'sum',
    'total_biometric_updates': 'sum',
    'updates_per_1000': 'mean',
    'mobility_indicator': 'mean',
    'saturation_ratio': 'mean'
}).reset_index()

X_state = state_features[['total_enrolments', 'total_demographic_updates', 
                          'total_biometric_updates', 'updates_per_1000', 
                          'mobility_indicator', 'saturation_ratio']].fillna(0)
X_state_scaled = scaler.fit_transform(X_state)

# Dendrogram
plt.figure(figsize=(14, 7))
dendrogram = shc.dendrogram(shc.linkage(X_state_scaled, method='ward'), 
                            labels=state_features['state'].values,
                            leaf_font_size=10)
plt.xlabel('States', fontsize=12)
plt.ylabel('Euclidean Distance', fontsize=12)
plt.title('Hierarchical Clustering - States', fontsize=14, fontweight='bold')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('outputs/figures/hierarchical_clustering_states.png', dpi=300)
print("✅ Saved: outputs/figures/hierarchical_clustering_states.png")

# Agglomerative clustering
n_state_clusters = 4
agg_cluster = AgglomerativeClustering(n_clusters=n_state_clusters, linkage='ward')
state_features['state_cluster'] = agg_cluster.fit_predict(X_state_scaled)

state_features.to_csv('outputs/tables/state_clusters.csv', index=False)
print(f"✅ Saved: outputs/tables/state_clusters.csv")

print(f"\nState Cluster Distribution:")
for i in range(n_state_clusters):
    states = state_features[state_features['state_cluster'] == i]['state'].tolist()
    print(f"\nCluster {i}: {', '.join(states)}")

# ============================================================================
# ANOMALY VISUALIZATION
# ============================================================================
print("\n[VIZ] Creating anomaly visualizations...")

# Anomaly score distribution
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Anomaly score histogram
axes[0, 0].hist(district_features['anomaly_prob'], bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].axvline(district_features[district_features['anomaly_score'] == -1]['anomaly_prob'].max(), 
                   color='red', linestyle='--', linewidth=2, label='Anomaly Threshold')
axes[0, 0].set_xlabel('Anomaly Score', fontsize=11)
axes[0, 0].set_ylabel('Frequency', fontsize=11)
axes[0, 0].set_title('Isolation Forest Anomaly Scores', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# 2. PCA with anomalies highlighted
colors = ['red' if x == -1 else 'blue' for x in district_features['anomaly_score']]
axes[0, 1].scatter(X_pca[:, 0], X_pca[:, 1], c=colors, s=50, alpha=0.6)
axes[0, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
axes[0, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
axes[0, 1].set_title('Anomalies in PCA Space (Red=Anomaly)', fontsize=12, fontweight='bold')
axes[0, 1].grid(alpha=0.3)

# 3. Cluster vs Anomaly
cluster_anomaly = pd.crosstab(district_features['cluster_name'], 
                               district_features['anomaly_score'], 
                               normalize='index') * 100
cluster_anomaly.plot(kind='bar', ax=axes[1, 0], color=['green', 'red'])
axes[1, 0].set_xlabel('Cluster', fontsize=11)
axes[1, 0].set_ylabel('Percentage', fontsize=11)
axes[1, 0].set_title('Anomaly Rate by Cluster', fontsize=12, fontweight='bold')
axes[1, 0].legend(['Normal', 'Anomaly'])
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].grid(alpha=0.3)

# 4. Anomaly severity by feature
anomaly_districts = district_features[district_features['anomaly_score'] == -1]
normal_districts = district_features[district_features['anomaly_score'] == 1]

feature_comparison = pd.DataFrame({
    'Anomalies': anomaly_districts[cluster_cols].mean(),
    'Normal': normal_districts[cluster_cols].mean()
})
feature_comparison.plot(kind='barh', ax=axes[1, 1], color=['red', 'blue'])
axes[1, 1].set_xlabel('Mean Value', fontsize=11)
axes[1, 1].set_title('Feature Comparison: Anomalies vs Normal', fontsize=12, fontweight='bold')
axes[1, 1].legend(['Anomalies', 'Normal'])
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/figures/anomaly_analysis_comprehensive.png', dpi=300)
print("✅ Saved: outputs/figures/anomaly_analysis_comprehensive.png")

# ============================================================================
# SAVE MODELS
# ============================================================================
print("\n[SAVE] Saving models...")
joblib.dump(kmeans, 'outputs/models/kmeans_district_clustering.pkl')
joblib.dump(iso_forest, 'outputs/models/isolation_forest_enhanced.pkl')
joblib.dump(scaler, 'outputs/models/scaler_clustering.pkl')
joblib.dump(pca, 'outputs/models/pca_visualization.pkl')
print("✅ Models saved")

# ============================================================================
# INSIGHTS SUMMARY
# ============================================================================
print("\n" + "="*80)
print("CLUSTERING & ANOMALY INSIGHTS")
print("="*80)

print(f"\n📊 CLUSTERING SUMMARY:")
print(f"   ✅ K-Means: {optimal_k} clusters identified")
print(f"   ✅ Silhouette Score: {silhouette_score(X_scaled, district_features['cluster']):.4f}")
print(f"   ✅ {len(district_features)} districts segmented")

print(f"\n🚨 ANOMALY DETECTION SUMMARY:")
print(f"   ✅ Isolation Forest: {n_anomalies} anomalies ({n_anomalies/len(district_features)*100:.1f}%)")
print(f"   ✅ DBSCAN: {n_outliers} outliers ({n_outliers/len(district_features)*100:.1f}%)")
print(f"   ✅ Top anomalies saved for investigation")

print(f"\n🗺️  STATE CLUSTERING:")
print(f"   ✅ {len(state_features)} states grouped into {n_state_clusters} clusters")
print(f"   ✅ Hierarchical dendrogram generated")

print(f"\n📁 OUTPUTS:")
print(f"   - outputs/tables/district_clusters.csv")
print(f"   - outputs/tables/isolation_forest_anomalies.csv")
print(f"   - outputs/tables/state_clusters.csv")
print(f"   - outputs/figures/clustering_*.png")
print(f"   - outputs/figures/anomaly_*.png")

print("\n" + "="*80)
print("CLUSTERING & ANOMALY DETECTION COMPLETE!")
print("="*80)
