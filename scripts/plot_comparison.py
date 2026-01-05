"""Generate comparison plots between original data and clustered results.

Saves plots to `results/plots/`:
 - pca_original_vs_cluster.png
 - boxplots_by_cluster.png
 - cluster_centers_heatmap.png

Usage:
    python3 scripts/plot_comparison.py
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CLEANED_CSV = os.path.join(BASE_DIR, 'dataset', 'cleaned', 'aggregated_vitals_cleaned.csv')
PATIENT_CLUSTERS = os.path.join(BASE_DIR, 'results', 'patient_clusters.csv')
CLUSTER_CENTERS = os.path.join(BASE_DIR, 'results', 'cluster_centers.csv')
OUT_DIR = os.path.join(BASE_DIR, 'results', 'plots')

os.makedirs(OUT_DIR, exist_ok=True)

# Load data
if not os.path.exists(CLEANED_CSV):
    print(f"Missing file: {CLEANED_CSV}")
    sys.exit(1)
if not os.path.exists(PATIENT_CLUSTERS):
    print(f"Missing file: {PATIENT_CLUSTERS}")
    sys.exit(1)

print("Loading data...")
df = pd.read_csv(CLEANED_CSV)
df_clusters = pd.read_csv(PATIENT_CLUSTERS)

# Merge on Subject_ID if present
if 'Subject_ID' in df.columns and 'Subject_ID' in df_clusters.columns:
    merged = pd.merge(df, df_clusters[['Subject_ID', 'Cluster']], on='Subject_ID', how='inner')
else:
    # Fall back: use df_clusters as the source data
    merged = df_clusters.copy()

features = ['Systolic_BP', 'Diastolic_BP', 'Heart_Rate', 'Oxygen_Saturation', 'Temperature']
X = merged[features].values

# Standardize for PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# Plot: PCA original (gray) vs clustered (colored)
plt.figure(figsize=(12, 5))
ax1 = plt.subplot(1, 2, 1)
ax1.scatter(X_pca[:, 0], X_pca[:, 1], c='lightgray', edgecolor='k', alpha=0.8)
ax1.set_title('PCA: Original (no clusters)')
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')

ax2 = plt.subplot(1, 2, 2)
palette = sns.color_palette('tab10')
clusters = merged['Cluster'].astype(int)
scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='tab10', edgecolor='k', alpha=0.9)
legend1 = ax2.legend(*scatter.legend_elements(), title='Cluster', loc='best')
ax2.add_artist(legend1)
ax2.set_title('PCA: Colored by Cluster')
ax2.set_xlabel('PC1')
ax2.set_ylabel('PC2')

plt.tight_layout()
pca_out = os.path.join(OUT_DIR, 'pca_original_vs_cluster.png')
plt.savefig(pca_out, dpi=200)
plt.close()
print(f"Saved: {pca_out}")

# Boxplots per cluster for each vital
plt.figure(figsize=(12, 9))
for i, feat in enumerate(features, 1):
    plt.subplot(3, 2, i)
    sns.boxplot(x='Cluster', y=feat, data=merged, palette='Set2')
    plt.title(f'Boxplot: {feat} by Cluster')
    plt.tight_layout()
box_out = os.path.join(OUT_DIR, 'boxplots_by_cluster.png')
plt.savefig(box_out, dpi=200)
plt.close()
print(f"Saved: {box_out}")

# Cluster centers heatmap if available
if os.path.exists(CLUSTER_CENTERS):
    cc = pd.read_csv(CLUSTER_CENTERS, index_col=0)
    plt.figure(figsize=(8, 4))
    sns.heatmap(cc, annot=True, fmt='.2f', cmap='Spectral')
    plt.title('Cluster Centers (original scale)')
    heat_out = os.path.join(OUT_DIR, 'cluster_centers_heatmap.png')
    plt.tight_layout()
    plt.savefig(heat_out, dpi=200)
    plt.close()
    print(f"Saved: {heat_out}")
else:
    print(f"Cluster centers file not found at: {CLUSTER_CENTERS}")

print("All plots generated.")
