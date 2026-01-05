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

# PCA with cluster centroids
try:
    centers_df = merged.groupby('Cluster')[features].mean().sort_index()
    centers_scaled = scaler.transform(centers_df.values)
    centers_pca = pca.transform(centers_scaled)
    plt.figure(figsize=(6, 5))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='tab10', alpha=0.7, edgecolor='k')
    for i, (x, y) in enumerate(centers_pca):
        plt.scatter(x, y, s=200, marker='X', c=[sns.color_palette('tab10')[i % 10]], edgecolor='k')
        plt.text(x + 0.02, y + 0.02, f'Center {i}', fontsize=9)
    plt.title('PCA: Clusters with Centroids')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    cent_out = os.path.join(OUT_DIR, 'pca_with_centroids.png')
    plt.tight_layout()
    plt.savefig(cent_out, dpi=200)
    plt.close()
    print(f"Saved: {cent_out}")
except Exception as e:
    print(f"Could not compute centroids plot: {e}")

# Pairplot colored by cluster (smaller size for readability)
try:
    gp = sns.pairplot(merged, vars=features, hue='Cluster', diag_kind='hist', corner=False, plot_kws={'alpha':0.6, 's':20})
    pair_out = os.path.join(OUT_DIR, 'pairplot_by_cluster.png')
    gp.savefig(pair_out)
    plt.close()
    print(f"Saved: {pair_out}")
except Exception as e:
    print(f"Pairplot failed: {e}")

# Histograms of features colored by cluster
plt.figure(figsize=(12, 10))
for i, feat in enumerate(features, 1):
    plt.subplot(3, 2, i)
    for c in sorted(merged['Cluster'].unique()):
        sns.histplot(merged[merged['Cluster'] == c][feat], stat='density', kde=False, label=f'Cluster {int(c)}', alpha=0.5)
    plt.title(f'Histogram: {feat} by Cluster')
    plt.legend()
    plt.tight_layout()
hist_out = os.path.join(OUT_DIR, 'features_histograms.png')
plt.savefig(hist_out, dpi=200)
plt.close()
print(f"Saved: {hist_out}")

# Silhouette plot
from sklearn.metrics import silhouette_samples, silhouette_score
n_clusters = len(np.unique(clusters))
if n_clusters >= 2:
    sil_vals = silhouette_samples(X_scaled, clusters)
    sil_avg = silhouette_score(X_scaled, clusters)
    plt.figure(figsize=(8, 6))
    y_lower = 10
    for i in range(n_clusters):
        ith_sil_vals = sil_vals[clusters == i]
        ith_sil_vals.sort()
        size_i = ith_sil_vals.shape[0]
        y_upper = y_lower + size_i
        color = sns.color_palette('tab10')[i % 10]
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_sil_vals, facecolor=color, edgecolor=color, alpha=0.7)
        plt.text(-0.05, y_lower + 0.5 * size_i, f'Cluster {i}')
        y_lower = y_upper + 10
    plt.axvline(sil_avg, color='red', linestyle='--')
    plt.xlabel('Silhouette coefficient')
    plt.ylabel('Cluster')
    plt.title(f'Silhouette plot (avg = {sil_avg:.3f})')
    sil_out = os.path.join(OUT_DIR, 'silhouette_plot.png')
    plt.tight_layout()
    plt.savefig(sil_out, dpi=200)
    plt.close()
    print(f"Saved: {sil_out}")
else:
    print('Not enough clusters to compute silhouette scores (need >=2).')

# Boxplots per cluster for each vital (existing)
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
