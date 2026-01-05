"""Cluster analysis utilities

- Compute hierarchical clustering and save a dendrogram.
- Fit AgglomerativeClustering to compare with existing KMeans assignment.
- Perform simple cluster stability checks via bootstrap (adjusted Rand index).

Usage: run as script from repo root. Outputs saved to results/ and results/plots/.
"""
from __future__ import annotations
import os
from typing import Callable, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'dataset', 'cleaned')
OUT_DIR = os.path.join(BASE_DIR, 'results')
PLOTS_DIR = os.path.join(OUT_DIR, 'plots')

os.makedirs(PLOTS_DIR, exist_ok=True)


def load_scaled_features(path: Optional[str] = None) -> np.ndarray:
    """Load scaled features (numpy array). Ensure float dtype and 2D shape."""
    path = path or os.path.join(DATA_DIR, 'scaled_features.npy')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scaled features not found at: {path}")
    X = np.load(path)
    if X.ndim != 2:
        raise ValueError("Expected 2D array for features (n_samples, n_features)")
    return X.astype(float)


def read_existing_clusters() -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray]]:
    """Read existing cluster assignments if present (KMeans results)."""
    clusters_path = os.path.join(BASE_DIR, 'results', 'patient_clusters.csv')
    if not os.path.exists(clusters_path):
        return None, None
    df = pd.read_csv(clusters_path)
    if 'Cluster' not in df.columns:
        return df, None
    labels = df['Cluster'].astype(int).to_numpy()
    return df, labels


def plot_dendrogram(X: np.ndarray, method: str = 'ward', metric: str = 'euclidean', out: Optional[str] = None) -> str:
    """Compute linkage and save a dendrogram plot. Returns path to saved plot."""
    import matplotlib.pyplot as plt

    if not isinstance(X, np.ndarray):
        X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError('X must be a 2D array')

    Z = linkage(X, method=method, metric=metric)

    plt.figure(figsize=(10, 6))
    dendrogram(Z, no_labels=True, color_threshold=None)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Samples')
    plt.ylabel('Distance')

    out = out or os.path.join(PLOTS_DIR, 'dendrogram.png')
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def fit_agglomerative(X: np.ndarray, n_clusters: int, linkage_method: str = 'ward') -> np.ndarray:
    """Fit AgglomerativeClustering and return integer labels."""
    if not isinstance(n_clusters, int) or n_clusters < 2:
        raise ValueError('n_clusters must be an integer >= 2')
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    labels = model.fit_predict(X)
    return labels.astype(int)


def bootstrap_stability(labels_ref: np.ndarray, X: np.ndarray, cluster_fn: Callable[[np.ndarray], np.ndarray], n_runs: int = 50, sample_frac: float = 0.8) -> Tuple[float, float]:
    """Estimate cluster stability by bootstrapping samples and comparing labels with adjusted Rand index.

    For each run, sample a subset of indices, fit clustering on the subset (via cluster_fn) and compare
    the resulting labels restricted to the subset with the reference labels at those indices.
    Returns mean and std of ARI across runs.
    """
    if labels_ref is None:
        raise ValueError('labels_ref must be provided to compare stability')
    n_samples = X.shape[0]
    if n_samples != labels_ref.shape[0]:
        raise ValueError('labels_ref length must match number of samples in X')

    rng = np.random.default_rng(42)
    aris = []
    for i in range(n_runs):
        # sample without replacement
        idx = rng.choice(n_samples, size=int(sample_frac * n_samples), replace=False)
        X_sub = X[idx]
        try:
            labels_sub = cluster_fn(X_sub)
        except Exception as e:
            # If clustering fails on subset, skip this run but warn
            print(f"Warning: clustering failed on bootstrap run {i}: {e}")
            continue
        # compare labels on the subset
        ari = adjusted_rand_score(labels_ref[idx], labels_sub)
        aris.append(ari)
    if len(aris) == 0:
        return 0.0, 0.0
    return float(np.mean(aris)), float(np.std(aris))


def main():
    X = load_scaled_features()
    df_clusters, labels_k = read_existing_clusters()
    if labels_k is None:
        # fallback: try to fit KMeans with a reasonable K
        print('No existing KMeans cluster labels found; fitting KMeans with K=3 as fallback')
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels_k = kmeans.fit_predict(X).astype(int)

    n_clusters = int(np.unique(labels_k).shape[0])

    print(f'Found {X.shape[0]} samples and {n_clusters} clusters (from KMeans)')

    # Dendrogram
    dend_out = plot_dendrogram(X)
    print(f'Dendrogram saved to: {dend_out}')

    # Agglomerative clustering
    labels_agg = fit_agglomerative(X, n_clusters=n_clusters)

    # Compare labels
    ari_k_vs_agg = adjusted_rand_score(labels_k, labels_agg)
    sil_k = silhouette_score(X, labels_k) if n_clusters >= 2 else float('nan')
    sil_agg = silhouette_score(X, labels_agg) if n_clusters >= 2 else float('nan')

    # Stability checks (bootstrap ARI)
    def kmeans_fn(X_sub: np.ndarray) -> np.ndarray:
        m = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        return m.fit_predict(X_sub).astype(int)

    def agg_fn(X_sub: np.ndarray) -> np.ndarray:
        m = AgglomerativeClustering(n_clusters=n_clusters)
        return m.fit_predict(X_sub).astype(int)

    k_mean_ari_mean, k_mean_ari_std = bootstrap_stability(labels_k, X, kmeans_fn, n_runs=40)
    agg_ari_mean, agg_ari_std = bootstrap_stability(labels_agg, X, agg_fn, n_runs=40)

    # Save summary to CSV
    out_csv = os.path.join(OUT_DIR, 'cluster_stability.csv')
    summary = pd.DataFrame([
        {
            'method': 'kmeans_existing',
            'n_clusters': n_clusters,
            'silhouette': float(sil_k),
            'stability_ari_mean': k_mean_ari_mean,
            'stability_ari_std': k_mean_ari_std,
        },
        {
            'method': 'agglomerative',
            'n_clusters': n_clusters,
            'silhouette': float(sil_agg),
            'stability_ari_mean': agg_ari_mean,
            'stability_ari_std': agg_ari_std,
        },
        {
            'method': 'kmeans_vs_agglomerative_ARI',
            'n_clusters': n_clusters,
            'silhouette': float('nan'),
            'stability_ari_mean': float(ari_k_vs_agg),
            'stability_ari_std': float('nan'),
        }
    ])
    summary.to_csv(out_csv, index=False)
    print(f'Cluster stability summary saved to: {out_csv}')

    # Append to segmentation report
    rep_path = os.path.join(OUT_DIR, 'segmentation_report.txt')
    lines = [
        '\n### Hierarchical & Stability Analysis\n',
        f'Number of clusters (reference KMeans): {n_clusters}\n',
        f'Adjusted Rand Index (KMeans vs Agglomerative): {ari_k_vs_agg:.3f}\n',
        f'KMeans silhouette: {sil_k:.3f}, bootstrap ARI mean: {k_mean_ari_mean:.3f} ± {k_mean_ari_std:.3f}\n',
        f'Agglomerative silhouette: {sil_agg:.3f}, bootstrap ARI mean: {agg_ari_mean:.3f} ± {agg_ari_std:.3f}\n',
    ]
    with open(rep_path, 'a') as fh:
        fh.writelines(lines)
    print(f'Appended hierarchical/stability summary to: {rep_path}')


if __name__ == '__main__':
    main()
