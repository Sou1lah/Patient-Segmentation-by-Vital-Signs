# Patient-Segmentation-by-Vital-Signs

## Overview ✅
This project performs patient segmentation based on vital signs (blood pressure, heart rate, oxygen saturation, temperature). It includes data cleaning, scaling, exploratory analysis, K‑Means clustering, and visualization of cluster results.

---

## Quick Start 🔧
1. Ensure required packages are installed: pandas, numpy, matplotlib, seaborn, scikit-learn

2. Run the clustering notebook to produce results (if not already):

   - `notebooks/01_data_cleaning.ipynb` — data cleaning & feature scaling
   - `notebooks/02_training_clustering.ipynb` — determine K, fit K-Means and save `results/patient_clusters.csv`, `results/cluster_centers.csv`

3. Generate comparison plots (original vs clustered):

```bash
python3 scripts/plot_comparison.py
```

---

## Saved Plots 📊
After running the script, plots will be saved under `results/plots/` and the README will reference them here:

- PCA comparison (original vs clustered): `results/plots/pca_original_vs_cluster.png`
- Boxplots per cluster: `results/plots/boxplots_by_cluster.png`
- Cluster centers heatmap: `results/plots/cluster_centers_heatmap.png`

(If the images are present, they will show below.)

---

## Files and Results ✨
- `dataset/cleaned/aggregated_vitals_cleaned.csv` — cleaned aggregated vitals
- `dataset/cleaned/scaled_features.npy` — scaled features used for clustering
- `results/patient_clusters.csv` — patient vitals with assigned `Cluster`
- `results/cluster_centers.csv` — cluster centers (original scale)
- `notebooks/` — exploratory analysis and clustering notebooks
- `scripts/plot_comparison.py` — generates visualization comparisons

---

## Notes
- Hierarchical clustering is optional and not implemented in the main pipeline (see `todo.md`).
- If you want, I can embed generated plot images into this README after you run the plotting script and confirm the output images. For now the README links the expected files.
