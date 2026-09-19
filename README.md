# camelsPE — HPC observable-pair sweep

Simulation-based inference on CAMELS/IllustrisTNG galaxies: for every pair of
observables, train small MLPs under a set of noise cases, run the shuffle test
to see how the two observables share information about the 35 parameters, and
cluster the result so parameters can be categorized algorithmically instead of
by eye.

```bash
python run_sweep.py                 # all 91 pairs: train, features, plots, cluster
sbatch run_sweep.sh                 # same, on SLURM (resubmit to resume)
sbatch --array=0-7 run_sweep.sh     # 8 parallel shards
python run_sweep.py --stages cluster   # re-cluster from saved features
python run_sweep.py --help
```

Everything lands in `sweep_output/`:

| Directory | Contents |
|---|---|
| `models/<pair>.pt` | trained weights per noise case, losses, R² matrices |
| `tables/<pair>/` | aligned + shuffled R² CSVs |
| `features/<pair>.csv` | one feature vector per parameter — the clustering input |
| `plots/<pair>/` | R² heatmaps, loss curves, a consolidated summary per parameter |
| `plots/summary/` | cross-pair heatmaps |
| `clusters/` | embedding, cluster labels, cluster profiles, medoid figures |

Data defaults to `../DATA/data_L50_TNG_v3.hdf5` (override with `--data`).

Requires: torch, numpy, pandas, h5py, scikit-learn, matplotlib, seaborn.
Optional: umap-learn and hdbscan (the clustering stage falls back to t-SNE and
KMeans without them).

See `CLAUDE.md` for the experiment design and the clustering rationale.
