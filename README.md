# camelsPE — HPC observable-pair sweep

Simulation-based inference on CAMELS/IllustrisTNG galaxies. For every pair of
observables, this trains small MLPs under a set of noise cases, runs the
shuffle test to see how the two observables share information about the 35
parameters, and saves models, per-parameter plots and feature vectors in one
continuous run.

```bash
sbatch run_sweep.sh                        # on SLURM (resubmit to resume)
python run_sweep.py                        # same, directly
python run_sweep.py --epochs 5 --pairs 0   # smoke test
python run_sweep.py --help
```

Everything lands in `sweep_output/`:

| Path | Contents |
|---|---|
| `models/<pair>.pt` | trained weights per noise case, losses, R² matrices |
| `tables/<pair>/` | aligned + shuffled R² CSVs |
| `features/<pair>.csv` | one feature vector per parameter |
| `plots/<pair>/` | R² heatmaps, loss curves, a consolidated summary per parameter |
| `plots/summary/` | cross-pair heatmaps |
| `features_all.csv` | every (pair, parameter) row + the path to its figure |
| `feature_columns.csv` | what each feature column means |
| `run_config.json` | hyperparameters, data file, git commit, timing |

`features_all.csv` is the starting point for downstream analysis
(dimensionality reduction, clustering), which is not part of this repo.

Data defaults to `../DATA/data_L50_TNG_v3.hdf5` (override with `--data`).

Requires: torch, numpy, pandas, h5py, scikit-learn, matplotlib, seaborn.

See `CLAUDE.md` for the experiment design and the feature definitions.
