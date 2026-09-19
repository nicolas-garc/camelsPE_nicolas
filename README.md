# camelsPE — observable-pair sweep

Simulation-based inference on CAMELS/IllustrisTNG galaxies. For every pair of
observables (14 observables → 91 pairs), this trains small MLPs under a set of
noise cases, runs a shuffle test to measure how the two observables share
information about the 35 parameters, and saves the models, a feature vector per
(pair, parameter), and the plots — in one continuous run.

The training data (3.4 MB) is included in `data/`, so a clone is all you need.

---

## Running it on HPC

**1. Get the code onto the cluster**

```bash
git clone --depth 1 --single-branch -b hpc-sweep <repo-url> camelsPE && cd camelsPE
```

`--depth 1 --single-branch` matters: the repo's history contains old notebooks
with embedded outputs (~1 GB), while this branch's working tree is only 3.5 MB.
If you were sent a `.tar.gz` instead, just unpack it — no git needed.

**2. Set up an environment** (Python 3.11 recommended; see `requirements.txt`)

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

A GPU is optional. These are small MLPs — the script uses CUDA when available
and falls back to CPU automatically.

**3. Smoke test first** (about a minute; confirms the data loads and plotting
works headless)

```bash
python run_sweep.py --epochs 5 --pairs 0 --out /tmp/smoke_test
```

It should end with `Failed pairs: none` and leave models, a feature CSV and 41
PNGs under `/tmp/smoke_test`.

**4. Edit `run_sweep.sh`** — three marked sections: partition/GPU, how to
activate the environment, and optionally the data path. The rest is generic.

**5. Submit**

```bash
sbatch run_sweep.sh
```

Expect roughly **1–2 days** of wall-clock for all 91 pairs and about **2 GB** of
output. Progress is logged per pair to `sweep_%j.out`.

**If the job hits its time limit, resubmit the same command.** Pairs whose
models are already saved are loaded instead of retrained, so the run continues
where it stopped and writes into the same output directory.

---

## What it produces

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
| `run_config.json` | hyperparameters, data file, git commit, failures, timing |

To send results back, the whole directory tars to ~2 GB; if that is awkward,
`features_all.csv`, `feature_columns.csv`, `run_config.json` and `tables/` are
only a few MB and carry all the numbers — plots can be regenerated from
`models/` later.

`features_all.csv` is the starting point for the downstream analysis
(dimensionality reduction, clustering), which is deliberately **not** part of
this repo.

---

## Useful flags

```bash
python run_sweep.py --help
python run_sweep.py --pairs 0 1 2      # only these pair indices (printed at startup)
python run_sweep.py --epochs 500       # shorter training
python run_sweep.py --overwrite        # retrain pairs that already have models
python run_sweep.py --data /path/to/data_L50_TNG_v3.hdf5
python run_sweep.py --out /scratch/$USER/sweep_output
```

## Troubleshooting

- **`Training data not found`** — pass `--data /path/to/data_L50_TNG_v3.hdf5`;
  the error lists the paths it searched.
- **A pair fails** — the run logs the traceback, records it in
  `run_config.json` under `failed_pairs`, and continues with the next pair.
- **matplotlib complains about a display** — it shouldn't; the script forces the
  headless `Agg` backend, so no `$DISPLAY` is needed.
- **Slow on CPU** — reduce `--epochs`, or split pairs across jobs with `--pairs`
  and merge the output directories afterwards.

See `CLAUDE.md` for the experiment design, the noise/shuffle-test rationale, and
the definition of every feature column.
