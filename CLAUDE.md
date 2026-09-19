# CAMELS SBI pipeline — project brief

Simulation-based inference on CAMELS/IllustrisTNG galaxies with small MLPs. The
goal isn't just parameter inference — it's understanding **how different
observables share information** about those parameters. Most of the machinery
here (noise cases, shuffle test, six-case taxonomy) exists to answer that
question.

**Repo**: `camelsPE/` (git). Branch: `hpc-sweep` — a stripped-down HPC
version. Notebooks, the moment network, heteroscedastic and SBI code live on
`moment-network`.
**Entry point**: `run_sweep.py` — one continuous run that, for every
observable pair, trains the models, extracts a feature vector per
(pair, parameter), and writes the consolidated summary for all 35 parameters.
`run_sweep.sh` is the SLURM wrapper. **Scope is training, features and plots
only**; dimensionality reduction and clustering are downstream and do not
belong in this repo.

**Module wiring**: `run_sweep.py` is the driver; the machinery lives in
`src/pipeline.py` (data/loader/prediction utilities) and `src/plots.py` (the
per-parameter plot functions the sweep uses). Both use a `configure(**kwargs)`
pattern: `run_sweep.configure_modules()` calls both per pair (before training,
after training, after R²). Plot functions take `param=None`-style keyword args
that fall back to the configured state, so
`plots.plot_prediction_attractor_map(param="θ4")` just works. If you change
shared state (e.g. re-run the split), re-call configure.

---

## Data

- **35 parameters** (`Parameters`, shape `(35, 2048)`, sliced to first 1024
  sims). Each has a `logflag` — 21 of them are log-transformed before z-score
  standardization. `output_dim = 35`.
- **14 observables** (from `data_L50_TNG_v3.hdf5`). Each is a per-sim vector of
  binned scaling-relation values (13–91 bins each). The loader picks up
  anything in the file whose last-axis length is `n_sims`, so it auto-adapts:
  `Ms_Mh_s{61,90}`, `MBH_Mh_s{61,90}`, `Mg_Mh_s{61,90}`, `Rs_Ms_s{61,90}`,
  `SFR_Ms_s{61,90}`, `Zs_Ms_s{61,90}`, `SFRH`, `SFRH_100Myr`.
  Bin-center arrays (`logMh_*`, `logMs_*`, `SFRH_z`) are single-row and are
  filtered out automatically.
- Observables are z-score normalized per bin into `x_normalized_dict`, then
  concatenated in `sorted(all_observables)` order to form the model input.

## Architecture

- **`SimpleMLP`** (in `src/models.py`): Linear → ReLU → Dropout stack. No
  BatchNorm/LayerNorm — this matters (see noise section).
- **Loss**: MSE in normalized-log space. R² is scored in linear space after
  `np.exp()` for the ~21 logged parameters. This asymmetry between training
  space and evaluation space is a known open item.
- **Training**: Adam (lr 1e-4, wd 1e-3), dropout 0.3, 1500 epochs, best
  weights restored by EMA-smoothed val loss (window 50). Split 80/10/10:
  val selects weights, test is used for all R² and shuffle numbers.
- The moment network (posterior-width head) is not part of this branch; see
  `moment-network`.

## The noise-mixing experiment

For each named case in `noise_cases` we pick a subset of observables and
assign each a noise level (interpreted as std). A fresh model is trained for
each case; noise is re-sampled every training epoch (`fit_with_epoch_noise`),
validation always uses clean data.

Naming convention (`run_sweep.noise_cases_for`): `"B_5.0_A_0.0"` means noise
5.0 on observable_2 (B) and 0.0 on observable_1 (A). Reference cases
`B_clean` / `A_clean` contain only one observable each. Plots swap these for
the real observable names via `display_names` / `relabel`.

**Deliberate design**: noise is added to the already-normalized observable and
**not renormalized** afterwards (`train.fit_with_epoch_noise`).
See the noise section below for the rationale — this has been re-derived
several times and should not be "fixed."

## The shuffle test — post-hoc, single trained model

For each already-trained model in a case, we ask: "which observable does the
model actually read?" Implementation lives in `resolve_shuffle`:

- **S₁ = `obs1_vs_truth`**: shuffle **observable_2** at validation; truths
  stay put. R² survives only if the model reads observable_1.
- **S₂ = `obs2_vs_truth`**: shuffle **observable_1** at validation; truths
  stay put. R² survives only if the model reads observable_2.

`run_sweep.evaluate_pair` stores these as `r2["shuf_obs2"]` (S₁) and
`r2["shuf_obs1"]` (S₂), each averaged over 10 permutations of the test set
(std in `*_std`). `plots.configure` still receives them under the legacy
names `r2_matrix_shifted_observable_only` / `r2_matrix_shifted_both`.

**observable_1 and observable_2** are the alphabetically sorted pair
(`combinations(sorted(...), 2)` in `run_sweep.py`), which matches the column
layout the model actually sees.

### `resolve_shuffle` — the canonical rule

```python
def resolve_shuffle(selected_observables, mode):
    if mode == "aligned":
        return set(), False
    sel = set(selected_observables)
    if mode == "obs1_vs_truth":
        return {observable_2} & sel, False   # shuffle obs2, keep truths
    if mode == "obs2_vs_truth":
        return {observable_1} & sel, False   # shuffle obs1, keep truths
```

`shuffle_y` is always `False`. Each mode literally shuffles one observable.
Keys are intersected with `selected_observables`, so:

- If the shuffled observable **is** in the case → real shuffle → real signal.
- If the shuffled observable **is not** in the case → empty shuffle keys →
  natural no-op → R² comes out equal to aligned R².

This handles the reference cases correctly with no special-case skips:

| shuffle | case | truth-aligned obs in case? | result | mechanism |
|---|---|---|---|---|
| S₁ (shuffle obs2) | `A_clean` | yes | **aligned** | no-op (obs2 absent) |
| S₁ (shuffle obs2) | `B_clean` | no | **collapse** | shuffle obs2, only input destroyed |
| S₂ (shuffle obs1) | `B_clean` | yes | **aligned** | no-op (obs1 absent) |
| S₂ (shuffle obs1) | `A_clean` | no | **collapse** | shuffle obs1, only input destroyed |

**The invariant**: R² collapses whenever the truth-aligned observable is
missing (the model has nothing to fall back on); R² stays aligned whenever
the truth-aligned observable is present.

## Noise: do NOT renormalize after adding it

`train.fit_with_epoch_noise` adds `N(0, noise_level)` to the
already-normalized observable, per column, and **deliberately does not
renormalize**. (This used to be `pipeline.add_noise`, called from a per-epoch
DataLoader; the noise draw now happens on the device inside the loop.) This
choice has been analyzed on a linear toy (see
`jupyter_notebook_n/toy_noise_mechanisms.py` on the `moment-network`
branch) and validated. Do not "fix" it.

The two clean-val schemes learn the identical mapping —
`w_current × sqrt(1+σ²) = w_renorm` exactly at every σ. The difference at
validation is which signal the shared learned weight is fed:

- **No renormalize (current)**: signal component in training input is `x` at
  std 1, matching clean val. Noisy channel's val contribution shrinks by the
  reliability `1/(1+σ²)` — the diminished dependence the network could learn.
- **Renormalize then clean val (rejected)**: training input divides by
  `sqrt(1+σ²)`, shrinking the signal component. Val still feeds full `x`, so
  the learned weight meets a signal `sqrt(1+σ²)`× larger than trained on. The
  renormalization *re-inflates the channel it's meant to degrade*.
- **Noise at both train and val**: valid, but answers a different question
  ("how much information does a noisier measurement carry"). Keep for
  robustness checks; not the primary experiment.

**Caveats to carry**: the argument is exact for a linear readout; a ReLU MLP
sees a shifted input scale at val, which shifts operating regime somewhat
(no norm layers in `SimpleMLP`, so linear story dominates). Also:
train loss is on noisy inputs, val loss on clean ones — the two loss curves
are not comparable and val below train is *not* an overfitting signal.

## Six-case observable-sharing taxonomy

From `CAMELS Categories.pdf`. Each θ can be classified by the combination of
(single-obs R²s, combined R², shuffle-ΔR², degradation curve shape):

1. **Constrained by one observable, no gain when combined.** Single row high,
   combined = high, other single at floor. Shuffle: scrambling the noisy
   observable has ~zero effect.
2. **Constrained by only one, but combined is better.** Signature plot: the
   param-curve rises in the both-clean middle relative to the asymmetric
   wings. The "flat" observable was doing degeneracy-breaking work invisible
   to its own single-observable score.
3. **Both single rows above threshold, combined above both.** (3a) symmetric:
   allegiance ≈ 0.5. (3b) asymmetric: allegiance pulled toward the tighter
   observable.
4. **Not constrained by either alone but constrained combined.** Both single
   rows at floor, combined above threshold. Degradation cliffs on either
   axis; shuffle-ΔR² drops on either shuffle symmetrically.
5. **Not constrained by either, not by combined.** Whole param column at
   floor.
6. **Both observables, no gain from combining.** Both single rows high,
   combined ≈ best single. **Distinguished from Case 4 by the degradation
   curve**: Case 6 degrades gracefully (redundant), Case 4 collapses.

Cases 4 and 6 have the same aligned + shuffled shuffle-plane signature at
(0, 0); only the aligned R² and the degradation-curve shape tell them apart.

## The chimera caveat (validation is off-manifold)

The shuffle test feeds the model observables from *two different simulations*
(the shuffled one comes from a different sim than the unshuffled one). No
such galaxy exists in the training distribution. This is a **saliency probe**
— "which channel does the output track?" — not posterior inference.

For Case 4 (crossed ridges) specifically, the model can extrapolate
confidently to a parameter combination *neither* simulation has, so the
combined-model prediction can land far off the truth-truth line. A 1D
allegiance score cannot detect this; multi-parameter or off-line-distance
diagnostics can.

## File map

```
GAL_SBI/
├── DATA/
│   └── data_L50_TNG_v3.hdf5    ← training data (default --data path)
└── camelsPE/                    ← git repo (remote is YongseokJo/camelsPE)
    ├── CLAUDE.md
    ├── run_sweep.py        ← THE script: train → features → plots, all pairs
    ├── run_sweep.sh        SLURM wrapper (resubmit to resume)
    ├── src/
    │   ├── models.py       SimpleMLP
    │   ├── train.py        fit_with_epoch_noise — device-resident loop
    │   ├── pipeline.py     normalize, eval loaders, resolve_shuffle,
    │   │                   get_case_predictions, average_r2_over_perms
    │   ├── features.py     per-(pair, parameter) feature vectors + column dictionary
    │   └── plots.py        the consolidated-summary panels
    └── sweep_output/       (gitignored) see "Outputs" below
```

## Running

- Everything: `python run_sweep.py` (or `sbatch run_sweep.sh`). 14
  observables → 91 pairs × 7 cases × 1500 epochs, then features and 35
  summary figures per pair.
- Resumable: pairs with `sweep_output/models/<pair>.pt` are loaded instead of
  retrained (features and plots are regenerated); `--overwrite` retrains.
- Smoke test: `python run_sweep.py --epochs 5 --pairs 0 --out /tmp/smoke`.
- Rough cost: features + plots are ~40 s per pair (~1 h total) and ~18 MB
  per pair (~1.6 GB total); training dominates wall-clock.

## Outputs — set up for downstream analysis

```
sweep_output/
├── models/<pair>.pt        weights per noise case, losses, R² matrices, hparams
├── tables/<pair>/*.csv     aligned + shuffled R², long-format dual table
├── features/<pair>.csv     35 rows (one per parameter)
├── plots/<pair>/           R² heatmaps, loss curves, θ<j>_summary.png × 35
├── plots/summary/          cross-pair heatmaps
├── features_all.csv        all 91 × 35 rows + `figure` path to each summary
├── feature_columns.csv     one-line meaning of every column
└── run_config.json         hyperparameters, data file, git commit, failures, timing
```

`features_all.csv` is the entry point for downstream dimensionality
reduction / clustering: one row per (pair, parameter), identity columns
(`pair`, `obs1`, `obs2`, `param`, `figure`) followed by ~41 numeric features.
The `figure` column maps any downstream result (a cluster, an outlier) back to
the plot that shows it.

The features are the numeric content of each consolidated summary, so
downstream work can run on numbers rather than images (a model over rendered
PNGs would mostly encode axis limits and point density):

| Panel | Features |
|---|---|
| dual R² curve | per-case aligned R², both single-observable R², gain from combining, degradation along each noise arm, curvature (cliff vs graceful) |
| attractor map | per-case slope of predicted vs true (shrinkage toward the prior) |
| shuffle scatters | where chimera predictions land between kept and swapped truth: median, IQR, fraction tracking each side, allegiance |
| pair-truth scatters | off-line distance from the anchor truth — the chimera extrapolation a 1D allegiance score misses |

Case-specific columns are named after the case (`r2_B_2.5_A_0.0`,
`slope_A_clean`), and the case keys are identical for every pair, so columns
line up across all rows.

## Code style

- **Simple, readable, flat**. Plain functions in `run_sweep.py`, no classes
  or config frameworks. Match the idioms already in the repo.
- **Minimize tests**. Verify by running the analysis and inspecting outputs
  (losses, R² values, plots, tensor shapes). Not by writing unit-test suites.
- **Comments**: light and physical (what/why), matching existing density.

## Things Claude has previously "fixed" that should stay as-is

Recording these to prevent future sessions from re-flagging them:

1. **Noise isn't renormalized after being added.** Deliberate; see the
   noise section above.
2. **Train loss > val loss** on plots. Not overfitting — train is on noisy
   input, val on clean. Consequence of design (1).
3. **Noise is drawn fresh every epoch**, not once up front, and validation
   and test inputs stay clean.
4. **`observable_1`/`observable_2` are the alphabetically sorted pair**, which
   matches the column order the model is fed.
5. **ΔR² heatmaps compute `shifted − original`** (negative = information
   lost). This is the user's chosen convention; the labels match.
6. **Single-observable reference cases produce collapse in one shuffle
   direction, aligned in the other.** By design — see the shuffle table.

## Open items (not yet addressed)

- **R² space mismatch**: model trains on MSE in normalized-log space; R² is
  computed in linear space after `exp()`. For logged parameters, R² is
  dominated by the tail.
- **Feature set is a first pass**: the vectors cover the four panel types
  but nothing about per-parameter posterior width (that would need the
  moment network from `moment-network`). Some columns are strongly
  correlated (e.g. per-case R² and slopes); expect to standardize and
  reduce before clustering downstream.
- **Short-name collisions in scatter plots**: `plots.py`'s `short_of` uses
  the first token of the observable name, so same-quantity pairs (e.g.
  `MBH_Mh_s61 × MBH_Mh_s90`) are both labeled "MBH" in the shuffle and
  pair-truth scatter panels.
