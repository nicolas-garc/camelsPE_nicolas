# CAMELS SBI pipeline — project brief

Simulation-based inference on CAMELS/IllustrisTNG galaxies with small MLPs. The
goal isn't just parameter inference — it's understanding **how different
observables share information** about those parameters. Most of the machinery
here (noise cases, shuffle test, six-case taxonomy) exists to answer that
question.

**Repo**: `camelsPE/` (git). Working branch: `notebook-shuffle-fixes`.
**Main notebook**: `camelsPE/jupyter_notebook_n/test-noise-Copy1.ipynb` (paired
with `.py` via jupytext — always edit the `.py`).

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
- **Moment network** (two-stage): first MLP predicts parameter means, then a
  second MLP is trained on `residuals² → log → standardized` targets to
  predict per-parameter variance. This gives you a direct readout of learned
  posterior width — worth leaning on when discussing "how much the model
  narrows the prior."

## The noise-mixing experiment

For each named case in `noise_cases` we pick a subset of observables and
assign each a noise level (interpreted as std). A fresh model is trained for
each case; noise is re-sampled every training epoch (`fit_with_epoch_noise`),
validation always uses clean data.

Naming convention in `noise_cases`: e.g. `"sfr_5.0_ms_0.0"` means SFR-side
noise 5.0, Ms-side noise 0.0. Reference cases `sfr_clean` and `ms_clean`
contain only one observable each (both at noise 0).

**Deliberate design**: noise is added to the already-normalized observable and
**not renormalized** afterwards (`add_noise`, not "add_noise_and_normalize").
See the noise section below for the rationale — this has been re-derived
several times and should not be "fixed."

## The shuffle test — post-hoc, single trained model

For each already-trained model in a case, we ask: "which observable does the
model actually read?" Implementation lives in `resolve_shuffle`:

- **S₁ = `obs1_vs_truth`**: shuffle **observable_2** at validation; truths
  stay put. R² survives only if the model reads observable_1.
- **S₂ = `obs2_vs_truth`**: shuffle **observable_1** at validation; truths
  stay put. R² survives only if the model reads observable_2.

The two loops populate `r2_matrix_shifted_observable_only` (S₁) and
`r2_matrix_shifted_both` (S₂). The naming of the second matrix is legacy —
it no longer literally shuffles "both." It's the dual of S₁.

**observable_1 and observable_2** are derived once from
`sorted(all_observables)[:2]` (deterministic across kernel restarts;
matches the column layout the model actually sees). Editing `noise_cases` to
use different observables auto-propagates. If you want a specific pair
comparison that isn't alphabetically first/second, override the two names by
hand.

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
| S₁ (shuffle obs2) | `ms_clean` | yes | **aligned** | no-op (obs2 absent) |
| S₁ (shuffle obs2) | `sfr_clean` | no | **collapse** | shuffle obs2, only input destroyed |
| S₂ (shuffle obs1) | `sfr_clean` | yes | **aligned** | no-op (obs1 absent) |
| S₂ (shuffle obs1) | `ms_clean` | no | **collapse** | shuffle obs1, only input destroyed |

**The invariant**: R² collapses whenever the truth-aligned observable is
missing (the model has nothing to fall back on); R² stays aligned whenever
the truth-aligned observable is present.

## Noise: do NOT renormalize after adding it

`add_noise(array_np, noise_level)` adds `N(0, noise_level)` to an
already-normalized observable and **deliberately does not renormalize**. This
choice has been analyzed on a linear toy (see
`jupyter_notebook_n/toy_noise_mechanisms.py`) and validated. Do not "fix" it.

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
│   └── data_L50_TNG_v3.hdf5    ← primary training data
├── camelsPE/                    ← git repo (remote is YongseokJo/camelsPE)
│   ├── CLAUDE.md           ← this file
│   ├── src/
│   │   ├── models.py       SimpleMLP (+ WIP VarMLP)
│   │   ├── train.py        fit, fit_with_epoch_noise, train_one_epoch
│   │   └── losses.py       MSELoss
│   ├── jupyter_notebook_n/
│   │   ├── test-noise-Copy1.{py,ipynb}    ← MAIN NOTEBOOK
│   │   ├── toy_noise_mechanisms.{py,ipynb} ← linear-Gaussian sandbox
│   │   ├── Noise_Injection, MomentNetwork, analysis, …
│   │   └── (unpaired scratch: Untitled*, *-Copy1, older test-*)
│   ├── main.py             script entry point
│   └── run.sh              SLURM job (HPC)
├── noise_results/, results/, grid_results/    ← output plots + R² matrices
└── (poster PNGs, older versions in original_code/)
```

## Code style

- **Simple, readable, notebook-flat**. Favor top-to-bottom cells over classes
  or config frameworks. Match the idioms already in the repo.
- **Minimize tests**. Verify by running the analysis and inspecting outputs
  (losses, R² values, plots, tensor shapes). Not by writing unit-test suites.
- **Comments**: light and physical (what/why), matching existing density.
- **Jupytext workflow**: edit the `.py`, then
  `python3 -m jupytext --sync <name>.ipynb` to update the paired notebook.
  Don't hand-edit `.ipynb` JSON.

## Things Claude has previously "fixed" that should stay as-is

Recording these to prevent future sessions from re-flagging them:

1. **`add_noise` doesn't renormalize.** Deliberate; see noise section above.
2. **Train loss > val loss** on plots. Not overfitting — train is on noisy
   input, val on clean. Consequence of design (1).
3. **`add_noise` uses `array_np = array_np + noise`, not `+=`**. Prevents
   accidental corruption of `x_normalized_dict` if called on an unindexed
   array. Numerically identical at all current call sites.
4. **`observable_1`/`observable_2` from `sorted(all_observables)[:2]`, not
   `list(...)[0/1]`**. The latter was hash-randomized between kernel restarts.
5. **ΔR² heatmaps compute `shifted − original`** (negative = information
   lost). This is the user's chosen convention; the labels match.
6. **Single-observable reference cases produce collapse in one shuffle
   direction, aligned in the other.** By design — see the shuffle table.

## Open items (not yet addressed)

- **Permutation averaging**: shuffle R² currently uses a single perm draw.
  At val size ~102, per-cell jitter is meaningful. Averaging over K=50 perms
  (no retraining needed) would give mean + std per cell.
- **R² space mismatch**: model trains on MSE in normalized-log space; R² is
  computed in linear space after `exp()`. For logged parameters, R² is
  dominated by the tail.
- **Loss-curve interpretation**: label the plots as "train (noisy input) vs
  val (clean input)" to prevent the val-below-train reading as no-overfit.
- **Consolidate `noise_cases` documentation**: names like `sfr_5.0_ms_0.0`
  encode which observable each noise level applies to; make this explicit
  somewhere.
