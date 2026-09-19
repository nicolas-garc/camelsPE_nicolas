"""Per-(pair, parameter) feature vectors — the numeric content of the plots.

One row per (observable pair, parameter); the columns are what a human reads
off `consolidated_param_summary`:

  R² curve      how well each single observable constrains θ, what combining
                adds, and whether degradation under noise is a cliff or graceful
  attractor     per-case slope of predicted vs true — shrinkage toward the prior
  shuffle grid  where chimera predictions land between the kept sim's truth and
                the swapped sim's truth — i.e. which observable the output tracks
  off-line      how far chimera predictions sit from the anchor truth, the
                extrapolation a 1D allegiance score cannot see (CLAUDE.md)

These are the axes of the six-case taxonomy, so clustering in this space is the
algorithmic version of sorting the plots by eye. Everything is scaled per
parameter (R² is already unitless; distances are in units of the parameter's
test-set spread) so rows are comparable across parameters and pairs.

Reads the same cached model outputs the plots use, so calling this alongside
plotting costs almost nothing extra.
"""
import numpy as np
import pandas as pd

import pipeline
import plots as plots_mod

# Case keys as built by run_sweep.noise_cases_for — same for every pair
BOTH_CLEAN = "B_0.0_A_0.0"
A_ALONE, B_ALONE = "A_clean", "B_clean"
NOISE_ARM_B = ["B_0.0_A_0.0", "B_2.5_A_0.0", "B_5.0_A_0.0"]   # increasing noise on obs2
NOISE_ARM_A = ["B_0.0_A_0.0", "B_0.0_A_2.5", "B_0.0_A_5.0"]   # increasing noise on obs1
NEAR = 0.25          # |u| < NEAR counts as "tracks that sim's truth"
MIN_TRUTH_GAP = 0.1  # skip chimera pairs whose truths barely differ (unstable ratio)


def _safe_ratio(a, b, default=0.5):
    """a / (a + b) with a guard for the degenerate both-zero case."""
    tot = a + b
    return np.where(np.abs(tot) < 1e-12, default, a / np.where(np.abs(tot) < 1e-12, 1.0, tot))


def _slopes(pred, true):
    """Per-parameter slope of predicted vs true: 1 = no shrinkage, 0 = prior mean."""
    t = true - true.mean(axis=0)
    p = pred - pred.mean(axis=0)
    var = (t * t).sum(axis=0)
    return np.where(var > 0, (t * p).sum(axis=0) / np.where(var > 0, var, 1.0), 0.0)


def _grid_stats(grid, truth, n_params):
    """Chimera-grid statistics per parameter.

    grid[j, k, :] is the prediction when the kept observable comes from sim j
    and the shuffled one from sim k. Normalizing by the two truths,

        u = (pred - θ_j) / (θ_k - θ_j)

    puts 0 at "the output followed the observable it kept" and 1 at "it
    followed the one swapped in". Returns medians, the fraction near each end,
    spread, and the off-line distance from the anchor truth (in units of the
    parameter's test-set std).
    """
    # Each unordered pair is used once: u for (j, k) is exactly 1 - u for
    # (k, j), so including both orders would pin every median at 0.5 and make
    # "tracks kept" and "tracks swapped" identical by construction.
    n = grid.shape[0]
    j, k = np.triu_indices(n, k=1)
    out = {}
    t_j, t_k = truth[j], truth[k]                     # (n_pairs, n_params)
    denom = t_k - t_j
    pred = grid[j, k]                                 # (n_pairs, n_params)
    spread = truth.std(axis=0)
    spread = np.where(spread > 0, spread, 1.0)

    keep = np.abs(denom) > MIN_TRUTH_GAP * spread     # per (pair, param)
    u = np.where(keep, (pred - t_j) / np.where(keep, denom, 1.0), np.nan)

    with np.errstate(invalid="ignore"):
        out["u_median"] = np.nanmedian(u, axis=0)
        out["u_iqr"] = (np.nanpercentile(u, 75, axis=0) - np.nanpercentile(u, 25, axis=0))
        out["frac_tracks_kept"] = np.nanmean(np.abs(u) < NEAR, axis=0)
        out["frac_tracks_swapped"] = np.nanmean(np.abs(u - 1.0) < NEAR, axis=0)
        # distance from the anchor's truth — large means the chimera pushed the
        # prediction somewhere neither simulation supports
        out["offline_dist"] = np.nanmean(np.abs(pred - t_j), axis=0) / spread
    for key in out:
        out[key] = np.nan_to_num(out[key], nan=0.0, posinf=0.0, neginf=0.0)[:n_params]
    return out


def pair_features(all_results, r2, param_names, obs1, obs2, space="processed"):
    """Feature table for one observable pair: one row per parameter."""
    cases = [r["case_name"] for r in all_results]
    ci = {c: i for i, c in enumerate(cases)}
    by_case = {r["case_name"]: r for r in all_results}
    n_params = len(param_names)
    aligned, s_obs2, s_obs1 = r2["aligned"], r2["shuf_obs2"], r2["shuf_obs1"]

    f = {"pair": f"{obs1}__{obs2}", "obs1": obs1, "obs2": obs2, "param": param_names}

    # --- R² curve -----------------------------------------------------------
    for c in cases:
        f[f"r2_{c}"] = aligned[ci[c]]
    r2_a, r2_b, r2_both = aligned[ci[A_ALONE]], aligned[ci[B_ALONE]], aligned[ci[BOTH_CLEAN]]
    best_single = np.maximum(r2_a, r2_b)
    f["r2_best_single"] = best_single
    f["r2_both_clean"] = r2_both
    f["gain_from_combining"] = r2_both - best_single
    f["single_asymmetry"] = r2_a - r2_b
    f["both_singles_low"] = np.minimum(r2_a, r2_b)

    # Degradation along each noise arm: total drop, plus curvature.
    # curvature > 0 = the midpoint sits above the chord -> graceful (redundant
    # information, case 6); < 0 = falls off a cliff early (case 4).
    for tag, arm in (("B", NOISE_ARM_B), ("A", NOISE_ARM_A)):
        lo, mid, hi = (aligned[ci[c]] for c in arm)
        f[f"degradation_{tag}"] = lo - hi
        f[f"curvature_{tag}"] = mid - 0.5 * (lo + hi)
    f["degradation_asymmetry"] = f["degradation_B"] - f["degradation_A"]

    # --- shuffle test at both-clean ----------------------------------------
    d_obs2 = s_obs2[ci[BOTH_CLEAN]] - r2_both     # obs2 shuffled -> tests reliance on obs1
    d_obs1 = s_obs1[ci[BOTH_CLEAN]] - r2_both
    f["delta_shuffle_obs2"] = d_obs2
    f["delta_shuffle_obs1"] = d_obs1
    f["shuffle_asymmetry"] = d_obs2 - d_obs1
    # 0 = all reliance on obs2, 1 = all on obs1, 0.5 = shared evenly
    f["allegiance"] = _safe_ratio(np.abs(d_obs2), np.abs(d_obs1))
    f["max_shuffle_drop"] = np.minimum(d_obs2, d_obs1)

    # --- prediction-space features -----------------------------------------
    truth = None
    for c in cases:
        pred, true = pipeline.get_case_predictions(by_case[c], mode="aligned", space=space)
        f[f"slope_{c}"] = _slopes(pred, true)
        if truth is None:
            truth = true
    f["slope_drop_B"] = f[f"slope_{NOISE_ARM_B[0]}"] - f[f"slope_{NOISE_ARM_B[-1]}"]
    f["slope_drop_A"] = f[f"slope_{NOISE_ARM_A[0]}"] - f[f"slope_{NOISE_ARM_A[-1]}"]

    # --- chimera grids at both-clean (cached by the plotting path) ----------
    result = by_case[BOTH_CLEAN]
    eval_idx = pipeline._eval_indices()
    n = len(eval_idx)
    for mode, tag in (("obs1_vs_truth", "keep_obs1"), ("obs2_vs_truth", "keep_obs2")):
        # p_idx=0 is irrelevant: the call caches the full (n*n, n_params) grid
        plots_mod._compute_all_chimera_preds(
            result, mode, pipeline.x_normalized_dict, eval_idx, 0,
            pipeline.batch_size, pipeline.device, space=space)
        flat = result[("_chimera_grid_v1", mode, tuple(np.asarray(eval_idx).tolist()))]
        grid = flat.reshape(n, n, -1)
        for key, val in _grid_stats(grid, truth, n_params).items():
            f[f"{key}_{tag}"] = val

    return pd.DataFrame(f)


def rule_based_case(df):
    """Label each row with the hand taxonomy (CLAUDE.md, cases 1-6) as a
    baseline to sanity-check clusters against. Thresholds are deliberately
    crude — this is an anchor for interpretation, not ground truth."""
    hi, gain_eps = 0.1, 0.05
    a, b = df["r2_A_clean"], df["r2_B_clean"]
    both, gain = df["r2_both_clean"], df["gain_from_combining"]
    a_hi, b_hi, both_hi = a >= hi, b >= hi, both >= hi
    graceful = (df[["curvature_A", "curvature_B"]].min(axis=1) >= 0)

    label = pd.Series("unclassified", index=df.index)
    label[~both_hi & ~a_hi & ~b_hi] = "5_unconstrained"
    label[both_hi & ~a_hi & ~b_hi] = "4_only_combined"
    one_only = (a_hi ^ b_hi) & both_hi
    label[one_only & (gain < gain_eps)] = "1_single_no_gain"
    label[one_only & (gain >= gain_eps)] = "2_single_but_gain"
    both_ok = a_hi & b_hi & both_hi
    label[both_ok & (gain >= gain_eps)] = "3_both_and_gain"
    label[both_ok & (gain < gain_eps) & graceful] = "6_both_redundant"
    label[both_ok & (gain < gain_eps) & ~graceful] = "4_only_combined"
    return label
