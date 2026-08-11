"""
Plot functions for the CAMELS SBI noise-mixing experiment.

Call configure() once after training to register shared state:

    from src import pipeline, plots
    pipeline.configure(observable_1=..., ...)
    plots.configure(all_results=all_results, output_dim=output_dim, ...)

Then all plot functions work with just the parameter index:

    for p in param_np:
        fig, stats = plots.plot_bias_progression_overlay(param=p)
"""
import sys as _sys
import os
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd

from pipeline import (
    get_case_predictions, make_val_loader_fn, make_pair_val_loader_fn,
    resolve_shuffle, sample_unique_unordered_pairs,
    predict_with_uncertainty, predict_covariance_matrix, cov_to_corr,
)

# ---------------------------------------------------------------------------
# Module-level state — set via configure() before calling plot functions.
# These names are used directly in function bodies (bare-name references).
# For names that are also used as function parameters (batch_size, device,
# perm), functions fall back to _cfg[name] when the parameter is None.
# ---------------------------------------------------------------------------
all_results = None
output_dim = None
observable_1 = None
observable_2 = None
logflag = None
means = None
stds = None
x_normalized_dict = None
y = None
idx_val = None
idx_test = None
param_names = None
noise_cases = None
batch_size = None
device = None
perm = None

_cfg = {}
_r2_matrices = {}


def _eval_idx():
    """Prefer idx_test if configured, else fall back to idx_val (pre-3-way-split behavior)."""
    return idx_test if idx_test is not None else idx_val


def configure(**kwargs):
    """Set shared state for all plot functions.

    Special keys:
        r2_matrix, r2_matrix_shifted_observable_only, r2_matrix_shifted_both
            — stored in _r2_matrices for plot_param_lines_min's R-squared strip.
    All other keys are set as module-level variables AND stored in _cfg
    (the dict is used when a function parameter shadows the module name).
    """
    r2_keys = {"r2_matrix", "r2_matrix_shifted_observable_only", "r2_matrix_shifted_both"}
    mod = _sys.modules[__name__]
    for k, v in kwargs.items():
        if k in r2_keys:
            _r2_matrices[k] = v
        else:
            setattr(mod, k, v)
    _cfg.update(kwargs)


def parse_case_name(name: str):
    """
    Parse names like:
      - '<obs>_clean'
      - '<obs1>_<noise1>_<obs2>_<noise2>'

    Returns a dict with keys:
      kind: 'single' or 'combo' or 'other'
      plus fields depending on kind.
    """
    parts = name.split('_')

    # single observable clean: '<obs>_clean'
    if parts[-1] == "clean":
        obs = "_".join(parts[:-1])
        return {"kind": "single", "obs": obs}

    # combo: '<obs1>_<noise1>_<obs2>_<noise2>'
    if len(parts) >= 4:
        noise2_str = parts[-1]
        obs2       = parts[-2]
        noise1_str = parts[-3]
        obs1       = "_".join(parts[:-3])
        try:
            noise1 = float(noise1_str)
            noise2 = float(noise2_str)
        except ValueError:
            return {"kind": "other"}
        return {
            "kind": "combo",
            "obs1": obs1,
            "noise1": noise1,
            "obs2": obs2,
            "noise2": noise2,
        }

    return {"kind": "other"}


def dual_clean_asym_order(cases):
    """
    Given a list of case names (unique, in original order),
    build an order that:

      clean_left
      -> asym block where left observable is clean
      -> both-clean combos
      -> asym block where right observable is clean
      -> clean_right
      -> symmetric noise combos
      -> anything else

    The choice of which observable is 'left' is determined so that
    the first combo in the original list is in the first asym block
    (continuity requirement).
    """
    cases = list(cases)
    parsed = {c: parse_case_name(c) for c in cases}

    singles = [c for c in cases if parsed[c]["kind"] == "single"]
    combos  = [c for c in cases if parsed[c]["kind"] == "combo"]
    others  = [c for c in cases if parsed[c]["kind"] not in ("single", "combo")]

    # If pattern not present, just return original order
    if len(singles) < 2 or not combos:
        return cases, 0

    # Map obs -> its clean single case
    obs_to_clean = {}
    for c in singles:
        info = parsed[c]
        obs_to_clean[info["obs"]] = c

    # Use the FIRST combo to decide which observable is "left":
    first_combo = parsed[combos[0]]
    obs1, obs2 = first_combo["obs1"], first_combo["obs2"]
    n1, n2     = first_combo["noise1"], first_combo["noise2"]

    # Determine which observable is clean in the first combo
    if n1 == 0.0 and n2 > 0.0:
        left_obs, right_obs = obs1, obs2
    elif n2 == 0.0 and n1 > 0.0:
        left_obs, right_obs = obs2, obs1
    else:
        # fallback if first combo is weird
        left_obs, right_obs = obs1, obs2

    # Get the corresponding clean cases
    clean_left  = obs_to_clean.get(left_obs)
    clean_right = obs_to_clean.get(right_obs)
    if clean_left is None or clean_right is None:
        # can't build the nice path, fallback
        return cases, 0

    # Now partition combos into blocks, preserving original relative order
    asym_left   = []  # left clean, right dirty
    asym_right  = []  # right clean, left dirty
    both_clean  = []  # both clean (noise 0,0)
    symmetric   = []  # symmetric noise (same nonzero)
    rest        = []  # anything else

    for c in combos:
        info = parsed[c]
        if info["kind"] != "combo":
            rest.append(c)
            continue

        # Re-express noises in terms of left/right observables
        if info["obs1"] == left_obs and info["obs2"] == right_obs:
            left_noise  = info["noise1"]
            right_noise = info["noise2"]
        elif info["obs2"] == left_obs and info["obs1"] == right_obs:
            left_noise  = info["noise2"]
            right_noise = info["noise1"]
        else:
            # unexpected observable pair
            rest.append(c)
            continue

        if left_noise == 0.0 and right_noise > 0.0:
            asym_left.append(c)
        elif right_noise == 0.0 and left_noise > 0.0:
            asym_right.append(c)
        elif left_noise == 0.0 and right_noise == 0.0:
            both_clean.append(c)
        elif left_noise == right_noise and left_noise > 0.0:
            symmetric.append(c)
        else:
            rest.append(c)

    # Build final order:
    ordered = (
        [clean_left] +
        asym_left +
        both_clean +
        asym_right +
        [clean_right] +
        symmetric +
        rest
    )

    # Define split_idx where symmetric noise starts (for vertical line)
    split_idx = len([clean_left] + asym_left + both_clean + asym_right + [clean_right])
    return ordered, split_idx


def plot_param_curve_dual(df: pd.DataFrame, param: str, figsize=(8, 5), show_band=True):
    d = df[df["param"] == param].copy()
    if d.empty:
        raise ValueError(f"No rows for param {param}")

    # Get unique cases in the order they appear
    cases_present = [c for c in dict.fromkeys(d["case"]) if pd.notna(c)]

    # === NEW: your ordering logic ===
    ordered_cases, split_idx = dual_clean_asym_order(cases_present)

    # Apply categorical ordering
    d["case"] = pd.Categorical(d["case"], categories=ordered_cases, ordered=True)
    d = d.sort_values("case", kind="stable").reset_index(drop=True)

    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=figsize)

    # resolve_shuffle's dual impl: obs1_vs_truth shuffles observable_2 (truths stay
    # with observable_1), obs2_vs_truth shuffles observable_1 (truths stay with
    # observable_2) -- label each curve with the observable actually being shuffled.
    ax.plot(x, d["r2_aligned"].to_numpy(), marker="o", linewidth=2, label="aligned")
    ax.plot(x, d["r2_shuf_obs1"].to_numpy(), marker="s", linestyle="--", linewidth=2,
            label=f"shuffled {observable_2} ({observable_1} truths)")
    ax.plot(x, d["r2_shuf_obs2"].to_numpy(), marker="^", linestyle="-.", linewidth=2,
            label=f"shuffled {observable_1} ({observable_2} truths)")

    # Shaded +/- 1 std band across shuffle_perms draws -- none on "aligned" (deterministic,
    # no draw-to-draw variance since it involves no permutation).
    if show_band and "r2_shuf_obs1_std" in d.columns:
        r2_1 = d["r2_shuf_obs1"].to_numpy()
        s1 = d["r2_shuf_obs1_std"].to_numpy()
        ax.fill_between(x, r2_1 - s1, r2_1 + s1, alpha=0.15)
    if show_band and "r2_shuf_obs2_std" in d.columns:
        r2_2 = d["r2_shuf_obs2"].to_numpy()
        s2 = d["r2_shuf_obs2_std"].to_numpy()
        ax.fill_between(x, r2_2 - s2, r2_2 + s2, alpha=0.15)

    ax.axhline(0, linestyle=":", linewidth=1)

    # Optional: vertical line separating "regular" from "symmetric noise"
    # split_idx is in *case-category* space; convert it to the row index where that category first appears.
    if 0 < split_idx < len(ordered_cases):
        first_sym_case = ordered_cases[split_idx]
        row_hits = np.where(d["case"].astype(str).to_numpy() == str(first_sym_case))[0]
        if len(row_hits) > 0:
            boundary = row_hits[0]
            ax.axvline(boundary - 0.5, linestyle=":", linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(d["case"].astype(str).tolist(), rotation=30, ha="right")
    ax.set_ylabel("R²")
    ax.set_title(f"{param}: R² vs noise (aligned + two shuffles)")
    ax.legend(frameon=False)

    plt.tight_layout()
    plt.show()


def plot_predictions_vs_true(parameters,
                             noise_cases,
                             *,
                             results=None,
                             x_dict=None,
                             y_vector=None,
                             idx=None,
                             batch_size=None,
                             param_labels=None,
                             mode="aligned",              # "aligned", "obs1_vs_truth", "obs2_vs_truth"
                             keys_to_shuffle=None,        # optional explicit observable(s) to shuffle (name or list of names)
                             perm=None,                   # optional permutation; defaults to np.random.permutation(len(idx))
                             marker_size=18,
                             save_path=None,
                             device=None):
    """
    Plot predicted vs true for selected parameters and noise cases with control over alignment/shuffling.

    parameters:
      - int or "θk" or label in param_labels, or list thereof, or "all"

    noise_cases:
      - case_name string or list of case_name strings, or "all"

    mode:
      - "aligned" (no shuffling)
      - "obs1_vs_truth" (shuffle observable_2; truths stay with observable_1)
      - "obs2_vs_truth" (shuffle observable_2 and truths together; truths follow observable_2)

    keys_to_shuffle:
      - optional override of which observable(s) to shuffle (string or list).
        If None, resolve_shuffle picks observable_2 for both comparison modes.

    perm:
      - permutation to apply within the validation split. If None, generates a fresh one of length len(idx).
    """
    if results is None: results = all_results
    if x_dict is None: x_dict = x_normalized_dict
    if y_vector is None: y_vector = y
    if idx is None: idx = _eval_idx()
    if batch_size is None: batch_size = _cfg["batch_size"]
    if device is None: device = _cfg["device"]
    if not results:
        raise ValueError("`results` is empty; train models and populate all_results first.")

    def _to_list(v):
        if isinstance(v, str) or not hasattr(v, "__iter__"):
            return [v]
        return list(v)

    # Resolve parameters
    if isinstance(parameters, str) and parameters.lower() == "all":
        param_list = list(range(output_dim))
    else:
        param_list = _to_list(parameters)

    # Resolve cases
    if isinstance(noise_cases, str) and noise_cases.lower() == "all":
        case_list = [r["case_name"] for r in results]
    else:
        case_list = _to_list(noise_cases)

    # Labels: prefer user-provided, then global param_names, else θi
    default_labels = param_labels or _cfg.get("param_names") or [f"θ{i}" for i in range(output_dim)]
    label_to_idx = {label: i for i, label in enumerate(default_labels)}

    def _resolve_param(p):
        if isinstance(p, int):
            if not 0 <= p < output_dim:
                raise ValueError(f"Parameter index {p} out of range (0..{output_dim-1}).")
            return p
        if isinstance(p, str):
            if p in label_to_idx:
                return label_to_idx[p]
            if p.startswith("θ") and p[1:].isdigit():
                i = int(p[1:])
                if not 0 <= i < output_dim:
                    raise ValueError(f"Parameter index {i} from {p} out of range.")
                return i
            if p.isdigit():
                i = int(p)
                if not 0 <= i < output_dim:
                    raise ValueError(f"Parameter index {i} from string out of range.")
                return i
        raise ValueError(f"Cannot interpret parameter identifier: {p}")

    param_indices = [_resolve_param(p) for p in param_list]
    param_labels_resolved = [default_labels[i] for i in param_indices]

    # Validate cases
    case_to_result = {r["case_name"]: r for r in results}
    missing = [c for c in case_list if c not in case_to_result]
    if missing:
        raise ValueError(f"Cases not found in all_results: {missing}")

    # Predictions come from the shared cache on each result (get_case_predictions) --
    # defaulting perm to the global `perm` (rather than a fresh random one) means this
    # hits the same cache entry the main shuffle loops already populated.
    def _collect_predictions(result):
        return get_case_predictions(result, mode=mode, perm=perm, keys_to_shuffle=keys_to_shuffle)

    # Plot grid: rows = cases, cols = parameters
    n_rows = len(case_list)
    n_cols = len(param_indices)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)

    for r, case_name in enumerate(case_list):
        preds, trues = _collect_predictions(case_to_result[case_name])
        for c, (p_idx, p_label) in enumerate(zip(param_indices, param_labels_resolved)):
            ax = axes[r, c]
            y_true = trues[:, p_idx]
            y_pred = preds[:, p_idx]
            ax.scatter(y_true, y_pred, s=marker_size, alpha=0.6, edgecolor="none")

            lo = float(min(y_true.min(), y_pred.min()))
            hi = float(max(y_true.max(), y_pred.max()))
            ax.plot([lo, hi], [lo, hi], "r--", lw=1.5)

            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            title_prefix = {"aligned": "Aligned",
                            "obs1_vs_truth": f"{observable_2} shuffled, {observable_1} truths",
                            "obs2_vs_truth": f"{observable_2} shuffled, {observable_2} truths"}[mode]
            ax.set_title(f"{title_prefix}\n{case_name} | {p_label}\nR²={r2:.3f}, RMSE={rmse:.3f}")
            if r == n_rows - 1:
                ax.set_xlabel("True")
            if c == 0:
                ax.set_ylabel("Predicted")
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200)
    return fig


def plot_param_all_val_lines(param,
                             *,
                             mode="aligned",          # "aligned", "obs1_vs_truth", "obs2_vs_truth"
                             n_sims=30,
                             seed=None,               # for reproducible sampling of sims
                             sim_indices=None,        # optional explicit indices within validation set (0..len(_eval_idx())-1)
                             cases="auto",            # "auto" uses ordered subset; or list of case names
                             keys_to_shuffle=None,    # optional explicit observable(s) to shuffle for obs1/obs2
                             perm=None,               # optional permutation (defaults to global 'perm' or fresh)
                             line_alpha=0.35,
                             line_width=1.2,
                             marker_size=0,
                             show_median=True,
                             results=None,
                             x_dict=None,
                             y_vector=None,
                             idx=None,
                             batch_size=None,
                             param_labels=None,
                             device=None,
                             save_path=None):
    """
    For a single parameter, plot lines for randomly selected validation sims.
    Each line connects that sim’s predicted value across noise cases (ordered core only).

    mode:
      - "aligned": no shuffle (X, Y aligned)
      - "obs1_vs_truth": shuffle the first selected observable only; truths aligned
      - "obs2_vs_truth": shuffle the second selected observable only; truths aligned
    """
    if results is None: results = all_results
    if x_dict is None: x_dict = x_normalized_dict
    if y_vector is None: y_vector = y
    if idx is None: idx = _eval_idx()
    if batch_size is None: batch_size = _cfg["batch_size"]
    if device is None: device = _cfg["device"]
    if not results:
        raise ValueError("`results` is empty; train models and populate all_results first.")

    # Resolve parameter index and label
    default_labels = param_labels or _cfg.get("param_names") or [f"θ{i}" for i in range(output_dim)]
    label_to_idx = {label: i for i, label in enumerate(default_labels)}
    if isinstance(param, int):
        if not 0 <= param < output_dim: raise ValueError(f"Parameter index {param} out of range.")
        param_idx = param
    elif isinstance(param, str):
        if param in label_to_idx:
            param_idx = label_to_idx[param]
        elif param.startswith("θ") and param[1:].isdigit():
            param_idx = int(param[1:])
        elif param.isdigit():
            param_idx = int(param)
        else:
            raise ValueError(f"Cannot interpret parameter identifier: {param}")
        if not 0 <= param_idx < output_dim: raise ValueError(f"Parameter index {param_idx} out of range.")
    else:
        raise ValueError("param must be an int or a string like 'θ3'.")
    param_label = default_labels[param_idx]

    # Build ordered case list (exclude symmetric/rest)
    all_case_names = [r["case_name"] for r in results]
    ordered_all, split_idx = dual_clean_asym_order(all_case_names)
    ordered_core = ordered_all[:split_idx]  # clean_left + asym_left + both_clean + asym_right + clean_right

    if cases == "auto":
        final_cases = [c for c in ordered_core if c in all_case_names]
    else:
        wanted = list(cases)
        final_cases = [c for c in ordered_core if c in wanted]
    if not final_cases:
        raise ValueError("No cases to plot after applying ordering/selection.")

    case_to_result = {r["case_name"]: r for r in results}

    # For obs1/obs2 modes, require >=2 observables per case
    if mode in ("obs1_vs_truth", "obs2_vs_truth"):
        final_cases = [c for c in final_cases if len(case_to_result[c]["selected_observables"]) >= 2]
        if not final_cases:
            raise ValueError("No multi-observable cases available for this mode after filtering.")

    # Permutation for shuffles (consistent across all cases)
    if perm is None:
        perm = _cfg.get("perm")
        if perm is None or len(perm) != len(idx):
            perm = np.random.permutation(len(idx))
    perm = np.asarray(perm)
    if len(perm) != len(idx):
        raise ValueError("perm length must match len(_eval_idx()).")

    # Predictions come from the shared cache on each result (get_case_predictions).
    def _collect_predictions(result):
        return get_case_predictions(result, mode=mode, perm=perm, keys_to_shuffle=keys_to_shuffle)

    # Collect predictions for each case into [n_cases, n_val]
    preds_by_case = []
    for case in final_cases:
        preds, _ = _collect_predictions(case_to_result[case])
        preds_by_case.append(preds[:, param_idx])
    preds_by_case = np.stack(preds_by_case, axis=0)  # [n_cases, n_val]

    # Choose sims to plot
    n_val = preds_by_case.shape[1]
    if sim_indices is None:
        rng = np.random.default_rng(seed)
        k = min(n_sims, n_val)
        sim_indices = rng.choice(n_val, size=k, replace=False)
    else:
        sim_indices = np.asarray(sim_indices)
        if np.any((sim_indices < 0) | (sim_indices >= n_val)):
            raise ValueError("sim_indices out of range for validation set.")

    # Plot lines: one per sim across cases
    x = np.arange(len(final_cases))
    fig, ax = plt.subplots(figsize=(max(7, len(final_cases)*0.6), 4.8))

    for i, s in enumerate(sim_indices):
        y_line = preds_by_case[:, s]  # length = n_cases
        # label only the first to avoid clutter
        ax.plot(x, y_line, color="tab:blue", alpha=line_alpha, linewidth=line_width,
                marker="o" if marker_size > 0 else None, markersize=marker_size,
                label="Sim (random)" if i == 0 else None)

    # Optional median across all sims at each case
    if show_median:
        med = np.median(preds_by_case, axis=1)
        ax.plot(x, med, color="k", linewidth=2, marker="o", markersize=4, label="Median")

    ax.set_xticks(x)
    ax.set_xticklabels(final_cases, rotation=60, ha="right")
    mode_title = {"aligned": "Aligned", "obs1_vs_truth": "Obs1 shuffled vs Truth", "obs2_vs_truth": "Obs2 shuffled vs Truth"}[mode]
    ax.set_title(f"{mode_title} — {param_label} (lines for {len(sim_indices)} random val sims)")
    ax.set_ylabel("Predicted value")
    ax.grid(True, alpha=0.3)
    if show_median:
        ax.legend(loc="best")

    fig.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200)
    return fig


def plot_param_lines_min(
    param,
    mode="aligned",          # "aligned", "obs1_vs_truth", "obs2_vs_truth"
    n_sims=30,
    seed=None,
    cases="auto",            # "auto" -> ordered core (excludes symmetric/rest); or list of case names
    keys_to_shuffle=None,     # optional explicit observable(s) to shuffle for obs1/obs2 modes
    residual=False,           # if True, plot (pred-true)/true with thresholds

    # NEW: case-dependent tolerance for residuals
    # tol(err,true) = max(precision_mult * RMSE_case, truth_rel_tol * |true|)
    # - set precision_mult or truth_rel_tol to None to disable that component
    precision_mult=None,
    truth_rel_tol=None,
    truth_abs_floor=None,      # min |true| used in denominator when residual=True
    threshold_action="zero",  # "zero" (within tol => 0) or "mask" (within tol => NaN)
    show_median=True,
    show_r2_strip=True,
    annotate_r2=True,
    r2_cmap="Spectral",
    r2_vmin=-1.0,
    r2_vmax=1.0,
    marker_size=0,
    line_alpha=0.6,
    line_width=1.2,
    save_path=None,

    # NEW: make each sim-line distinguishable
    distinct_lines=True,           # if True, vary color/style per sim
    distinct_mode="color+style",   # "color", "style", "color+style"
    cmap_name="tab20",             # colormap used when distinct_mode includes color
    shuffle_style_assignment=True, # shuffle color assignment so adjacent lines differ more
    legend_sims=False,             # if True, legend entry per sim (can be cluttered)
):
    # Resolve parameter index/label (supports int, "θk", or label in param_names)
    labels = _cfg.get("param_names") or [f"θ{i}" for i in range(output_dim)]
    lbl2idx = {l: i for i, l in enumerate(labels)}
    if isinstance(param, int):
        pidx = param
    elif isinstance(param, str):
        pidx = lbl2idx.get(param, None)
        if pidx is None and param.startswith("θ") and param[1:].isdigit():
            pidx = int(param[1:])
        if pidx is None and param.isdigit():
            pidx = int(param)
    else:
        raise TypeError("param must be int or str")

    if pidx is None or pidx < 0 or pidx >= len(labels):
        raise ValueError(f"Could not resolve param={param!r} to a valid index")

    param_label = labels[pidx]

    # Build ordered core cases
    all_case_names = [r["case_name"] for r in all_results]
    ordered_all, split_idx = dual_clean_asym_order(all_case_names)
    ordered_core = ordered_all[:split_idx]

    if cases == "auto":
        final_cases = [c for c in ordered_core if c in all_case_names]
    else:
        wanted = list(cases)
        final_cases = [c for c in ordered_core if c in wanted]

    case_to_result = {r["case_name"]: r for r in all_results}
    if mode in ("obs1_vs_truth", "obs2_vs_truth"):
        final_cases = [
            c for c in final_cases
            if len(case_to_result[c]["selected_observables"]) >= 2
        ]

    if len(final_cases) == 0:
        raise ValueError("No cases to plot after filtering (final_cases is empty).")

    # Use global perm if present; otherwise create one
    _perm = _cfg.get("perm")
    if _perm is None or len(_perm) != len(_eval_idx()):
        rng_perm = np.random.default_rng(seed)
        _perm = rng_perm.permutation(len(_eval_idx()))
    _perm = np.asarray(_perm)

    # For a case, choose which keys to shuffle
    def choose_shuffle_keys(res):
        if mode == "aligned":
            return None
        if keys_to_shuffle is not None:
            return [keys_to_shuffle] if isinstance(keys_to_shuffle, str) else list(keys_to_shuffle)
        ks = sorted(res["selected_observables"].keys())
        return [ks[0]] if mode == "obs1_vs_truth" else [ks[1]]

    # Collect predictions (and truths if residual) for full val set per case, then stack
    preds_by_case = []
    trues_by_case = []
    for c in final_cases:
        res = case_to_result[c]
        skeys = choose_shuffle_keys(res)

        loader_fn = make_val_loader_fn(
            selected_observables=res["selected_observables"],
            x_dict=x_normalized_dict,
            y_vector=y,
            idx=idx_val,
            batch_size=batch_size,
            key_to_shuffle=skeys,
            perm=_perm if skeys is not None else None,
            shuffle_y=False,
        )
        loader = loader_fn()
        model = res["model"].to(device)

        P, T = [], []
        model.eval()
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                P.append(model(xb).cpu())
                T.append(yb.cpu())

        P = torch.cat(P).numpy()
        T = torch.cat(T).numpy()

        # Undo normalization/log
        P = P * stds + means
        T = T * stds + means
        P[:, logflag] = np.exp(P[:, logflag])
        T[:, logflag] = np.exp(T[:, logflag])

        preds_by_case.append(P[:, pidx])
        trues_by_case.append(T[:, pidx])

    preds_by_case = np.stack(preds_by_case, 0)  # [n_cases, n_val]
    trues_by_case = np.stack(trues_by_case, 0)

    # Choose sims to draw lines
    n_val = preds_by_case.shape[1]
    rng = np.random.default_rng(seed)
    k = min(n_sims, n_val)
    sims = rng.choice(n_val, size=k, replace=False)

    # Values to plot
    if residual:
        err = (preds_by_case - trues_by_case)

        # Build per-element tolerance: max(case precision, truth-relative)
        tol = None
        if precision_mult is not None:
            precision_mult = float(precision_mult)
            if precision_mult < 0:
                raise ValueError("precision_mult must be >= 0.")
            rmse_case = np.sqrt(np.mean(err ** 2, axis=1))  # [n_cases]
            abs_tol = (precision_mult * rmse_case)[:, None]  # [n_cases, 1]
            tol = abs_tol

        if truth_rel_tol is not None:
            truth_rel_tol = float(truth_rel_tol)
            if truth_rel_tol < 0:
                raise ValueError("truth_rel_tol must be >= 0.")
            rel_tol = truth_rel_tol * np.abs(trues_by_case)
            tol = rel_tol if tol is None else np.maximum(tol, rel_tol)

        if tol is not None:
            within = np.abs(err) <= tol
            if threshold_action == "zero":
                err = np.where(within, 0.0, err)
            elif threshold_action == "mask":
                err = np.where(within, np.nan, err)
            else:
                raise ValueError("threshold_action must be 'zero' or 'mask'.")

        denom = trues_by_case
        if truth_abs_floor is not None:
            floor = float(truth_abs_floor)
            if floor <= 0:
                raise ValueError("truth_abs_floor must be > 0.")
            denom = np.where(np.abs(denom) >= floor, denom, np.where(denom >= 0, floor, -floor))

        Y = err / denom
    else:
        Y = preds_by_case

    # Plot
    x = np.arange(len(final_cases))
    r2_by_case = None
    if show_r2_strip:
        matrix_name = {
            "aligned": "r2_matrix",
            "obs1_vs_truth": "r2_matrix_shifted_observable_only",
            "obs2_vs_truth": "r2_matrix_shifted_both",
        }[mode]
        r2_mat = _r2_matrices.get(matrix_name)
        if r2_mat is None:
            raise ValueError(f"{matrix_name} is not defined; compute the R² matrices before using show_r2_strip=True.")
        all_case_order = [r["case_name"] for r in all_results]
        case_to_ridx = {case_name: i for i, case_name in enumerate(all_case_order)}
        r2_by_case = np.array([r2_mat[case_to_ridx[c], pidx] for c in final_cases], dtype=float)
        fig = plt.figure(figsize=(max(7.8, len(final_cases) * 0.62), 5.7))
        gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[12, 0.9], hspace=0.05)
        ax = fig.add_subplot(gs[0, 0])
        ax_r2 = fig.add_subplot(gs[1, 0], sharex=ax)
    else:
        fig, ax = plt.subplots(figsize=(max(7, len(final_cases) * 0.6), 4.8))
        ax_r2 = None

    # --- NEW: build distinguishable styles for each sim line ---
    if distinct_lines:
        rng_style = np.random.default_rng(seed if seed is not None else 0)

        # colors
        if distinct_mode in ("color", "color+style"):
            cmap = plt.get_cmap(cmap_name)
            colors = [cmap(i / max(1, k - 1)) for i in range(k)]
            if shuffle_style_assignment:
                rng_style.shuffle(colors)
        else:
            colors = [None] * k

        # styles/markers
        if distinct_mode in ("style", "color+style"):
            line_styles = ["-", "--", "-.", ":"]
            markers = ["o", "s", "^", "D", "v", "P", "X", "*", ">", "<"]
        else:
            line_styles = ["-"]
            markers = [None]
    # -----------------------------------------------------------

    for i, s in enumerate(sims):
        y_line = Y[:, s]

        if distinct_lines:
            color = colors[i] if colors[i] is not None else "tab:blue"
            ls = line_styles[i % len(line_styles)]
            mk = markers[i % len(markers)] if marker_size > 0 else None
        else:
            color = "tab:blue"
            ls = "-"
            mk = "o" if marker_size > 0 else None

        ax.plot(
            x, y_line,
            color=color,
            linestyle=ls,
            alpha=line_alpha,
            linewidth=line_width,
            marker=mk,
            markersize=marker_size,
            label=(f"Sim {int(s)}" if legend_sims else ("Sim (random)" if i == 0 else None)),
        )

    if show_median:
        med = np.median(Y, axis=1)
        ax.plot(x, med, color="k", linewidth=2, marker="o", markersize=4, label="Median")

    ax.set_xticks(x)
    ax.set_xticklabels(final_cases, rotation=60, ha="right")
    mode_title = {
        "aligned": "Aligned",
        "obs1_vs_truth": "Obs1 shuffled vs Truth",
        "obs2_vs_truth": "Obs2 shuffled vs Truth",
    }[mode]
    ax.set_title(f"{mode_title} — {param_label} (lines for {k} sims)")
    ax.set_ylabel("Residual (pred − true)" if residual else "Predicted value")
    ax.grid(True, alpha=0.3)
    if show_r2_strip:
        plt.setp(ax.get_xticklabels(), visible=False)

    if show_median or legend_sims:
        ax.legend(loc="best", fontsize=9)

    if show_r2_strip:
        im = ax_r2.imshow(
            r2_by_case[None, :],
            aspect="auto",
            cmap=r2_cmap,
            vmin=r2_vmin,
            vmax=r2_vmax,
            extent=(-0.5, len(final_cases) - 0.5, 0.0, 1.0),
        )
        ax_r2.set_yticks([])
        ax_r2.set_ylabel("R²", rotation=0, labelpad=14, va="center")
        ax_r2.set_xticks(x)
        ax_r2.set_xticklabels(final_cases, rotation=60, ha="right")
        ax_r2.tick_params(axis="x", length=0)
        for spine in ax_r2.spines.values():
            spine.set_visible(False)
        for xpos in np.arange(0.5, len(final_cases) - 0.5, 1.0):
            ax_r2.axvline(xpos, color="black", linewidth=0.8, alpha=0.9)
        if annotate_r2:
            mid = 0.5 * (r2_vmin + r2_vmax)
            for xi, val in enumerate(r2_by_case):
                txt_color = "white" if val < mid else "black"
                ax_r2.text(xi, 0.5, f"{val:.2f}", ha="center", va="center", fontsize=8, color=txt_color)
        cbar = fig.colorbar(im, ax=[ax, ax_r2], orientation="vertical", fraction=0.035, pad=0.02)
        cbar.set_label("R²")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
    return fig


def plot_param_all_val(param,
                       *,
                       mode="aligned",          # "aligned", "obs1_vs_truth", "obs2_vs_truth"
                       cases="auto",            # "auto" uses ordered subset; or list of case names
                       keys_to_shuffle=None,    # optional explicit observable(s) to shuffle for obs1/obs2
                       perm=None,               # optional permutation (defaults to global 'perm' or fresh)
                       jitter=0.15,             # x jitter so points don't overlap
                       marker_size=18,
                       alpha=0.6,
                       show_median=True,
                       results=None,
                       x_dict=None,
                       y_vector=None,
                       idx=None,
                       batch_size=None,
                       param_labels=None,
                       device=None,
                       save_path=None):
    """
    Plot scatter for one parameter across ALL validation sims:
      - x-axis: noise cases (ordered: clean_left, asym_left, both_clean, asym_right, clean_right)
      - each point: predicted value for one sim under that case and truth mode.

    mode:
      - "aligned": no shuffle (X, Y aligned)
      - "obs1_vs_truth": shuffle the first selected observable only; truths aligned
      - "obs2_vs_truth": shuffle the second selected observable only; truths aligned
    """
    if results is None: results = all_results
    if x_dict is None: x_dict = x_normalized_dict
    if y_vector is None: y_vector = y
    if idx is None: idx = _eval_idx()
    if batch_size is None: batch_size = _cfg["batch_size"]
    if device is None: device = _cfg["device"]
    if not results:
        raise ValueError("`results` is empty; train models and populate all_results first.")

    # Resolve parameter index and label
    default_labels = param_labels or _cfg.get("param_names") or [f"θ{i}" for i in range(output_dim)]
    label_to_idx = {label: i for i, label in enumerate(default_labels)}
    if isinstance(param, int):
        if not 0 <= param < output_dim:
            raise ValueError(f"Parameter index {param} out of range (0..{output_dim-1}).")
        param_idx = param
    elif isinstance(param, str):
        if param in label_to_idx:
            param_idx = label_to_idx[param]
        elif param.startswith("θ") and param[1:].isdigit():
            param_idx = int(param[1:])
        elif param.isdigit():
            param_idx = int(param)
        else:
            raise ValueError(f"Cannot interpret parameter identifier: {param}")
        if not 0 <= param_idx < output_dim:
            raise ValueError(f"Parameter index {param_idx} out of range.")
    else:
        raise ValueError("param must be an int or a string like 'θ3'.")

    param_label = default_labels[param_idx]

    # Build ordered case list (exclude symmetric/rest)
    all_case_names = [r["case_name"] for r in results]
    ordered_all, split_idx = dual_clean_asym_order(all_case_names)
    ordered_core = ordered_all[:split_idx]  # clean_left + asym_left + both_clean + asym_right + clean_right

    if cases == "auto":
        final_cases = [c for c in ordered_core if c in all_case_names]
    else:
        wanted = list(cases)
        # keep requested cases but maintain the ordered_core sequence
        final_cases = [c for c in ordered_core if c in wanted]

    if not final_cases:
        raise ValueError("No cases to plot after applying ordering/selection.")

    case_to_result = {r["case_name"]: r for r in results}

    # Permutation for shuffles (consistent across all cases)
    if perm is None:
        perm = _cfg.get("perm")
        if perm is None or len(perm) != len(idx):
            perm = np.random.permutation(len(idx))
    perm = np.asarray(perm)
    if len(perm) != len(idx):
        raise ValueError("perm length must match len(_eval_idx()).")

    # Predictions come from the shared cache on each result (get_case_predictions).
    def _collect_predictions(result):
        return get_case_predictions(result, mode=mode, perm=perm, keys_to_shuffle=keys_to_shuffle)

    # Build scatter data
    xs, ys = [], []
    medians = []
    for ci, case in enumerate(final_cases):
        preds, _trues = _collect_predictions(case_to_result[case])
        y_pred_case = preds[:, param_idx]  # shape [n_val]
        # jittered x positions for each val sample
        x_center = float(ci)
        x_vals = x_center + (np.random.rand(len(y_pred_case)) - 0.5) * 2 * jitter
        xs.append(x_vals)
        ys.append(y_pred_case.astype(float))
        medians.append(np.median(y_pred_case))

    # Plot
    fig, ax = plt.subplots(figsize=(max(7, len(final_cases)*0.6), 4.8))
    for ci in range(len(final_cases)):
        ax.scatter(xs[ci], ys[ci], s=marker_size, alpha=alpha)

    # Optional median marker per case
    if show_median:
        ax.plot(range(len(final_cases)), medians, color="k", linewidth=2, marker="o", markersize=5, label="Median")

    ax.set_xticks(range(len(final_cases)))
    ax.set_xticklabels(final_cases, rotation=60, ha="right")
    mode_title = {"aligned": "Aligned", "obs1_vs_truth": "Obs1 shuffled vs Truth", "obs2_vs_truth": "Obs2 shuffled vs Truth"}[mode]
    ax.set_title(f"{mode_title} — {param_label} (all validation sims)")
    ax.set_ylabel("Predicted value")
    ax.grid(True, alpha=0.3)
    if show_median:
        ax.legend(loc="best")

    fig.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200)
    return fig


def plot_param_pair_normalized_values(
    param,
    *,
    mode="obs1_vs_truth",        # "obs1_vs_truth" or "obs2_vs_truth" or "aligned"
    n_pairs=500,
    seed=None,
    pair_indices=None,

    cases="auto",
    keys_to_shuffle=None,
    perm=None,

    # choose which truth is sim1 (0) and sim2 (1)
    normalize_endpoints="obs1_to_obs2",  # "obs1_to_obs2" or "obs2_to_obs1"

    # NEW: choose space for normalization/plotting
    # "physical": exp() logflag params (current behavior)
    # "log": keep log-space values (no exp)
    space="physical",

    # NEW: make obs1/obs2 definition explicit
    # "sorted" -> obs1/obs2 are the first/second keys in sorted(selected_observables.keys())
    # or pass a 2-tuple/list of keys: (obs1_key, obs2_key)
    obs_key_order="sorted",

    # plotting
    line_alpha=0.55,
    line_width=1.5,
    marker_size=3,
    show_median=True,
    show_reference_lines=True,

    # clip + styling
    clip_range=(-3.0, 3.0),
    y_limits=(-3.0, 3.0),
    distinct_styles=True,

    drop_degenerate_pairs=True,

    # data/model context
    results=None,
    x_dict=None,
    y_vector=None,
    idx=None,
    batch_size=None,
    param_labels=None,
    device=None,

    save_path=None,
):
    """
    Pair-normalized prediction value for random validation rows (pairs) across noise cases.

    Definitions:
      - "validation row j" = one sample in the validation DataLoader (after shuffling inputs if enabled).
      - There are two observable channels in each case. We label them:
            obs1_key, obs2_key
        By default these are the first/second keys in sorted(selected_observables.keys()).
      - In shuffling modes:
            obs1_vs_truth: obs1_key is permuted (its inputs come from row perm[j])
            obs2_vs_truth: obs2_key is permuted
        Y/truth remains from the base row j (idx[j]) in the DataLoader, but for analysis we also define
        a "truth tied to each observable" based on which sim provided that observable.

    For each row j we define two observable-tied truths:
      t_obs1(j) = truth of the sim that supplied obs1 in that row
      t_obs2(j) = truth of the sim that supplied obs2 in that row

    We then choose sim1->0 and sim2->1 via normalize_endpoints:
      obs1_to_obs2: t0=t_obs1, t1=t_obs2
      obs2_to_obs1: t0=t_obs2, t1=t_obs1

    Pair-normalized value:
        p_norm(case, j) = (pred(case,j) - t0(j)) / (t1(j) - t0(j))

    The 'space' argument controls whether the above is computed in physical space (exp applied)
    or log space (no exp).
    """
    if results is None: results = all_results
    if x_dict is None: x_dict = x_normalized_dict
    if y_vector is None: y_vector = y
    if idx is None: idx = _eval_idx()
    if batch_size is None: batch_size = _cfg["batch_size"]
    if device is None: device = _cfg["device"]
    if not results:
        raise ValueError("`results` is empty; train models and populate all_results first.")

    # --- Resolve parameter index/label ---
    default_labels = param_labels or _cfg.get("param_names") or [f"θ{i}" for i in range(output_dim)]
    label_to_idx = {label: i for i, label in enumerate(default_labels)}
    if isinstance(param, int):
        if not 0 <= param < output_dim:
            raise ValueError(f"Parameter index {param} out of range.")
        param_idx = param
    elif isinstance(param, str):
        if param in label_to_idx:
            param_idx = label_to_idx[param]
        elif param.startswith("θ") and param[1:].isdigit():
            param_idx = int(param[1:])
        elif param.isdigit():
            param_idx = int(param)
        else:
            raise ValueError(f"Cannot interpret parameter identifier: {param}")
        if not 0 <= param_idx < output_dim:
            raise ValueError(f"Parameter index {param_idx} out of range.")
    else:
        raise ValueError("param must be an int or a string like 'θ3'.")
    param_label = default_labels[param_idx]

    # --- Build ordered case list ---
    all_case_names = [r["case_name"] for r in results]
    ordered_all, split_idx = dual_clean_asym_order(all_case_names)
    ordered_core = ordered_all[:split_idx]

    if cases == "auto":
        final_cases = [c for c in ordered_core if c in all_case_names]
    else:
        wanted = list(cases)
        final_cases = [c for c in ordered_core if c in wanted]
    if not final_cases:
        raise ValueError("No cases to plot after applying ordering/selection.")

    case_to_result = {r["case_name"]: r for r in results}

    # --- Require >=2 observables (pairs) ---
    final_cases = [c for c in final_cases if len(case_to_result[c]["selected_observables"]) >= 2]
    if not final_cases:
        raise ValueError("No multi-observable cases available (need >=2 observables).")

    # --- Perm for shuffles ---
    if perm is None:
        perm = _cfg.get("perm")
        if perm is None or len(perm) != len(idx):
            perm = np.random.permutation(len(idx))
    perm = np.asarray(perm)
    if len(perm) != len(idx):
        raise ValueError("perm length must match len(_eval_idx()).")

    # Predictions come from the shared cache on each result (get_case_predictions).
    # This function reconstructs both truth alignments itself (t_obs1/t_obs2 below)
    # from which keys moved, so it never needs shuffle_y -- resolve_shuffle already
    # always returns shuffle_y=False, so the shared cache's behavior matches.
    def _resolve_shuffle_keys(result):
        if keys_to_shuffle is not None:
            return [keys_to_shuffle] if isinstance(keys_to_shuffle, str) else list(keys_to_shuffle)
        keys, _ = resolve_shuffle(result["selected_observables"], mode)
        return sorted(keys) or None

    def _collect_predictions(result):
        return get_case_predictions(result, mode=mode, perm=perm, keys_to_shuffle=keys_to_shuffle, space=space)

    # --- Collect predictions per case: [n_cases, n_val] ---
    preds_by_case = []
    for case in final_cases:
        preds, _ = _collect_predictions(case_to_result[case])
        preds_by_case.append(preds[:, param_idx])
    preds_by_case = np.stack(preds_by_case, axis=0)

    # --- Get truth array in idx-order ---
    _, true_ref = _collect_predictions(case_to_result[final_cases[0]])
    n_val = true_ref.shape[0]

    # --- Identify obs1/obs2 keys ---
    keys_sorted = sorted(case_to_result[final_cases[0]]["selected_observables"].keys())
    if obs_key_order == "sorted":
        # follow the analysis-wide pair rather than alphabetical order
        obs1_key, obs2_key = observable_1, observable_2
        if obs1_key not in keys_sorted or obs2_key not in keys_sorted:
            raise ValueError(
                f"{observable_1}/{observable_2} not both in selected_observables. Available: {keys_sorted}"
            )
    else:
        # explicit keys
        if not (isinstance(obs_key_order, (tuple, list)) and len(obs_key_order) == 2):
            raise ValueError("obs_key_order must be 'sorted' or a 2-tuple/list (obs1_key, obs2_key).")
        obs1_key, obs2_key = obs_key_order
        # sanity check: ensure they exist
        if obs1_key not in keys_sorted or obs2_key not in keys_sorted:
            raise ValueError(f"obs_key_order keys must be in selected_observables. Available: {keys_sorted}")

    # Drop cases that don't contain both keys (consistency)
    kept_cases, kept_preds = [], []
    for i, case in enumerate(final_cases):
        keys_i = set(case_to_result[case]["selected_observables"].keys())
        if obs1_key in keys_i and obs2_key in keys_i:
            kept_cases.append(case)
            kept_preds.append(preds_by_case[i])
    if not kept_cases:
        raise ValueError("No cases contain both obs1 and obs2 keys consistently.")
    final_cases = kept_cases
    preds_by_case = np.stack(kept_preds, axis=0)
    n_cases = preds_by_case.shape[0]

    # Determine which keys are shuffled (using first kept case)
    shuffle_keys = set(_resolve_shuffle_keys(case_to_result[final_cases[0]]) or [])

    # Truth arrays for base and permuted rows
    truth_base = true_ref[:, param_idx]
    truth_perm = true_ref[perm, param_idx]

    # Truth tied to each observable for every row j
    t_obs1 = truth_perm if obs1_key in shuffle_keys else truth_base
    t_obs2 = truth_perm if obs2_key in shuffle_keys else truth_base

    # Choose sim1/sim2 endpoints
    if normalize_endpoints == "obs1_to_obs2":
        t0, t1 = t_obs1, t_obs2
        endpoint_desc = f"sim1={obs1_key}→0, sim2={obs2_key}→1"
    elif normalize_endpoints == "obs2_to_obs1":
        t0, t1 = t_obs2, t_obs1
        endpoint_desc = f"sim1={obs2_key}→0, sim2={obs1_key}→1"
    else:
        raise ValueError("normalize_endpoints must be 'obs1_to_obs2' or 'obs2_to_obs1'.")

    denom = (t1 - t0)

    # Choose which rows (pairs) to plot
    if pair_indices is None:
        rng = np.random.default_rng(seed)
        k = min(n_pairs, n_val)
        pair_indices = rng.choice(n_val, size=k, replace=False)
    else:
        pair_indices = np.asarray(pair_indices)
        if np.any((pair_indices < 0) | (pair_indices >= n_val)):
            raise ValueError("pair_indices out of range for validation set.")

    if drop_degenerate_pairs:
        keep = denom[pair_indices] != 0
        pair_indices = pair_indices[keep]
        if pair_indices.size == 0:
            raise ValueError("All selected pairs were degenerate for this parameter (t1 == t0).")

    # Pair-normalize
    P = preds_by_case[:, pair_indices]
    t0_sel = t0[pair_indices][None, :]
    denom_sel = denom[pair_indices][None, :]
    P_norm = (P - t0_sel) / denom_sel

    if clip_range is not None:
        lo, hi = clip_range
        P_norm = np.clip(P_norm, lo, hi)

    # --- Plot ---
    x = np.arange(n_cases)
    fig, ax = plt.subplots(figsize=(max(7, n_cases * 0.6), 4.8))

    if show_reference_lines:
        ax.axhline(0.0, linewidth=1.0, alpha=0.6)
        ax.axhline(1.0, linewidth=1.0, alpha=0.6)

    linestyles = ["-", "--", "-.", ":"]
    for i in range(P_norm.shape[1]):
        ls = linestyles[i % len(linestyles)] if distinct_styles else "-"
        ax.plot(
            x, P_norm[:, i],
            alpha=line_alpha,
            linewidth=line_width,
            linestyle=ls,
            marker="o" if marker_size > 0 else None,
            markersize=marker_size,
        )

    if show_median:
        med = np.median(P_norm, axis=1)
        ax.plot(x, med, color="k", linewidth=2.2, marker="o", markersize=4, label="Median")

    shuffled_desc = (
        f"shuffled={sorted(shuffle_keys)}" if mode != "aligned" else "shuffled=None (aligned)"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(final_cases, rotation=60, ha="right")
    ax.set_ylabel(f"Pair-normalized value (space={space})")
    ax.set_title(f"{mode} — {param_label} ({endpoint_desc}; {shuffled_desc}; {P_norm.shape[1]} pairs)")
    ax.grid(True, alpha=0.3)
    if y_limits is not None:
        ax.set_ylim(y_limits)
    else:
        ax.set_ylim(-0.05, 1.05)

    if show_median:
        ax.legend(loc="best")

    fig.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200)

    return fig


def inspect_unordered_pair_denominator_distribution(
    param,
    *,
    obs1_key=None,
    obs2_key=None,
    normalize_endpoints="obs1_to_obs2",
    n_pairs=5000,
    seed=0,
    pairs=None,
    cases="auto",
    space="processed",
    bins=60,
    min_abs_denom=None,
    results=None,
    x_dict=None,
    y_vector=None,
    idx=None,
    batch_size=None,
    param_labels=None,
    device=None,
):
    if results is None: results = all_results
    if x_dict is None: x_dict = x_normalized_dict
    if y_vector is None: y_vector = y
    if idx is None: idx = _eval_idx()
    if batch_size is None: batch_size = _cfg["batch_size"]
    if device is None: device = _cfg["device"]
    if obs1_key is None: obs1_key = observable_1
    if obs2_key is None: obs2_key = observable_2
    if not results:
        raise ValueError("`results` is empty; train models and populate all_results first.")

    default_labels = param_labels or _cfg.get("param_names") or [f"θ{i}" for i in range(output_dim)]
    label_to_idx = {label: i for i, label in enumerate(default_labels)}
    if isinstance(param, int):
        param_idx = param
    elif isinstance(param, str):
        if param in label_to_idx:
            param_idx = label_to_idx[param]
        elif param.startswith("θ") and param[1:].isdigit():
            param_idx = int(param[1:])
        elif param.isdigit():
            param_idx = int(param)
        else:
            raise ValueError(f"Cannot interpret parameter identifier: {param}")
    else:
        raise ValueError("param must be an int or a string like 'θ3'.")
    if not (0 <= param_idx < output_dim):
        raise ValueError(f"Parameter index {param_idx} out of range.")
    param_label = default_labels[param_idx]

    def _convert_space(arr):
        arr = np.array(arr, copy=True)
        if space == "processed":
            return arr
        arr = arr * stds + means
        if space == "log":
            return arr
        if space == "physical":
            arr[:, logflag] = np.exp(arr[:, logflag])
            return arr
        raise ValueError("space must be 'processed', 'log', or 'physical'.")

    all_case_names = [r["case_name"] for r in results]
    ordered_all, split_idx = dual_clean_asym_order(all_case_names)
    ordered_core = ordered_all[:split_idx]
    if cases == "auto":
        final_cases = [c for c in ordered_core if c in all_case_names]
    else:
        wanted = list(cases)
        final_cases = [c for c in ordered_core if c in wanted]
    case_to_result = {r["case_name"]: r for r in results}
    final_cases = [c for c in final_cases if obs1_key in case_to_result[c]["selected_observables"] and obs2_key in case_to_result[c]["selected_observables"]]
    if not final_cases:
        raise ValueError(f"No cases contain both obs keys: {obs1_key}, {obs2_key}.")

    n_val = len(idx)
    if pairs is None:
        pairs = sample_unique_unordered_pairs(n_val=n_val, n_pairs=n_pairs, seed=seed)
    else:
        pairs = np.asarray(pairs, dtype=int)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must be shape [n_pairs, 2].")
        if np.any(pairs < 0) or np.any(pairs >= n_val):
            raise ValueError("pairs contain indices outside [0, n_val).")
        pairs = np.sort(pairs, axis=1)
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
        if pairs.shape[0] == 0:
            raise ValueError("No valid pairs after removing i==j.")

    ref_case = final_cases[0]
    result = case_to_result[ref_case]
    cache_key = (
        "_unordered_pair_cache_v2",
        ref_case,
        obs1_key,
        obs2_key,
        space,
        tuple(map(int, pairs.ravel())),
    )
    if cache_key not in result:
        model = result["model"].to(device)
        loader = make_pair_val_loader_fn(
            selected_observables=result["selected_observables"],
            x_dict=x_dict,
            y_vector=y_vector,
            idx=idx,
            batch_size=batch_size,
            obs1_key=obs1_key,
            obs2_key=obs2_key,
            pairs=pairs,
        )
        preds = []
        ycats = []
        model.eval()
        with torch.no_grad():
            for xb, ybcat in loader:
                xb = xb.to(device)
                preds.append(model(xb).cpu())
                ycats.append(ybcat.cpu())
        pred_np = torch.cat(preds).numpy()
        ycat_np = torch.cat(ycats).numpy()
        y_i_np = _convert_space(ycat_np[:, :output_dim])
        y_j_np = _convert_space(ycat_np[:, output_dim:])
        pred_np = _convert_space(pred_np)
        result[cache_key] = (pred_np, y_i_np, y_j_np)
    _, y_i_np, y_j_np = result[cache_key]

    truth_i = y_i_np[:, param_idx].copy()
    truth_j = y_j_np[:, param_idx].copy()
    if normalize_endpoints == "obs1_to_obs2":
        t0 = truth_i
        t1 = truth_j
    elif normalize_endpoints == "obs2_to_obs1":
        t0 = truth_j
        t1 = truth_i
    else:
        raise ValueError("normalize_endpoints must be 'obs1_to_obs2' or 'obs2_to_obs1'.")

    denom = t1 - t0
    denom_abs = np.abs(denom)
    keep_mask = np.ones_like(denom, dtype=bool)
    if min_abs_denom is not None:
        keep_mask &= (denom_abs >= float(min_abs_denom))

    quantiles = np.quantile(denom_abs, [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].hist(denom_abs, bins=bins, color="tab:blue", alpha=0.8)
    if min_abs_denom is not None:
        axes[0].axvline(float(min_abs_denom), color="red", linestyle="--", linewidth=1.5)
    axes[0].set_xlabel("|t1 - t0|")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Denominator magnitude: {param_label}")
    axes[0].grid(True, alpha=0.3)

    xs = np.sort(denom_abs)
    ys = np.arange(1, xs.size + 1) / xs.size
    axes[1].plot(xs, ys, linewidth=2)
    if min_abs_denom is not None:
        axes[1].axvline(float(min_abs_denom), color="red", linestyle="--", linewidth=1.5)
    axes[1].set_xlabel("|t1 - t0|")
    axes[1].set_ylabel("CDF")
    axes[1].set_title("Cumulative distribution")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        f"Unordered-pair denominator distribution for {param_label} ({space} space)\n"
        f"obs1={obs1_key}, obs2={obs2_key}, sampled pairs={pairs.shape[0]}, kept if thresholded={int(np.sum(keep_mask))}"
    )
    fig.tight_layout()

    stats = {
        "pairs": pairs,
        "denom": denom,
        "denom_abs": denom_abs,
        "quantiles": {
            "min": float(quantiles[0]),
            "p01": float(quantiles[1]),
            "p05": float(quantiles[2]),
            "p10": float(quantiles[3]),
            "p25": float(quantiles[4]),
            "p50": float(quantiles[5]),
            "p75": float(quantiles[6]),
            "p90": float(quantiles[7]),
            "p95": float(quantiles[8]),
            "p99": float(quantiles[9]),
            "max": float(quantiles[10]),
        },
        "fraction_below_threshold": (float(np.mean(denom_abs < float(min_abs_denom))) if min_abs_denom is not None else None),
        "n_pairs_total": int(pairs.shape[0]),
        "n_pairs_kept": int(np.sum(keep_mask)),
        "reference_case": ref_case,
        "note": "Denominator distribution is case-independent for the current unordered-pair setup; it is computed from the first compatible case only.",
    }
    return fig, stats


def plot_raw_parameter_histogram(
    param,
    *,
    space="processed",
    bins=60,
    density=False,
    idx=None,
    y_vector=None,
    param_labels=None,
):
    if y_vector is None: y_vector = y
    if idx is None: idx = _eval_idx()
    default_labels = param_labels or _cfg.get("param_names") or [f"θ{i}" for i in range(output_dim)]
    label_to_idx = {label: i for i, label in enumerate(default_labels)}
    if isinstance(param, int):
        param_idx = param
    elif isinstance(param, str):
        if param in label_to_idx:
            param_idx = label_to_idx[param]
        elif param.startswith("θ") and param[1:].isdigit():
            param_idx = int(param[1:])
        elif param.isdigit():
            param_idx = int(param)
        else:
            raise ValueError(f"Cannot interpret parameter identifier: {param}")
    else:
        raise ValueError("param must be an int or a string like 'θ3'.")
    if not (0 <= param_idx < output_dim):
        raise ValueError(f"Parameter index {param_idx} out of range.")

    param_label = default_labels[param_idx]
    idx = np.asarray(idx, dtype=int)
    values = np.array(y_vector[idx], copy=True)

    if space == "processed":
        pass
    elif space == "log":
        values = values * stds + means
    elif space == "physical":
        values = values * stds + means
        values[:, logflag] = np.exp(values[:, logflag])
    else:
        raise ValueError("space must be 'processed', 'log', or 'physical'.")

    vals = values[:, param_idx]
    quantiles = np.quantile(vals, [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0])

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.hist(vals, bins=bins, density=density, color="tab:blue", alpha=0.8)
    ax.set_xlabel(f"{param_label} value")
    ax.set_ylabel("Density" if density else "Count")
    ax.set_title(f"Raw distribution of {param_label} ({space} space)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    stats = {
        "values": vals,
        "space": space,
        "param_idx": int(param_idx),
        "param_label": param_label,
        "quantiles": {
            "min": float(quantiles[0]),
            "p01": float(quantiles[1]),
            "p05": float(quantiles[2]),
            "p10": float(quantiles[3]),
            "p25": float(quantiles[4]),
            "p50": float(quantiles[5]),
            "p75": float(quantiles[6]),
            "p90": float(quantiles[7]),
            "p95": float(quantiles[8]),
            "p99": float(quantiles[9]),
            "max": float(quantiles[10]),
        },
    }
    return fig, stats


# ============================================================
# 3) PLOT: unordered pairs + pair-normalization in chosen target space
# ============================================================


def plot_param_unordered_pair_normalized_values(
    param,
    *,
    obs1_key=None,
    obs2_key=None,

    normalize_endpoints="obs1_to_obs2",   # or "obs2_to_obs1"

    n_pairs=5000,
    seed=0,
    pairs=None,

    cases="auto",

    # default is now the exact processed model target space
    # "processed" : standardized partially logged target space
    # "log"       : undo standardization only
    # "physical"  : undo standardization + exp(logflag)
    space="processed",

    # keep only one global endpoint-separation filter
    min_abs_denom=None,
    drop_degenerate_pairs=True,

    show_reference_lines=True,
    show_some_lines=True,
    max_lines=120,
    line_alpha=0.20,
    line_width=1.0,

    show_mean=True,
    show_median=True,
    show_errorbars=True,
    errorbar_capsize=3,
    show_r2_strip=True,
    annotate_r2=True,
    r2_cmap="Spectral",
    r2_vmin=-1.0,
    r2_vmax=1.0,

    clip_range=None,
    y_limits=None,

    results=None,
    x_dict=None,
    y_vector=None,
    idx=None,
    batch_size=None,
    param_labels=None,
    device=None,

    save_path=None,
):
    """
    For each sampled unordered pair (i,j) with i<j:
      - build the input from obs1_key(sim i) + obs2_key(sim j)
      - get the predicted parameter and the two endpoint truths
      - compute pair-normalized position:
            P_norm = (pred - t0) / (t1 - t0)

    Spaces:
      - processed : standardized partially logged target space (default)
      - log       : undo standardization only
      - physical  : undo standardization and exp(logflag columns)

    The denominator filter is global:
      - drop exact degeneracies if requested
      - optionally require |t1 - t0| >= min_abs_denom
    """
    if results is None: results = all_results
    if x_dict is None: x_dict = x_normalized_dict
    if y_vector is None: y_vector = y
    if idx is None: idx = _eval_idx()
    if batch_size is None: batch_size = _cfg["batch_size"]
    if device is None: device = _cfg["device"]
    if obs1_key is None: obs1_key = observable_1
    if obs2_key is None: obs2_key = observable_2
    if not results:
        raise ValueError("`results` is empty; train models and populate all_results first.")

    # -----------------------------
    # Resolve parameter index/label
    # -----------------------------
    default_labels = param_labels or _cfg.get("param_names") or [f"θ{i}" for i in range(output_dim)]
    label_to_idx = {label: i for i, label in enumerate(default_labels)}

    if isinstance(param, int):
        param_idx = param
    elif isinstance(param, str):
        if param in label_to_idx:
            param_idx = label_to_idx[param]
        elif param.startswith("θ") and param[1:].isdigit():
            param_idx = int(param[1:])
        elif param.isdigit():
            param_idx = int(param)
        else:
            raise ValueError(f"Cannot interpret parameter identifier: {param}")
    else:
        raise ValueError("param must be an int or a string like 'θ3'.")

    if not (0 <= param_idx < output_dim):
        raise ValueError(f"Parameter index {param_idx} out of range.")

    param_label = default_labels[param_idx]

    # -----------------------------
    # Helper: move arrays into requested space
    # -----------------------------
    def _convert_space(arr):
        arr = np.array(arr, copy=True)

        if space == "processed":
            return arr

        arr = arr * stds + means

        if space == "log":
            return arr

        if space == "physical":
            arr[:, logflag] = np.exp(arr[:, logflag])
            return arr

        raise ValueError("space must be 'processed', 'log', or 'physical'.")

    # -----------------------------
    # Build ordered case list
    # -----------------------------
    all_case_names = [r["case_name"] for r in results]
    ordered_all, split_idx = dual_clean_asym_order(all_case_names)
    ordered_core = ordered_all[:split_idx]

    if cases == "auto":
        final_cases = [c for c in ordered_core if c in all_case_names]
    else:
        wanted = list(cases)
        final_cases = [c for c in ordered_core if c in wanted]

    if not final_cases:
        raise ValueError("No cases to plot after applying ordering/selection.")

    case_to_result = {r["case_name"]: r for r in results}

    # Keep only cases containing both obs keys
    kept = []
    for c in final_cases:
        keys = set(case_to_result[c]["selected_observables"].keys())
        if obs1_key in keys and obs2_key in keys:
            kept.append(c)

    final_cases = kept
    if not final_cases:
        raise ValueError(f"No cases contain both obs keys: {obs1_key}, {obs2_key}.")

    # -----------------------------
    # Sample or validate pairs
    # -----------------------------
    n_val = len(idx)
    if pairs is None:
        pairs = sample_unique_unordered_pairs(n_val=n_val, n_pairs=n_pairs, seed=seed)
    else:
        pairs = np.asarray(pairs, dtype=int)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must be shape [n_pairs, 2].")
        if np.any(pairs < 0) or np.any(pairs >= n_val):
            raise ValueError("pairs contain indices outside [0, n_val).")
        pairs = np.sort(pairs, axis=1)
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
        if pairs.shape[0] == 0:
            raise ValueError("No valid pairs after removing i==j.")

    # -----------------------------
    # Evaluate each case on the SAME pair dataset
    # -----------------------------
    preds_by_case = []
    truth_i = None
    truth_j = None

    for case in final_cases:
        result = case_to_result[case]

        cache_key = (
            "_unordered_pair_cache_v2",
            case,
            obs1_key,
            obs2_key,
            space,
            tuple(map(int, pairs.ravel()))
        )

        if cache_key not in result:
            model = result["model"].to(device)

            loader = make_pair_val_loader_fn(
                selected_observables=result["selected_observables"],
                x_dict=x_dict,
                y_vector=y_vector,
                idx=idx,
                batch_size=batch_size,
                obs1_key=obs1_key,
                obs2_key=obs2_key,
                pairs=pairs,
            )

            preds = []
            ycats = []
            model.eval()
            with torch.no_grad():
                for xb, ybcat in loader:
                    xb = xb.to(device)
                    preds.append(model(xb).cpu())
                    ycats.append(ybcat.cpu())

            pred_np = torch.cat(preds).numpy()     # [n_pairs, output_dim]
            ycat_np = torch.cat(ycats).numpy()     # [n_pairs, 2*output_dim]

            y_i_np = ycat_np[:, :output_dim]
            y_j_np = ycat_np[:, output_dim:]

            pred_np = _convert_space(pred_np)
            y_i_np = _convert_space(y_i_np)
            y_j_np = _convert_space(y_j_np)

            result[cache_key] = (pred_np, y_i_np, y_j_np)

        pred_np, y_i_np, y_j_np = result[cache_key]

        preds_by_case.append(pred_np[:, param_idx])

        if truth_i is None:
            truth_i = y_i_np[:, param_idx].copy()
            truth_j = y_j_np[:, param_idx].copy()

    preds_by_case = np.stack(preds_by_case, axis=0)   # [n_cases, n_pairs]
    n_cases = preds_by_case.shape[0]

    # -----------------------------
    # Endpoint truths
    # -----------------------------
    if normalize_endpoints == "obs1_to_obs2":
        t0 = truth_i
        t1 = truth_j
        endpoint_desc = f"sim1={obs1_key}→0, sim2={obs2_key}→1"
    elif normalize_endpoints == "obs2_to_obs1":
        t0 = truth_j
        t1 = truth_i
        endpoint_desc = f"sim1={obs2_key}→0, sim2={obs1_key}→1"
    else:
        raise ValueError("normalize_endpoints must be 'obs1_to_obs2' or 'obs2_to_obs1'.")

    denom = t1 - t0

    # -----------------------------
    # Global denominator filter only
    # -----------------------------
    mask = np.ones_like(denom, dtype=bool)

    if drop_degenerate_pairs:
        mask &= (denom != 0)

    if min_abs_denom is not None:
        mask &= (np.abs(denom) >= float(min_abs_denom))

    if not np.any(mask):
        raise ValueError("No pairs left after denominator filtering.")

    preds_by_case = preds_by_case[:, mask]
    t0 = t0[mask]
    t1 = t1[mask]
    denom = denom[mask]
    pairs_kept = pairs[mask]

    # -----------------------------
    # Pair-normalization
    # -----------------------------
    P_norm_raw = (preds_by_case - t0[None, :]) / denom[None, :]

    # Extra direct closeness diagnostic
    d0 = np.abs(preds_by_case - t0[None, :])
    d1 = np.abs(preds_by_case - t1[None, :])
    frac_closer_to_0 = np.mean(d0 < d1, axis=1)
    frac_closer_to_1 = np.mean(d1 < d0, axis=1)
    frac_exact_ties  = np.mean(d0 == d1, axis=1)

    mean = np.mean(P_norm_raw, axis=1)
    std = np.std(P_norm_raw, axis=1, ddof=1) if P_norm_raw.shape[1] > 1 else np.zeros_like(mean)
    median = np.median(P_norm_raw, axis=1)

    P_norm_vis = P_norm_raw
    if clip_range is not None:
        lo, hi = clip_range
        P_norm_vis = np.clip(P_norm_raw, lo, hi)

    # -----------------------------
    # Plot
    # -----------------------------
    x = np.arange(n_cases)
    r2_by_case = None
    if show_r2_strip:
        r2_mat = _r2_matrices.get("r2_matrix")
        if r2_mat is None:
            raise ValueError("r2_matrix is not defined; compute the aligned R² matrix before using show_r2_strip=True.")
        all_case_order = [r["case_name"] for r in all_results]
        case_to_ridx = {case_name: i for i, case_name in enumerate(all_case_order)}
        r2_by_case = np.array([r2_mat[case_to_ridx[c], param_idx] for c in final_cases], dtype=float)
        fig = plt.figure(figsize=(max(7.8, n_cases * 0.62), 6.1))
        gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[12, 0.9], hspace=0.05)
        ax = fig.add_subplot(gs[0, 0])
        ax_r2 = fig.add_subplot(gs[1, 0], sharex=ax)
    else:
        fig, ax = plt.subplots(figsize=(max(7, n_cases * 0.6), 5.2))
        ax_r2 = None

    if show_reference_lines:
        ax.axhline(0.0, linewidth=1.0, alpha=0.6)
        ax.axhline(1.0, linewidth=1.0, alpha=0.6)

    if show_some_lines:
        n_draw = min(max_lines, P_norm_vis.shape[1])
        for k in range(n_draw):
            ax.plot(x, P_norm_vis[:, k], alpha=line_alpha, linewidth=line_width)

    if show_mean:
        if show_errorbars:
            ax.errorbar(
                x, mean, yerr=std,
                fmt="o-", linewidth=2.4, markersize=4,
                capsize=errorbar_capsize, elinewidth=1.6,
                label="Mean ± 1σ"
            )
        else:
            ax.plot(x, mean, "o-", linewidth=2.4, markersize=4, label="Mean")

    if show_median:
        ax.plot(x, median, "s-", linewidth=2.0, markersize=4, label="Median")

    ax.set_xticks(x)
    ax.set_xticklabels(final_cases, rotation=60, ha="right")
    ax.set_ylabel(f"Pair-normalized value (space={space})")
    ax.set_title(
        f"{param_label} — unordered pairs ({endpoint_desc})\n"
        f"kept {P_norm_raw.shape[1]} pairs (of {pairs.shape[0]}), min_abs_denom={min_abs_denom}"
    )
    ax.grid(True, alpha=0.3)
    if show_r2_strip:
        plt.setp(ax.get_xticklabels(), visible=False)

    if y_limits is not None:
        ax.set_ylim(y_limits)

    ax.legend(loc="best")

    if show_r2_strip:
        im = ax_r2.imshow(
            r2_by_case[None, :],
            aspect="auto",
            cmap=r2_cmap,
            vmin=r2_vmin,
            vmax=r2_vmax,
            extent=(-0.5, len(final_cases) - 0.5, 0.0, 1.0),
        )
        ax_r2.set_yticks([])
        ax_r2.set_ylabel("R²", rotation=0, labelpad=14, va="center")
        ax_r2.set_xticks(x)
        ax_r2.set_xticklabels(final_cases, rotation=60, ha="right")
        ax_r2.tick_params(axis="x", length=0)
        for spine in ax_r2.spines.values():
            spine.set_visible(False)
        for xpos in np.arange(0.5, len(final_cases) - 0.5, 1.0):
            ax_r2.axvline(xpos, color="black", linewidth=0.8, alpha=0.9)
        if annotate_r2:
            mid = 0.5 * (r2_vmin + r2_vmax)
            for xi, val in enumerate(r2_by_case):
                txt_color = "white" if val < mid else "black"
                ax_r2.text(xi, 0.5, f"{val:.2f}", ha="center", va="center", fontsize=8, color=txt_color)
        cbar = fig.colorbar(im, ax=[ax, ax_r2], orientation="vertical", fraction=0.035, pad=0.02)
        cbar.set_label("R²")
    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200)

    stats = {
        "pairs_kept": pairs_kept,
        "mean": mean,
        "std": std,
        "median": median,
        "n_pairs_kept": int(P_norm_raw.shape[1]),
        "n_pairs_total": int(pairs.shape[0]),
        "space": space,
        "endpoint_desc": endpoint_desc,
        "min_abs_denom": min_abs_denom,
        "frac_closer_to_0": frac_closer_to_0,
        "frac_closer_to_1": frac_closer_to_1,
        "frac_exact_ties": frac_exact_ties,
    }
    return fig, stats


def plot_param_unordered_pair_decomposed_values(
    param,
    *,
    obs1_key=None,
    obs2_key=None,
    n_pairs=5000,
    seed=0,
    pairs=None,
    cases="auto",
    case_subset="all",   # "all", "aligned", "unaligned"
    space="processed",   # "processed", "log", or "physical"
    mode="directional",  # "directional" or "absolute"
    min_abs_denom=None,
    drop_degenerate_pairs=True,

    show_reference_lines=True,
    show_some_lines=True,
    max_lines=120,
    line_alpha=0.20,
    line_width=1.0,

    show_mean=True,
    show_median=True,
    show_errorbars=True,
    errorbar_capsize=3,
    show_r2_strip=True,
    annotate_r2=True,
    r2_cmap="Spectral",
    r2_vmin=-1.0,
    r2_vmax=1.0,

    clip_range=None,
    y_limits=None,
    view_margin=None,
    auto_clip_to_view=False,

    results=None,
    x_dict=None,
    y_vector=None,
    idx=None,
    batch_size=None,
    param_labels=None,
    device=None,

    save_path=None,
):
    if results is None: results = all_results
    if x_dict is None: x_dict = x_normalized_dict
    if y_vector is None: y_vector = y
    if idx is None: idx = _eval_idx()
    if batch_size is None: batch_size = _cfg["batch_size"]
    if device is None: device = _cfg["device"]
    if obs1_key is None: obs1_key = observable_1
    if obs2_key is None: obs2_key = observable_2
    if not results:
        raise ValueError("`results` is empty; train models and populate all_results first.")

    obs1_label = "observable 1"
    obs2_label = "observable 2"

    default_labels = param_labels or _cfg.get("param_names") or [f"θ{i}" for i in range(output_dim)]
    label_to_idx = {label: i for i, label in enumerate(default_labels)}

    if isinstance(param, int):
        param_idx = param
    elif isinstance(param, str):
        if param in label_to_idx:
            param_idx = label_to_idx[param]
        elif param.startswith("θ") and param[1:].isdigit():
            param_idx = int(param[1:])
        elif param.isdigit():
            param_idx = int(param)
        else:
            raise ValueError(f"Cannot interpret parameter identifier: {param}")
    else:
        raise ValueError("param must be an int or a string like 'θ3'.")

    if not (0 <= param_idx < output_dim):
        raise ValueError(f"Parameter index {param_idx} out of range.")

    param_label = default_labels[param_idx]

    def _convert_space(arr):
        arr = np.array(arr, copy=True)
        if space == "processed":
            return arr
        arr = arr * stds + means
        if space == "log":
            return arr
        if space == "physical":
            arr[:, logflag] = np.exp(arr[:, logflag])
            return arr
        raise ValueError("space must be 'processed', 'log', or 'physical'.")

    all_case_names = [r["case_name"] for r in results]
    ordered_all, split_idx = dual_clean_asym_order(all_case_names)

    aligned_cases = [c for c in ordered_all[:split_idx] if c in all_case_names]
    unaligned_cases = [c for c in ordered_all[split_idx:] if c in all_case_names]

    if case_subset == "all":
        base_cases = aligned_cases + unaligned_cases
    elif case_subset == "aligned":
        base_cases = aligned_cases
    elif case_subset == "unaligned":
        base_cases = unaligned_cases
    else:
        raise ValueError("case_subset must be 'all', 'aligned', or 'unaligned'.")

    if cases == "auto":
        final_cases = base_cases
    else:
        wanted = list(cases)
        final_cases = [c for c in base_cases if c in wanted]

    if not final_cases:
        raise ValueError("No cases to plot after applying ordering/selection.")

    case_to_result = {r["case_name"]: r for r in results}

    kept = []
    for c in final_cases:
        keys = set(case_to_result[c]["selected_observables"].keys())
        if obs1_key in keys and obs2_key in keys:
            kept.append(c)
    final_cases = kept

    if not final_cases:
        raise ValueError(f"No cases contain both obs keys: {obs1_key}, {obs2_key}.")

    n_val = len(idx)
    if pairs is None:
        pairs = sample_unique_unordered_pairs(n_val=n_val, n_pairs=n_pairs, seed=seed)
    else:
        pairs = np.asarray(pairs, dtype=int)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must be shape [n_pairs, 2].")
        if np.any(pairs < 0) or np.any(pairs >= n_val):
            raise ValueError("pairs contain indices outside [0, n_val).")
        pairs = np.sort(pairs, axis=1)
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
        if pairs.shape[0] == 0:
            raise ValueError("No valid pairs after removing i==j.")

    preds_by_case = []
    truth_i = None
    truth_j = None

    for case in final_cases:
        result = case_to_result[case]

        cache_key = (
            "_unordered_pair_cache_v3",
            case,
            obs1_key,
            obs2_key,
            space,
            tuple(map(int, pairs.ravel()))
        )

        if cache_key not in result:
            model = result["model"].to(device)

            loader = make_pair_val_loader_fn(
                selected_observables=result["selected_observables"],
                x_dict=x_dict,
                y_vector=y_vector,
                idx=idx,
                batch_size=batch_size,
                obs1_key=obs1_key,
                obs2_key=obs2_key,
                pairs=pairs,
            )

            preds = []
            ycats = []
            model.eval()
            with torch.no_grad():
                for xb, ybcat in loader:
                    xb = xb.to(device)
                    preds.append(model(xb).cpu())
                    ycats.append(ybcat.cpu())

            pred_np = torch.cat(preds).numpy()
            ycat_np = torch.cat(ycats).numpy()

            y_i_np = ycat_np[:, :output_dim]
            y_j_np = ycat_np[:, output_dim:]

            pred_np = _convert_space(pred_np)
            y_i_np = _convert_space(y_i_np)
            y_j_np = _convert_space(y_j_np)

            result[cache_key] = (pred_np, y_i_np, y_j_np)

        pred_np, y_i_np, y_j_np = result[cache_key]

        preds_by_case.append(pred_np[:, param_idx])

        if truth_i is None:
            truth_i = y_i_np[:, param_idx].copy()
            truth_j = y_j_np[:, param_idx].copy()

    preds_by_case = np.stack(preds_by_case, axis=0)

    denom = truth_j - truth_i
    mask = np.ones_like(denom, dtype=bool)

    if drop_degenerate_pairs:
        mask &= (denom != 0)
    if min_abs_denom is not None:
        mask &= (np.abs(denom) >= float(min_abs_denom))

    if not np.any(mask):
        raise ValueError("No pairs left after denominator filtering.")

    preds_by_case = preds_by_case[:, mask]
    truth_i = truth_i[mask]
    truth_j = truth_j[mask]
    pairs_kept = pairs[mask]

    if mode == "directional":
        delta1_raw = preds_by_case - truth_i[None, :]
        delta2_raw = preds_by_case - truth_j[None, :]
        ylab1 = f"Prediction - truth from {obs1_label}"
        ylab2 = f"Prediction - truth from {obs2_label}"
    elif mode == "absolute":
        delta1_raw = np.abs(preds_by_case - truth_i[None, :])
        delta2_raw = np.abs(preds_by_case - truth_j[None, :])
        ylab1 = f"|Prediction - truth from {obs1_label}|"
        ylab2 = f"|Prediction - truth from {obs2_label}|"
    else:
        raise ValueError("mode must be 'directional' or 'absolute'.")

    def _summarize(arr):
        mean = np.mean(arr, axis=1)
        std = np.std(arr, axis=1, ddof=1) if arr.shape[1] > 1 else np.zeros(arr.shape[0])
        median = np.median(arr, axis=1)
        return mean, std, median

    mean1, std1, median1 = _summarize(delta1_raw)
    mean2, std2, median2 = _summarize(delta2_raw)

    delta1_vis = delta1_raw
    delta2_vis = delta2_raw

    if view_margin is not None and auto_clip_to_view:
        vm = float(view_margin)
        if vm < 0:
            raise ValueError("view_margin must be >= 0.")
        lo, hi = -vm, 1 + vm
        delta1_vis = np.clip(delta1_raw, lo, hi)
        delta2_vis = np.clip(delta2_raw, lo, hi)
    elif clip_range is not None:
        lo, hi = clip_range
        delta1_vis = np.clip(delta1_raw, lo, hi)
        delta2_vis = np.clip(delta2_raw, lo, hi)

    x = np.arange(len(final_cases))
    r2_by_case = None
    if show_r2_strip:
        r2_mat = _r2_matrices.get("r2_matrix")
        if r2_mat is None:
            raise ValueError("r2_matrix is not defined; compute the aligned R² matrix before using show_r2_strip=True.")
        all_case_order = [r["case_name"] for r in all_results]
        case_to_ridx = {case_name: i for i, case_name in enumerate(all_case_order)}
        r2_by_case = np.array([r2_mat[case_to_ridx[c], param_idx] for c in final_cases], dtype=float)
        fig = plt.figure(figsize=(max(7.8, len(final_cases) * 0.62), 9.4))
        gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[8, 8, 0.9], hspace=0.08)
        ax_top = fig.add_subplot(gs[0, 0])
        ax_bottom = fig.add_subplot(gs[1, 0], sharex=ax_top)
        ax_r2 = fig.add_subplot(gs[2, 0], sharex=ax_top)
        axes = [ax_top, ax_bottom]
    else:
        fig, axes = plt.subplots(2, 1, figsize=(max(7, len(final_cases) * 0.6), 8.5), sharex=True)
        ax_r2 = None

    panels = [
        (axes[0], delta1_vis, mean1, std1, median1, ylab1),
        (axes[1], delta2_vis, mean2, std2, median2, ylab2),
    ]

    for ax, arr_vis, mean, std, median, ylabel in panels:
        if show_reference_lines:
            ax.axhline(0.0, linewidth=1.0, alpha=0.6)

        if show_some_lines:
            n_draw = min(max_lines, arr_vis.shape[1])
            for k in range(n_draw):
                ax.plot(x, arr_vis[:, k], alpha=line_alpha, linewidth=line_width)

        if show_mean:
            if show_errorbars:
                ax.errorbar(
                    x, mean, yerr=std,
                    fmt="o-", linewidth=2.4, markersize=4,
                    capsize=errorbar_capsize, elinewidth=1.6,
                    label="Mean ± 1σ"
                )
            else:
                ax.plot(x, mean, "o-", linewidth=2.4, markersize=4, label="Mean")

        if show_median:
            ax.plot(x, median, "s-", linewidth=2.0, markersize=4, label="Median")

        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

        if view_margin is not None:
            vm = float(view_margin)
            if vm < 0:
                raise ValueError("view_margin must be >= 0.")
            ax.set_ylim(-vm, 1 + vm)
        elif y_limits is not None:
            ax.set_ylim(y_limits)

        ax.legend(loc="best")

    if show_r2_strip:
        plt.setp(axes[0].get_xticklabels(), visible=False)
        plt.setp(axes[1].get_xticklabels(), visible=False)
        im = ax_r2.imshow(
            r2_by_case[None, :],
            aspect="auto",
            cmap=r2_cmap,
            vmin=r2_vmin,
            vmax=r2_vmax,
            extent=(-0.5, len(final_cases) - 0.5, 0.0, 1.0),
        )
        ax_r2.set_yticks([])
        ax_r2.set_ylabel("R²", rotation=0, labelpad=14, va="center")
        ax_r2.set_xticks(x)
        ax_r2.set_xticklabels(final_cases, rotation=60, ha="right")
        ax_r2.tick_params(axis="x", length=0)
        for spine in ax_r2.spines.values():
            spine.set_visible(False)
        for xpos in np.arange(0.5, len(final_cases) - 0.5, 1.0):
            ax_r2.axvline(xpos, color="black", linewidth=0.8, alpha=0.9)
        if annotate_r2:
            mid = 0.5 * (r2_vmin + r2_vmax)
            for xi, val in enumerate(r2_by_case):
                txt_color = "white" if val < mid else "black"
                ax_r2.text(xi, 0.5, f"{val:.2f}", ha="center", va="center", fontsize=8, color=txt_color)
        cbar = fig.colorbar(im, ax=[axes[0], axes[1], ax_r2], orientation="vertical", fraction=0.035, pad=0.02)
        cbar.set_label("R²")
    else:
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(final_cases, rotation=60, ha="right")

    fig.suptitle(
        f"{param_label} — decomposed unordered-pair view ({space} space, mode={mode})\n"
        f"kept {delta1_raw.shape[1]} pairs (of {pairs.shape[0]}), min_abs_denom={min_abs_denom}"
    )
    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200)

    stats = {
        "pairs_kept": pairs_kept,
        "delta1_mean": mean1,
        "delta1_std": std1,
        "delta1_median": median1,
        "delta2_mean": mean2,
        "delta2_std": std2,
        "delta2_median": median2,
        "n_pairs_kept": int(delta1_raw.shape[1]),
        "n_pairs_total": int(pairs.shape[0]),
        "space": space,
        "mode": mode,
        "min_abs_denom": min_abs_denom,
    }

    return fig, stats


def _constraint_2d_reduce_observable(arr, *, component=None, reducer="mean"):
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr

    if component is not None:
        if not (0 <= int(component) < arr.shape[1]):
            raise ValueError(f"component={component} out of range for observable with shape {arr.shape}")
        return arr[:, int(component)]

    if reducer == "mean":
        return np.mean(arr, axis=1)
    if reducer == "median":
        return np.median(arr, axis=1)
    if reducer == "sum":
        return np.sum(arr, axis=1)

    raise ValueError("reducer must be one of {'mean', 'median', 'sum'} when component is None.")


def _collect_constraint_2d_case_arrays(
    case_name,
    *,
    mode="aligned",          # "aligned", "obs1_vs_truth", "obs2_vs_truth"
    obs1_key=None,
    obs2_key=None,
    keys_to_shuffle=None,
    perm=None,
    target_space="physical", # "processed", "log", "physical"
    results=None,
    x_dict=None,
    y_vector=None,
    idx=None,
    batch_size=None,
    device=None,
):
    if results is None: results = all_results
    if x_dict is None: x_dict = x_normalized_dict
    if y_vector is None: y_vector = y
    if idx is None: idx = _eval_idx()
    if batch_size is None: batch_size = _cfg["batch_size"]
    if device is None: device = _cfg["device"]
    if obs1_key is None: obs1_key = observable_1
    if obs2_key is None: obs2_key = observable_2
    if not results:
        raise ValueError("`results` is empty; train models and populate all_results first.")

    case_to_result = {r["case_name"]: r for r in results}
    if case_name not in case_to_result:
        raise ValueError(f"Unknown case_name: {case_name}")

    result = case_to_result[case_name]
    selected_keys = sorted(result["selected_observables"].keys())

    # default to the analysis-wide pair rather than alphabetical order
    if obs1_key is None:
        obs1_key = observable_1
    if obs2_key is None:
        obs2_key = observable_2

    if obs1_key not in result["selected_observables"] or obs2_key not in result["selected_observables"]:
        raise ValueError(
            f"obs1_key={obs1_key!r} and obs2_key={obs2_key!r} must both be in selected_observables for {case_name}."
        )

    def _resolve_shuffle_keys():
        # obs1_vs_truth keeps obs1 aligned to truth by shuffling obs2 (and vice
        # versa) -- matches the canonical resolve_shuffle convention. shuffle_y
        # stays False; truth alignment is read off which key got shuffled.
        if mode == "aligned":
            return None
        if keys_to_shuffle is not None:
            return [keys_to_shuffle] if isinstance(keys_to_shuffle, str) else list(keys_to_shuffle)
        if mode == "obs1_vs_truth":
            return [obs2_key]
        if mode == "obs2_vs_truth":
            return [obs1_key]
        raise ValueError("mode must be 'aligned', 'obs1_vs_truth', or 'obs2_vs_truth'.")

    shuffle_keys = _resolve_shuffle_keys()

    if perm is None and shuffle_keys is not None:
        perm = _cfg.get("perm")
        if perm is None or len(perm) != len(idx):
            perm = np.random.permutation(len(idx))
    if perm is not None:
        perm = np.asarray(perm)
        if len(perm) != len(idx):
            raise ValueError("perm length must match len(idx).")

    cache_key = (
        "_constraint_2d_cache_v1",
        case_name,
        obs1_key,
        obs2_key,
        mode,
        tuple(shuffle_keys or []),
        tuple(map(int, perm)) if perm is not None else None,
        target_space,
    )

    if cache_key not in result:
        model = result["model"].to(device)
        loader_fn = make_val_loader_fn(
            selected_observables=result["selected_observables"],
            x_dict=x_dict,
            y_vector=y_vector,
            idx=idx,
            batch_size=batch_size,
            key_to_shuffle=shuffle_keys,
            perm=perm if shuffle_keys is not None else None,
            shuffle_y=False,
        )
        loader = loader_fn()

        preds, trues = [], []
        model.eval()
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                preds.append(model(xb).cpu())
                trues.append(yb.cpu())

        pred_np = torch.cat(preds).numpy()
        true_np = torch.cat(trues).numpy()

        if target_space == "processed":
            pass
        elif target_space == "log":
            pred_np = pred_np * stds + means
            true_np = true_np * stds + means
        elif target_space == "physical":
            pred_np = pred_np * stds + means
            true_np = true_np * stds + means
            pred_np[:, logflag] = np.exp(pred_np[:, logflag])
            true_np[:, logflag] = np.exp(true_np[:, logflag])
        else:
            raise ValueError("target_space must be 'processed', 'log', or 'physical'.")

        idx_arr = np.asarray(idx)
        obs1_arr = np.array(x_dict[obs1_key][idx_arr], copy=True)
        obs2_arr = np.array(x_dict[obs2_key][idx_arr], copy=True)

        if shuffle_keys is not None and perm is not None:
            if obs1_key in set(shuffle_keys):
                obs1_arr = obs1_arr[perm]
            if obs2_key in set(shuffle_keys):
                obs2_arr = obs2_arr[perm]

        result[cache_key] = {
            "pred": pred_np,
            "true": true_np,
            "obs1": obs1_arr,
            "obs2": obs2_arr,
            "obs1_key": obs1_key,
            "obs2_key": obs2_key,
        }

    return result[cache_key]


def plot_constraint_structure_2d(
    param,
    case_name,
    *,
    obs1_key=None,
    obs2_key=None,
    mode="aligned",
    keys_to_shuffle=None,
    perm=None,
    target_space="physical",
    x_dict=None,
    results=None,
    y_vector=None,
    idx=None,
    batch_size=None,
    param_labels=None,
    device=None,
    obs1_component=None,
    obs2_component=None,
    observable_reducer="mean",
    plot_kind="hexbin",      # "hexbin" or "scatter"
    gridsize=45,
    alpha=0.35,
    point_size=12,
    residual_center=0.0,
    residual_quantile_clip=0.98,
    cmap_value="viridis",
    cmap_residual="coolwarm",
    save_path=None,
):
    """
    2D constraint-structure view for one case and one target parameter.

    Panels:
      1) true parameter value
      2) predicted parameter value
      3) residual = predicted - true

    x-axis : obs1
    y-axis : obs2
    color  : chosen quantity
    """
    if results is None: results = all_results
    if x_dict is None: x_dict = x_normalized_dict
    if y_vector is None: y_vector = y
    if idx is None: idx = _eval_idx()
    if batch_size is None: batch_size = _cfg["batch_size"]
    if device is None: device = _cfg["device"]
    if obs1_key is None: obs1_key = observable_1
    if obs2_key is None: obs2_key = observable_2
    default_labels = param_labels or _cfg.get("param_names") or [f"θ{i}" for i in range(output_dim)]
    label_to_idx = {label: i for i, label in enumerate(default_labels)}

    if isinstance(param, int):
        param_idx = param
    elif isinstance(param, str):
        if param in label_to_idx:
            param_idx = label_to_idx[param]
        elif param.startswith("θ") and param[1:].isdigit():
            param_idx = int(param[1:])
        elif param.isdigit():
            param_idx = int(param)
        else:
            raise ValueError(f"Cannot interpret parameter identifier: {param}")
    else:
        raise ValueError("param must be an int or a string like 'θ3'.")

    if not (0 <= param_idx < output_dim):
        raise ValueError(f"Parameter index {param_idx} out of range.")

    param_label = default_labels[param_idx]

    data = _collect_constraint_2d_case_arrays(
        case_name=case_name,
        mode=mode,
        obs1_key=obs1_key,
        obs2_key=obs2_key,
        keys_to_shuffle=keys_to_shuffle,
        perm=perm,
        target_space=target_space,
        results=results,
        x_dict=x_dict,
        y_vector=y_vector,
        idx=idx,
        batch_size=batch_size,
        device=device,
    )

    x_obs = _constraint_2d_reduce_observable(
        data["obs1"], component=obs1_component, reducer=observable_reducer
    )
    y_obs = _constraint_2d_reduce_observable(
        data["obs2"], component=obs2_component, reducer=observable_reducer
    )

    z_true = data["true"][:, param_idx]
    z_pred = data["pred"][:, param_idx]
    z_resid = z_pred - z_true

    resid_lim = np.quantile(np.abs(z_resid), residual_quantile_clip)
    resid_lim = max(float(resid_lim), 1e-12)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)

    panels = [
        ("True", z_true, cmap_value, None, None),
        ("Predicted", z_pred, cmap_value, None, None),
        ("Residual (pred - true)", z_resid, cmap_residual, -resid_lim + residual_center, resid_lim + residual_center),
    ]

    for ax, (title, cvals, cmap, vmin, vmax) in zip(axes, panels):
        if plot_kind == "scatter":
            mappable = ax.scatter(
                x_obs, y_obs,
                c=cvals,
                s=point_size,
                alpha=alpha,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                edgecolors="none",
            )
        elif plot_kind == "hexbin":
            mappable = ax.hexbin(
                x_obs, y_obs,
                C=cvals,
                reduce_C_function=np.mean,
                gridsize=gridsize,
                cmap=cmap,
                mincnt=1,
                vmin=vmin,
                vmax=vmax,
            )
        else:
            raise ValueError("plot_kind must be 'scatter' or 'hexbin'.")

        cb = fig.colorbar(mappable, ax=ax)
        cb.set_label(title)

        ax.set_xlabel(
            f"{data['obs1_key']}"
            + (f" [component {obs1_component}]" if obs1_component is not None else f" [{observable_reducer}]")
        )
        ax.set_ylabel(
            f"{data['obs2_key']}"
            + (f" [component {obs2_component}]" if obs2_component is not None else f" [{observable_reducer}]")
        )
        ax.set_title(title)
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        f"{case_name} | {param_label} | mode={mode} | target_space={target_space}",
        fontsize=13,
    )

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200)

    stats = {
        "case_name": case_name,
        "param_idx": int(param_idx),
        "param_label": param_label,
        "obs1_key": data["obs1_key"],
        "obs2_key": data["obs2_key"],
        "mode": mode,
        "target_space": target_space,
        "residual_std": float(np.std(z_resid)),
        "residual_mean": float(np.mean(z_resid)),
        "true_range": (float(np.min(z_true)), float(np.max(z_true))),
        "pred_range": (float(np.min(z_pred)), float(np.max(z_pred))),
    }

    return fig, stats


def plot_unordered_pair_prediction_map(
    param,
    *,
    obs1_key=None,
    obs2_key=None,
    case_name=None,
    n_pairs=5000,
    seed=0,
    pairs=None,
    cases="auto",
    space="processed",          # "processed", "log", "physical"
    value_kind="pred",          # "pred", "true_i", "true_j", "resid_i", "resid_j"
    obs1_component=None,
    obs2_component=None,
    observable_reducer="mean",  # used if component is None
    plot_kind="hexbin",         # "hexbin" or "scatter"
    gridsize=45,
    alpha=0.35,
    point_size=12,
    cmap="viridis",
    cmap_residual="coolwarm",
    residual_quantile_clip=0.98,
    results=None,
    x_dict=None,
    y_vector=None,
    idx=None,
    batch_size=None,
    param_labels=None,
    device=None,
    save_path=None,
):
    """
    Plot unordered pairs in 2D observable space:
      x = obs1 from sim i
      y = obs2 from sim j
      color = chosen parameter quantity for that pair

    value_kind:
      - "pred"    : model prediction for the pair
      - "true_i"  : truth tied to obs1 / sim i
      - "true_j"  : truth tied to obs2 / sim j
      - "resid_i" : pred - true_i
      - "resid_j" : pred - true_j
    """
    if results is None: results = all_results
    if x_dict is None: x_dict = x_normalized_dict
    if y_vector is None: y_vector = y
    if idx is None: idx = _eval_idx()
    if batch_size is None: batch_size = _cfg["batch_size"]
    if device is None: device = _cfg["device"]
    if obs1_key is None: obs1_key = observable_1
    if obs2_key is None: obs2_key = observable_2
    if not results:
        raise ValueError("`results` is empty; train models and populate all_results first.")

    default_labels = param_labels or _cfg.get("param_names") or [f"θ{i}" for i in range(output_dim)]
    label_to_idx = {label: i for i, label in enumerate(default_labels)}

    if isinstance(param, int):
        param_idx = param
    elif isinstance(param, str):
        if param in label_to_idx:
            param_idx = label_to_idx[param]
        elif param.startswith("θ") and param[1:].isdigit():
            param_idx = int(param[1:])
        elif param.isdigit():
            param_idx = int(param)
        else:
            raise ValueError(f"Cannot interpret parameter identifier: {param}")
    else:
        raise ValueError("param must be an int or a string like 'θ3'.")

    if not (0 <= param_idx < output_dim):
        raise ValueError(f"Parameter index {param_idx} out of range.")

    param_label = default_labels[param_idx]

    def _convert_space(arr):
        arr = np.array(arr, copy=True)
        if space == "processed":
            return arr
        arr = arr * stds + means
        if space == "log":
            return arr
        if space == "physical":
            arr[:, logflag] = np.exp(arr[:, logflag])
            return arr
        raise ValueError("space must be 'processed', 'log', or 'physical'.")

    def _reduce_observable(arr, *, component=None, reducer="mean"):
        arr = np.asarray(arr)
        if arr.ndim == 1:
            return arr
        if component is not None:
            component = int(component)
            if not (0 <= component < arr.shape[1]):
                raise ValueError(f"component={component} out of range for shape {arr.shape}")
            return arr[:, component]
        if reducer == "mean":
            return np.mean(arr, axis=1)
        if reducer == "median":
            return np.median(arr, axis=1)
        if reducer == "sum":
            return np.sum(arr, axis=1)
        raise ValueError("observable_reducer must be one of {'mean', 'median', 'sum'}")

    all_case_names = [r["case_name"] for r in results]
    ordered_all, split_idx = dual_clean_asym_order(all_case_names)
    ordered_core = ordered_all[:split_idx]

    if cases == "auto":
        final_cases = [c for c in ordered_core if c in all_case_names]
    else:
        wanted = list(cases)
        final_cases = [c for c in ordered_core if c in wanted]

    case_to_result = {r["case_name"]: r for r in results}
    final_cases = [c for c in final_cases if obs1_key in case_to_result[c]["selected_observables"]
                   and obs2_key in case_to_result[c]["selected_observables"]]

    if not final_cases:
        raise ValueError(f"No cases contain both obs keys: {obs1_key}, {obs2_key}.")

    if case_name is None:
        case_name = final_cases[0]
    if case_name not in case_to_result:
        raise ValueError(f"Unknown case_name: {case_name}")

    result = case_to_result[case_name]

    n_val = len(idx)
    if pairs is None:
        pairs = sample_unique_unordered_pairs(n_val=n_val, n_pairs=n_pairs, seed=seed)
    else:
        pairs = np.asarray(pairs, dtype=int)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must be shape [n_pairs, 2].")
        if np.any(pairs < 0) or np.any(pairs >= n_val):
            raise ValueError("pairs contain indices outside [0, n_val).")
        pairs = np.sort(pairs, axis=1)
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
        if pairs.shape[0] == 0:
            raise ValueError("No valid pairs after removing i==j.")

    cache_key = (
        "_unordered_pair_cache_v2",
        case_name,
        obs1_key,
        obs2_key,
        space,
        tuple(map(int, pairs.ravel()))
    )

    if cache_key not in result:
        model = result["model"].to(device)

        loader = make_pair_val_loader_fn(
            selected_observables=result["selected_observables"],
            x_dict=x_dict,
            y_vector=y_vector,
            idx=idx,
            batch_size=batch_size,
            obs1_key=obs1_key,
            obs2_key=obs2_key,
            pairs=pairs,
        )

        preds = []
        ycats = []
        model.eval()
        with torch.no_grad():
            for xb, ybcat in loader:
                xb = xb.to(device)
                preds.append(model(xb).cpu())
                ycats.append(ybcat.cpu())

        pred_np = torch.cat(preds).numpy()
        ycat_np = torch.cat(ycats).numpy()
        y_i_np = ycat_np[:, :output_dim]
        y_j_np = ycat_np[:, output_dim:]

        pred_np = _convert_space(pred_np)
        y_i_np = _convert_space(y_i_np)
        y_j_np = _convert_space(y_j_np)

        result[cache_key] = (pred_np, y_i_np, y_j_np)

    pred_np, y_i_np, y_j_np = result[cache_key]

    idx = np.asarray(idx)
    pairs = np.asarray(pairs, dtype=int)
    i_rows = pairs[:, 0]
    j_rows = pairs[:, 1]
    i_idx = idx[i_rows]
    j_idx = idx[j_rows]

    obs1_vals = _reduce_observable(
        x_dict[obs1_key][i_idx],
        component=obs1_component,
        reducer=observable_reducer,
    )
    obs2_vals = _reduce_observable(
        x_dict[obs2_key][j_idx],
        component=obs2_component,
        reducer=observable_reducer,
    )

    pred_vals = pred_np[:, param_idx]
    true_i_vals = y_i_np[:, param_idx]
    true_j_vals = y_j_np[:, param_idx]

    if value_kind == "pred":
        color_vals = pred_vals
        title_suffix = "Predicted value"
        cmap_use = cmap
        vmin = vmax = None
    elif value_kind == "true_i":
        color_vals = true_i_vals
        title_suffix = f"True value from {obs1_key}"
        cmap_use = cmap
        vmin = vmax = None
    elif value_kind == "true_j":
        color_vals = true_j_vals
        title_suffix = f"True value from {obs2_key}"
        cmap_use = cmap
        vmin = vmax = None
    elif value_kind == "resid_i":
        color_vals = pred_vals - true_i_vals
        title_suffix = f"Residual vs {obs1_key} truth"
        cmap_use = cmap_residual
        lim = np.quantile(np.abs(color_vals), residual_quantile_clip)
        vmin, vmax = -lim, lim
    elif value_kind == "resid_j":
        color_vals = pred_vals - true_j_vals
        title_suffix = f"Residual vs {obs2_key} truth"
        cmap_use = cmap_residual
        lim = np.quantile(np.abs(color_vals), residual_quantile_clip)
        vmin, vmax = -lim, lim
    else:
        raise ValueError("value_kind must be one of {'pred', 'true_i', 'true_j', 'resid_i', 'resid_j'}")

    fig, ax = plt.subplots(figsize=(6.8, 5.4))

    if plot_kind == "scatter":
        mappable = ax.scatter(
            obs1_vals,
            obs2_vals,
            c=color_vals,
            s=point_size,
            alpha=alpha,
            cmap=cmap_use,
            vmin=vmin,
            vmax=vmax,
            edgecolors="none",
        )
    elif plot_kind == "hexbin":
        mappable = ax.hexbin(
            obs1_vals,
            obs2_vals,
            C=color_vals,
            reduce_C_function=np.mean,
            gridsize=gridsize,
            mincnt=1,
            cmap=cmap_use,
            vmin=vmin,
            vmax=vmax,
        )
    else:
        raise ValueError("plot_kind must be 'scatter' or 'hexbin'.")

    cbar = fig.colorbar(mappable, ax=ax)
    cbar.set_label(title_suffix)

    obs1_label = f"{obs1_key}" + (f" [component {obs1_component}]" if obs1_component is not None else f" [{observable_reducer}]")
    obs2_label = f"{obs2_key}" + (f" [component {obs2_component}]" if obs2_component is not None else f" [{observable_reducer}]")

    ax.set_xlabel(obs1_label)
    ax.set_ylabel(obs2_label)
    ax.set_title(f"{case_name} | {param_label} | {title_suffix}\nspace={space}, pairs={pairs.shape[0]}")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200)

    stats = {
        "case_name": case_name,
        "param_idx": int(param_idx),
        "param_label": param_label,
        "obs1_key": obs1_key,
        "obs2_key": obs2_key,
        "space": space,
        "value_kind": value_kind,
        "n_pairs": int(pairs.shape[0]),
        "color_mean": float(np.mean(color_vals)),
        "color_std": float(np.std(color_vals)),
    }

    return fig, stats


def plot_predictions_vs_true_by_noise(parameters,
                                       *,
                                       mode="aligned",       # "aligned", "obs1_vs_truth", "obs2_vs_truth"
                                       cases="auto",         # "auto" -> ordered core cases; or explicit list
                                       results=None,
                                       x_dict=None,
                                       y_vector=None,
                                       idx=None,
                                       batch_size=None,
                                       param_labels=None,
                                       perm=None,
                                       marker_size=18,
                                       save_dir=None,
                                       device=None):
    """
    One figure per parameter, one row of prediction-vs-truth scatters per
    noise case, laid out left-to-right in dual_clean_asym_order's core order:
    clean_obs1 -> (obs1 clean, obs2 ramping up) -> both clean ->
    (obs2 clean, obs1 ramping up) -> clean_obs2. Scanning a row left to right
    shows how a single parameter's predictions degrade/recover as each
    observable's noise level changes.
    """
    if results is None: results = all_results
    if x_dict is None: x_dict = x_normalized_dict
    if y_vector is None: y_vector = y
    if idx is None: idx = _eval_idx()
    if batch_size is None: batch_size = _cfg["batch_size"]
    if device is None: device = _cfg["device"]
    if not results:
        raise ValueError("`results` is empty; train models and populate all_results first.")

    def _to_list(v):
        if isinstance(v, str) or not hasattr(v, "__iter__"):
            return [v]
        return list(v)

    if isinstance(parameters, str) and parameters.lower() == "all":
        param_list = list(range(output_dim))
    else:
        param_list = _to_list(parameters)
        if not param_list:
            raise ValueError("Provide at least one parameter (index or label).")

    default_labels = param_labels or _cfg.get("param_names") or [f"θ{i}" for i in range(output_dim)]
    label_to_idx = {label: i for i, label in enumerate(default_labels)}

    def _resolve_param(p):
        if isinstance(p, int):
            if not 0 <= p < output_dim:
                raise ValueError(f"Parameter index {p} out of range (0..{output_dim-1}).")
            return p
        if isinstance(p, str):
            if p in label_to_idx:
                return label_to_idx[p]
            if p.startswith("θ") and p[1:].isdigit():
                i = int(p[1:])
                if not 0 <= i < output_dim:
                    raise ValueError(f"Parameter index {i} from {p} out of range.")
                return i
            if p.isdigit():
                i = int(p)
                if not 0 <= i < output_dim:
                    raise ValueError(f"Parameter index {i} from string out of range.")
                return i
        raise ValueError(f"Cannot interpret parameter identifier: {p}")

    param_indices = [_resolve_param(p) for p in param_list]
    param_labels_resolved = [default_labels[i] for i in param_indices]

    # Order cases via the same clean-obs1 -> ... -> clean-obs2 path used elsewhere
    case_to_result = {r["case_name"]: r for r in results}
    all_case_names = [r["case_name"] for r in results]
    ordered_all, split_idx = dual_clean_asym_order(all_case_names)
    ordered_core = ordered_all[:split_idx]

    if cases == "auto":
        case_list = [c for c in ordered_core if c in case_to_result]
    else:
        wanted = set(_to_list(cases))
        case_list = [c for c in ordered_core if c in wanted]
    if not case_list:
        raise ValueError("No cases to plot after ordering/selection.")

    if mode in ("obs1_vs_truth", "obs2_vs_truth"):
        case_list = [c for c in case_list if len(case_to_result[c]["selected_observables"]) >= 2]
        if not case_list:
            raise ValueError("No multi-observable cases available for this mode after filtering.")

    if perm is None:
        perm = _cfg.get("perm")
        if perm is None or len(perm) != len(idx):
            perm = np.random.permutation(len(idx))
    perm = np.asarray(perm)

    def _collect_predictions(result):
        return get_case_predictions(result, mode=mode, perm=perm)

    mode_title = {"aligned": "Aligned",
                  "obs1_vs_truth": f"{observable_2} shuffled, {observable_1} truths",
                  "obs2_vs_truth": f"{observable_2} shuffled, {observable_2} truths"}[mode]

    figs = {}
    for p_idx, p_label in zip(param_indices, param_labels_resolved):
        n_cols = len(case_list)
        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4.2), squeeze=False)
        axes = axes[0]

        for c, case_name in enumerate(case_list):
            preds, trues = _collect_predictions(case_to_result[case_name])
            ax = axes[c]
            y_true = trues[:, p_idx]
            y_pred = preds[:, p_idx]
            ax.scatter(y_true, y_pred, s=marker_size, alpha=0.6, edgecolor="none")

            lo = float(min(y_true.min(), y_pred.min()))
            hi = float(max(y_true.max(), y_pred.max()))
            ax.plot([lo, hi], [lo, hi], "r--", lw=1.5)

            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            ax.set_title(f"{case_name}\nR²={r2:.3f}, RMSE={rmse:.3f}")
            ax.set_xlabel("True")
            if c == 0:
                ax.set_ylabel("Predicted")
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"{mode_title} — {p_label}: prediction vs truth across noise cases", y=1.03)
        fig.tight_layout()
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, f"{p_label}_by_noise.png"), dpi=200, bbox_inches="tight")
        figs[p_label] = fig

    return figs


def plot_bias_progression_overlay(param,
                                   *,
                                   case_sequence=None,
                                   obs_pair=None,
                                   results=None,
                                   param_labels=None,
                                   n_bins=15,
                                   min_per_bin=5,
                                   reference_case=None,
                                   show_scatter_for=None,
                                   scatter_alpha=0.15,
                                   scatter_size=6,
                                   figsize=(9, 7),
                                   perm=None,
                                   residualize_against=None,
                                   save_path=None):
    """
    Overlay per-case (pred - true) vs true bias curves for one parameter, with
    a shared x-axis and a marginal histogram of true values on top. Cases are
    ordered along the dual_clean_asym_order sequence for a chosen pair of
    observables (clean_left -> asym_left -> both_clean -> asym_right ->
    clean_right); color encodes position along that ramp.

    Generalizes to any two observables in `results`. Case sequence resolution:
      1. If `case_sequence` is given, use it as-is.
      2. Else if `obs_pair=(obsA, obsB)` is given, filter results to combo
         cases involving exactly that pair (plus their single-obs references)
         and run dual_clean_asym_order on that filtered list.
      3. Else auto-detect: if exactly one distinct observable pair exists
         across combo cases, use it. Otherwise raise, listing available pairs.

    `reference_case` defaults to the "both clean" combo (noise1==0 and
    noise2==0) in the resolved sequence, auto-detected via parse_case_name
    rather than hardcoded.

    Only aligned-mode predictions are pulled (via get_case_predictions), and
    we assert row-for-row truth agreement across cases so the "true value on
    the x-axis" is unambiguous.

    `residualize_against`:
      - None (default): plot raw (pred - true) per case. Every parameter shows
        a universal negative slope because MSE minimization shrinks predictions
        toward the prior mean; the slope steepness is diagnostic of how weakly
        constrained the parameter is.
      - "reference": non-parametric per-simulation subtraction of the
        reference case's residuals. Because every case shares idx_val, the
        truth cancels and each case's plotted quantity becomes
            delta_j = pred_case_j - pred_ref_j
        binned by true value. No linear/spline fit is assumed on the reference
        curve. Requires a reference_case (defaults to the both-clean combo).
        Reference curve is exactly zero for every simulation, so its line is
        flat at 0 with zero SE by construction; other curves show, per bin,
        how the prediction shifts when the training regime changes.

    Returns (fig, stats) with stats = {case_sequence, obs_pair,
    reference_case, bin_edges, bin_centers, binned per case,
    residualize_against}.
    """
    if results is None: results = all_results
    if results is None:
        results = all_results

    # ---- parameter index (same resolver shape as plot_predictions_vs_true_by_noise)
    default_labels = param_labels or _cfg.get("param_names") or [f"θ{i}" for i in range(output_dim)]
    label_to_idx = {label: i for i, label in enumerate(default_labels)}
    if isinstance(param, int):
        if not 0 <= param < output_dim:
            raise ValueError(f"Parameter index {param} out of range (0..{output_dim-1}).")
        p_idx = param
    elif isinstance(param, str):
        if param in label_to_idx:
            p_idx = label_to_idx[param]
        elif param.startswith("θ") and param[1:].isdigit():
            p_idx = int(param[1:])
        elif param.isdigit():
            p_idx = int(param)
        else:
            raise ValueError(f"Cannot interpret parameter identifier: {param!r}")
        if not 0 <= p_idx < output_dim:
            raise ValueError(f"Parameter index {p_idx} out of range.")
    else:
        raise ValueError(f"Cannot interpret parameter identifier: {param!r}")
    p_label = default_labels[p_idx]

    # ---- resolve case sequence and obs_pair
    case_to_result = {r["case_name"]: r for r in results}

    if case_sequence is None:
        combo_pairs = set()
        for c in case_to_result:
            info = parse_case_name(c)
            if info["kind"] == "combo":
                combo_pairs.add(frozenset((info["obs1"], info["obs2"])))

        if obs_pair is None:
            if not combo_pairs:
                raise ValueError("No combo cases in results; cannot auto-detect an observable pair.")
            if len(combo_pairs) > 1:
                pretty = sorted(sorted(list(p)) for p in combo_pairs)
                raise ValueError(
                    f"Multiple observable pairs found in results: {pretty}. "
                    f"Pass obs_pair=(obsA, obsB) to disambiguate."
                )
            obs_pair = tuple(sorted(next(iter(combo_pairs))))
        else:
            wanted = frozenset(obs_pair)
            if wanted not in combo_pairs:
                pretty = sorted(sorted(list(p)) for p in combo_pairs)
                raise ValueError(
                    f"obs_pair={tuple(obs_pair)} not found in combo cases. Available pairs: {pretty}."
                )
            obs_pair = tuple(obs_pair)

        obs_set = set(obs_pair)
        filtered = []
        for c in case_to_result:
            info = parse_case_name(c)
            if info["kind"] == "combo" and {info["obs1"], info["obs2"]} == obs_set:
                filtered.append(c)
            elif info["kind"] == "single" and info["obs"] in obs_set:
                filtered.append(c)

        ordered_all, split_idx = dual_clean_asym_order(filtered)
        case_sequence = ordered_all[:split_idx]
    else:
        case_sequence = list(case_sequence)
        if obs_pair is None:
            for c in case_sequence:
                info = parse_case_name(c)
                if info["kind"] == "combo":
                    obs_pair = (info["obs1"], info["obs2"])
                    break

    if not case_sequence:
        raise ValueError("Resolved case_sequence is empty -- nothing to plot.")
    missing = [c for c in case_sequence if c not in case_to_result]
    if missing:
        raise ValueError(f"Cases not found in results: {missing}")

    # ---- reference case (both-clean combo)
    if reference_case is None:
        for c in case_sequence:
            info = parse_case_name(c)
            if info["kind"] == "combo" and info.get("noise1") == 0.0 and info.get("noise2") == 0.0:
                reference_case = c
                break
    if reference_case is not None and reference_case not in case_sequence:
        raise ValueError(f"reference_case {reference_case!r} not in resolved case_sequence.")

    # ---- collect predictions, verify aligned-mode truth agreement across cases
    per_case = {}
    ref_true = None
    for c in case_sequence:
        preds, trues = get_case_predictions(case_to_result[c], mode="aligned", perm=perm)
        y_pred = preds[:, p_idx]
        y_true = trues[:, p_idx]
        if ref_true is None:
            ref_true = y_true
        else:
            if not np.allclose(y_true, ref_true, rtol=1e-8, atol=1e-8):
                raise ValueError(
                    f"Truths for case {c!r} do not match the first case's truths row-for-row. "
                    f"get_case_predictions(mode='aligned') should share idx_val across cases."
                )
        per_case[c] = (y_pred, y_true, y_pred - y_true)

    # ---- optional residualization: per-simulation subtraction of the reference
    # case's residuals. No fit -- we exploit that every case shares idx_val, so
    # true cancels and each case's plotted quantity becomes pred_case - pred_ref
    # binned by true value. The reference case's plotted quantity is exactly 0.
    if residualize_against == "reference":
        if reference_case is None:
            raise ValueError(
                "residualize_against='reference' requires a reference_case, but none "
                "is set (no both-clean combo found). Pass reference_case explicitly."
            )
        ref_resid = per_case[reference_case][2]
        for c, (yp, yt, r) in per_case.items():
            per_case[c] = (yp, yt, r - ref_resid)
    elif residualize_against is not None:
        raise ValueError(
            f"residualize_against={residualize_against!r} not recognized. "
            f"Use None (default) or 'reference'."
        )

    # ---- shared bin edges from the reference truths
    lo, hi = float(ref_true.min()), float(ref_true.max())
    edges = np.linspace(lo, hi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    def _bin_resid(y_true, resid):
        b_idx = np.clip(np.digitize(y_true, edges) - 1, 0, n_bins - 1)
        means = np.full(n_bins, np.nan)
        se    = np.full(n_bins, np.nan)
        for b in range(n_bins):
            m = b_idx == b
            n = int(m.sum())
            if n >= min_per_bin:
                r = resid[m]
                means[b] = r.mean()
                se[b]    = r.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
        return means, se

    binned = {c: _bin_resid(v[1], v[2]) for c, v in per_case.items()}

    # ---- figure: stacked axes, shared x
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 4], hspace=0.06)
    ax_hist = fig.add_subplot(gs[0])
    ax_main = fig.add_subplot(gs[1], sharex=ax_hist)

    ax_hist.hist(ref_true, bins=edges, color="#B4B2A9", edgecolor="none")
    ax_hist.set_yticks([])
    for side in ("top", "right", "left"):
        ax_hist.spines[side].set_visible(False)
    ax_hist.tick_params(axis="x", labelbottom=False)

    cmap = plt.get_cmap("coolwarm")
    n = len(case_sequence)
    positions = np.linspace(0, 1, n) if n > 1 else np.array([0.5])
    colors = {c: cmap(p) for c, p in zip(case_sequence, positions)}

    if show_scatter_for:
        for c in show_scatter_for:
            if c not in per_case:
                continue
            y_true = per_case[c][1]; resid = per_case[c][2]
            ax_main.scatter(y_true, resid, s=scatter_size, alpha=scatter_alpha,
                            color=colors[c], edgecolor="none", zorder=1)

    for c in case_sequence:
        if c == reference_case:
            continue
        means, se = binned[c]
        m = ~np.isnan(means)
        ax_main.plot(centers[m], means[m], "-", color=colors[c], lw=1.6, label=c, zorder=2)
        ax_main.fill_between(centers[m], (means - se)[m], (means + se)[m],
                             color=colors[c], alpha=0.15, linewidth=0, zorder=2)

    if reference_case is not None:
        means, se = binned[reference_case]
        m = ~np.isnan(means)
        ax_main.plot(centers[m], means[m], "-", color="black", lw=2.4,
                     label=f"{reference_case} (reference)", zorder=3)
        ax_main.fill_between(centers[m], (means - se)[m], (means + se)[m],
                             color="black", alpha=0.18, linewidth=0, zorder=3)

    ax_main.axhline(0.0, color="k", ls="--", lw=0.8, alpha=0.6)
    ax_main.set_xlabel(f"True {p_label} (physical space)")
    if residualize_against == "reference":
        ax_main.set_ylabel(f"Pred_case − Pred_ref   ({p_label})")
    else:
        ax_main.set_ylabel(f"Predicted − True   ({p_label})")
    ax_main.grid(alpha=0.25)
    ax_main.legend(fontsize=8, loc="best", framealpha=0.9)

    # small annotation naming what the plot means
    if residualize_against == "reference":
        annot = (f"per-simulation subtraction of {reference_case!r}\n"
                 f"(no fit; truth cancels, so plotted = pred_case − pred_ref)\n"
                 f"reference curve is exactly 0 by construction")
    else:
        annot = ("negative slope = MSE shrinkage baseline\n"
                 "(steepness ~ inverse constraining power)\n"
                 "pass residualize_against='reference' to subtract it")
    ax_main.text(0.02, 0.98, annot, transform=ax_main.transAxes,
                 va="top", ha="left", fontsize=8, color="#3C3489",
                 bbox=dict(facecolor="white", edgecolor="#B4B2A9",
                          alpha=0.9, boxstyle="round,pad=0.4"))

    pair_str = f"{obs_pair[0]} ↔ {obs_pair[1]}" if obs_pair else "unknown pair"
    ax_hist.set_title(
        f"Bias progression for {p_label}\n"
        f"observables: {pair_str}   |   {len(case_sequence)} cases in dual-clean-asym order",
        fontsize=11
    )

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    stats = {
        "case_sequence": case_sequence,
        "obs_pair": obs_pair,
        "reference_case": reference_case,
        "n_bins": n_bins,
        "bin_edges": edges,
        "bin_centers": centers,
        "binned": {c: {"mean": m, "se": s} for c, (m, s) in binned.items()},
        "residualize_against": residualize_against,
    }
    return fig, stats


def plot_directional_pull_by_true(param,
                                    *,
                                    case_sequence=None,
                                    obs_pair=None,
                                    results=None,
                                    param_labels=None,
                                    n_bins=15,
                                    min_per_bin=5,
                                    figsize=(9, 6.5),
                                    perm=None,
                                    save_path=None):
    """
    Per-simulation directional pull vs bin-conditional accuracy, aggregated by
    true parameter value.

    For each val simulation j across the ordered case sequence, computes:
      - pull_j     = slope of pred_case_j regressed on sequence position.
                     SIGN = direction (positive: predictions increased along the
                     sequence from clean_left toward clean_right; negative: the
                     opposite). MAGNITUDE = strength of that directional pull.
      - error_j    = median over cases of |pred_case_j - true_j|. Unsigned
                     accuracy floor for that sim.

    Then bins both by the sim's true θ. In each bin plots:
      - solid line: mean(pull_j) ± SE ribbon.  Above 0 = pulled up in this bin;
        below 0 = pulled down.
      - grey band: ±mean(error_j) around 0.  This is the accuracy envelope —
        the pull needs to exit this band to be credible; a pull line inside the
        band means the shift is smaller than typical prediction error for sims
        in that bin.
      - top marginal: histogram of true θ using the same bin edges.

    Reads: does the direction of pull differ between edge bins and middle bins
    (does the parameter range change how observables disagree)? And where does
    that pull exceed the model's typical error (credible) vs stay inside it
    (drowned in noise)?

    Sequence position is uniform (0, 1, ..., n-1) along dual_clean_asym_order
    -- the case sequence is a conceptual ramp, not a numeric axis, so we don't
    try to space cases by noise level.

    Generalizes to any observable pair in results (same resolution logic as
    plot_bias_progression_overlay). Only aligned-mode predictions are pulled;
    truth agreement across cases is verified so the "true θ on x-axis" is
    unambiguous.

    Returns (fig, stats) with per-sim pull_j / error_j arrays and per-bin
    aggregates.
    """
    if results is None: results = all_results
    if results is None:
        results = all_results

    # ---- parameter index (same resolver as sibling function)
    default_labels = param_labels or _cfg.get("param_names") or [f"θ{i}" for i in range(output_dim)]
    label_to_idx = {label: i for i, label in enumerate(default_labels)}
    if isinstance(param, int):
        if not 0 <= param < output_dim:
            raise ValueError(f"Parameter index {param} out of range (0..{output_dim-1}).")
        p_idx = param
    elif isinstance(param, str):
        if param in label_to_idx:
            p_idx = label_to_idx[param]
        elif param.startswith("θ") and param[1:].isdigit():
            p_idx = int(param[1:])
        elif param.isdigit():
            p_idx = int(param)
        else:
            raise ValueError(f"Cannot interpret parameter identifier: {param!r}")
        if not 0 <= p_idx < output_dim:
            raise ValueError(f"Parameter index {p_idx} out of range.")
    else:
        raise ValueError(f"Cannot interpret parameter identifier: {param!r}")
    p_label = default_labels[p_idx]

    # ---- resolve case sequence + obs_pair (mirrors plot_bias_progression_overlay)
    case_to_result = {r["case_name"]: r for r in results}
    if case_sequence is None:
        combo_pairs = set()
        for c in case_to_result:
            info = parse_case_name(c)
            if info["kind"] == "combo":
                combo_pairs.add(frozenset((info["obs1"], info["obs2"])))
        if obs_pair is None:
            if not combo_pairs:
                raise ValueError("No combo cases in results; cannot auto-detect an observable pair.")
            if len(combo_pairs) > 1:
                pretty = sorted(sorted(list(p)) for p in combo_pairs)
                raise ValueError(
                    f"Multiple observable pairs found in results: {pretty}. "
                    f"Pass obs_pair=(obsA, obsB) to disambiguate."
                )
            obs_pair = tuple(sorted(next(iter(combo_pairs))))
        else:
            wanted = frozenset(obs_pair)
            if wanted not in combo_pairs:
                pretty = sorted(sorted(list(p)) for p in combo_pairs)
                raise ValueError(
                    f"obs_pair={tuple(obs_pair)} not found in combo cases. Available pairs: {pretty}."
                )
            obs_pair = tuple(obs_pair)
        obs_set = set(obs_pair)
        filtered = []
        for c in case_to_result:
            info = parse_case_name(c)
            if info["kind"] == "combo" and {info["obs1"], info["obs2"]} == obs_set:
                filtered.append(c)
            elif info["kind"] == "single" and info["obs"] in obs_set:
                filtered.append(c)
        ordered_all, split_idx = dual_clean_asym_order(filtered)
        case_sequence = ordered_all[:split_idx]
    else:
        case_sequence = list(case_sequence)
        if obs_pair is None:
            for c in case_sequence:
                info = parse_case_name(c)
                if info["kind"] == "combo":
                    obs_pair = (info["obs1"], info["obs2"])
                    break

    if not case_sequence:
        raise ValueError("Resolved case_sequence is empty -- nothing to plot.")
    if len(case_sequence) < 2:
        raise ValueError(f"Need at least 2 cases to compute a directional slope; got {case_sequence}.")
    missing = [c for c in case_sequence if c not in case_to_result]
    if missing:
        raise ValueError(f"Cases not found in results: {missing}")

    # ---- collect predictions, verify aligned-mode truth agreement
    n_cases = len(case_sequence)
    preds_by_case = np.zeros((n_cases, 0))       # placeholder; sized on first case
    true_ref = None
    for k, c in enumerate(case_sequence):
        preds, trues = get_case_predictions(case_to_result[c], mode="aligned", perm=perm)
        yp = preds[:, p_idx]
        yt = trues[:, p_idx]
        if true_ref is None:
            true_ref = yt
            preds_by_case = np.zeros((n_cases, len(yp)))
        else:
            if not np.allclose(yt, true_ref, rtol=1e-8, atol=1e-8):
                raise ValueError(
                    f"Truths for case {c!r} do not match the first case's truths row-for-row. "
                    f"get_case_predictions(mode='aligned') should share idx_val across cases."
                )
        preds_by_case[k] = yp

    n_sims = preds_by_case.shape[1]
    # sequence position: uniform 0..n_cases-1
    pos = np.arange(n_cases, dtype=float)
    pos_c = pos - pos.mean()                     # centered for numeric stability
    denom = float((pos_c * pos_c).sum())
    if denom == 0.0:
        raise ValueError("Sequence has zero variance in position -- cannot compute slope.")

    # per-sim signed pull: least-squares slope of pred vs sequence position
    pred_c = preds_by_case - preds_by_case.mean(axis=0, keepdims=True)
    pull = (pos_c[:, None] * pred_c).sum(axis=0) / denom      # shape (n_sims,)
    # per-sim accuracy: median |pred - true| across cases
    abs_err = np.abs(preds_by_case - true_ref[None, :])
    error = np.median(abs_err, axis=0)                        # shape (n_sims,)

    # ---- bin by true value
    lo, hi = float(true_ref.min()), float(true_ref.max())
    edges = np.linspace(lo, hi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = np.clip(np.digitize(true_ref, edges) - 1, 0, n_bins - 1)

    mean_pull = np.full(n_bins, np.nan)
    se_pull   = np.full(n_bins, np.nan)
    mean_err  = np.full(n_bins, np.nan)
    counts    = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        m = bin_idx == b
        n = int(m.sum())
        counts[b] = n
        if n >= min_per_bin:
            pj = pull[m]; ej = error[m]
            mean_pull[b] = pj.mean()
            se_pull[b]   = pj.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
            mean_err[b]  = ej.mean()

    # ---- figure
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 4], hspace=0.06)
    ax_hist = fig.add_subplot(gs[0])
    ax_main = fig.add_subplot(gs[1], sharex=ax_hist)

    ax_hist.hist(true_ref, bins=edges, color="#B4B2A9", edgecolor="none")
    ax_hist.set_yticks([])
    for side in ("top", "right", "left"):
        ax_hist.spines[side].set_visible(False)
    ax_hist.tick_params(axis="x", labelbottom=False)

    # accuracy envelope: grey ±error band around zero
    good = ~np.isnan(mean_err)
    ax_main.fill_between(centers[good], -mean_err[good], mean_err[good],
                         color="#B4B2A9", alpha=0.35, linewidth=0,
                         label="±median |pred − true| (accuracy floor)")

    ax_main.axhline(0.0, color="k", ls="--", lw=0.8, alpha=0.6)

    # pull line + SE ribbon
    good_p = ~np.isnan(mean_pull)
    ax_main.fill_between(centers[good_p],
                         (mean_pull - se_pull)[good_p], (mean_pull + se_pull)[good_p],
                         color="#3C3489", alpha=0.25, linewidth=0)
    ax_main.plot(centers[good_p], mean_pull[good_p], "-", color="#3C3489",
                 lw=2.2, marker="o", ms=4, label="mean directional pull")

    # mark bins where |pull| exceeds accuracy floor (credible directional pull)
    credible = np.abs(mean_pull) > mean_err
    if credible.any():
        ax_main.plot(centers[credible], mean_pull[credible], "o",
                     mfc="none", mec="#791F1F", mew=1.6, ms=10, zorder=5,
                     label="|pull| > accuracy floor")

    ax_main.set_xlabel(f"True {p_label} (physical space)")
    ax_main.set_ylabel(f"Directional pull   ({p_label} / case-step)")
    ax_main.grid(alpha=0.25)
    ax_main.legend(fontsize=8, loc="best", framealpha=0.9)

    pair_str = f"{obs_pair[0]} ↔ {obs_pair[1]}" if obs_pair else "unknown pair"
    seq_str = f"{case_sequence[0]} → {case_sequence[-1]}"
    ax_hist.set_title(
        f"Directional pull for {p_label}\n"
        f"observables: {pair_str}   |   sequence: {seq_str}   ({n_cases} cases)",
        fontsize=11
    )

    annot = ("sign of pull = direction predictions moved along the sequence\n"
             "(positive: later cases predict higher than earlier)\n"
             "|pull| < grey band → smaller than typical prediction error")
    ax_main.text(0.02, 0.98, annot, transform=ax_main.transAxes,
                 va="top", ha="left", fontsize=8, color="#3C3489",
                 bbox=dict(facecolor="white", edgecolor="#B4B2A9",
                          alpha=0.9, boxstyle="round,pad=0.4"))

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    stats = {
        "case_sequence": case_sequence,
        "obs_pair": obs_pair,
        "sequence_position": pos,
        "per_sim_pull": pull,
        "per_sim_error": error,
        "per_sim_true": true_ref,
        "bin_edges": edges,
        "bin_centers": centers,
        "bin_counts": counts,
        "bin_mean_pull": mean_pull,
        "bin_se_pull": se_pull,
        "bin_mean_error": mean_err,
    }
    return fig, stats


def plot_prediction_attractor_map(param,
                                    *,
                                    case_sequence=None,
                                    obs_pair=None,
                                    results=None,
                                    param_labels=None,
                                    n_bins=15,
                                    min_per_bin=5,
                                    reference_case=None,
                                    weight_by=None,           # None | "aligned_r2"
                                    show_identity=True,
                                    show_prior_mean=True,
                                    show_se_band=True,
                                    figsize=(8.5, 8),
                                    perm=None,
                                    save_path=None):
    """
    One-panel condensation of the per-case prediction-vs-truth scatters:
    each case becomes a binned line of mean(pred) vs true, so N noise cases
    show as N curves in the same panel and you see how the pred/truth
    relationship evolves across the case sequence in one view.

    Reference lines drawn on the plot:
      - y = x (identity): predictions equal truth, no shrinkage. Model has
        perfect info about θ.
      - y = mean(true): predictions collapsed to the prior mean. Model has
        no info.  Every real case's line sits between these two extremes;
        vertical position at each true-θ tells you how much predictions are
        being pulled toward the mean at that part of the parameter range.

    Cases are colored along dual_clean_asym_order with coolwarm; the
    reference case (auto-detected as the both-clean combo, or passed
    explicitly) is drawn in bold black on top so it stands out as the
    "best-info" baseline every other case can be compared against.

    weight_by:
      - None (default): all case lines drawn at uniform alpha.
      - "aligned_r2": per-case line alpha scaled by its aligned R² --
        cases with poor R² fade out so you're not lulled by a confident-
        looking line from a case whose predictions are actually noise.

    Case sequence resolution mirrors plot_bias_progression_overlay:
    obs_pair auto-detect from single-pair results, or pass explicitly.
    Only aligned-mode predictions are pulled; truth agreement across
    cases is verified row-for-row.

    Returns (fig, stats) with per-case binned means/SEs and per-case
    aligned R² for downstream use.
    """
    if results is None: results = all_results
    if results is None:
        results = all_results

    # ---- parameter index (mirrors sibling functions)
    default_labels = param_labels or _cfg.get("param_names") or [f"θ{i}" for i in range(output_dim)]
    label_to_idx = {label: i for i, label in enumerate(default_labels)}
    if isinstance(param, int):
        if not 0 <= param < output_dim:
            raise ValueError(f"Parameter index {param} out of range (0..{output_dim-1}).")
        p_idx = param
    elif isinstance(param, str):
        if param in label_to_idx:
            p_idx = label_to_idx[param]
        elif param.startswith("θ") and param[1:].isdigit():
            p_idx = int(param[1:])
        elif param.isdigit():
            p_idx = int(param)
        else:
            raise ValueError(f"Cannot interpret parameter identifier: {param!r}")
        if not 0 <= p_idx < output_dim:
            raise ValueError(f"Parameter index {p_idx} out of range.")
    else:
        raise ValueError(f"Cannot interpret parameter identifier: {param!r}")
    p_label = default_labels[p_idx]

    # ---- resolve case sequence + obs_pair (mirrors sibling functions)
    case_to_result = {r["case_name"]: r for r in results}
    if case_sequence is None:
        combo_pairs = set()
        for c in case_to_result:
            info = parse_case_name(c)
            if info["kind"] == "combo":
                combo_pairs.add(frozenset((info["obs1"], info["obs2"])))
        if obs_pair is None:
            if not combo_pairs:
                raise ValueError("No combo cases in results; cannot auto-detect an observable pair.")
            if len(combo_pairs) > 1:
                pretty = sorted(sorted(list(p)) for p in combo_pairs)
                raise ValueError(
                    f"Multiple observable pairs found in results: {pretty}. "
                    f"Pass obs_pair=(obsA, obsB) to disambiguate."
                )
            obs_pair = tuple(sorted(next(iter(combo_pairs))))
        else:
            wanted = frozenset(obs_pair)
            if wanted not in combo_pairs:
                pretty = sorted(sorted(list(p)) for p in combo_pairs)
                raise ValueError(
                    f"obs_pair={tuple(obs_pair)} not found in combo cases. Available pairs: {pretty}."
                )
            obs_pair = tuple(obs_pair)
        obs_set = set(obs_pair)
        filtered = []
        for c in case_to_result:
            info = parse_case_name(c)
            if info["kind"] == "combo" and {info["obs1"], info["obs2"]} == obs_set:
                filtered.append(c)
            elif info["kind"] == "single" and info["obs"] in obs_set:
                filtered.append(c)
        ordered_all, split_idx = dual_clean_asym_order(filtered)
        case_sequence = ordered_all[:split_idx]
    else:
        case_sequence = list(case_sequence)
        if obs_pair is None:
            for c in case_sequence:
                info = parse_case_name(c)
                if info["kind"] == "combo":
                    obs_pair = (info["obs1"], info["obs2"])
                    break

    if not case_sequence:
        raise ValueError("Resolved case_sequence is empty -- nothing to plot.")
    missing = [c for c in case_sequence if c not in case_to_result]
    if missing:
        raise ValueError(f"Cases not found in results: {missing}")

    # ---- reference case
    if reference_case is None:
        for c in case_sequence:
            info = parse_case_name(c)
            if info["kind"] == "combo" and info.get("noise1") == 0.0 and info.get("noise2") == 0.0:
                reference_case = c
                break
    if reference_case is not None and reference_case not in case_sequence:
        raise ValueError(f"reference_case {reference_case!r} not in resolved case_sequence.")

    # ---- collect predictions, verify truth alignment
    per_case = {}
    true_ref = None
    for c in case_sequence:
        preds, trues = get_case_predictions(case_to_result[c], mode="aligned", perm=perm)
        yp = preds[:, p_idx]; yt = trues[:, p_idx]
        if true_ref is None:
            true_ref = yt
        else:
            if not np.allclose(yt, true_ref, rtol=1e-8, atol=1e-8):
                raise ValueError(
                    f"Truths for case {c!r} do not match the first case's truths row-for-row. "
                    f"get_case_predictions(mode='aligned') should share idx_val across cases."
                )
        per_case[c] = (yp, yt)

    # ---- per-case aligned R² (for optional weighting)
    aligned_r2 = {}
    tt = true_ref
    ss_tot = float(((tt - tt.mean())**2).sum())
    for c, (yp, _) in per_case.items():
        ss_res = float(((yp - tt)**2).sum())
        aligned_r2[c] = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # ---- shared bin edges from the true θ range
    lo, hi = float(true_ref.min()), float(true_ref.max())
    edges = np.linspace(lo, hi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = np.clip(np.digitize(true_ref, edges) - 1, 0, n_bins - 1)

    def _bin_mean_se(y):
        means = np.full(n_bins, np.nan)
        se    = np.full(n_bins, np.nan)
        for b in range(n_bins):
            m = bin_idx == b
            n = int(m.sum())
            if n >= min_per_bin:
                v = y[m]
                means[b] = v.mean()
                se[b]    = v.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
        return means, se

    binned = {c: _bin_mean_se(per_case[c][0]) for c in case_sequence}

    # ---- alpha weights per case
    n_cases = len(case_sequence)
    if weight_by == "aligned_r2":
        r2_vals = np.array([aligned_r2[c] for c in case_sequence])
        r2_min, r2_max = float(r2_vals.min()), float(r2_vals.max())
        if r2_max > r2_min:
            alphas = 0.35 + 0.55 * (r2_vals - r2_min) / (r2_max - r2_min)
        else:
            alphas = np.full(n_cases, 0.85)
    elif weight_by is None:
        alphas = np.full(n_cases, 0.85)
    else:
        raise ValueError(f"weight_by={weight_by!r} not recognized. Use None or 'aligned_r2'.")

    # ---- figure
    fig, ax = plt.subplots(figsize=figsize)

    prior_mean = float(true_ref.mean())

    # reference lines (drawn under everything)
    if show_identity:
        ax.plot([lo, hi], [lo, hi], "--", color="#444441", lw=1.2, alpha=0.7,
                label="y = x (identity: no shrinkage)", zorder=1)
    if show_prior_mean:
        ax.axhline(prior_mean, color="#5F5E5A", ls=":", lw=1.1, alpha=0.7,
                   label=f"y = mean(true θ) = {prior_mean:.3g} (full shrinkage)", zorder=1)

    # per-case binned lines, ordered along the sequence
    cmap = plt.get_cmap("coolwarm")
    positions = np.linspace(0, 1, n_cases) if n_cases > 1 else np.array([0.5])
    colors = {c: cmap(p) for c, p in zip(case_sequence, positions)}

    for c, alpha_c in zip(case_sequence, alphas):
        if c == reference_case:
            continue
        means, se = binned[c]
        m = ~np.isnan(means)
        r2 = aligned_r2[c]
        lab = f"{c}  (R²={r2:+.2f})"
        if show_se_band:
            ax.fill_between(centers[m], (means - se)[m], (means + se)[m],
                            color=colors[c], alpha=0.15 * (alpha_c / 0.85),
                            linewidth=0, zorder=2)
        ax.plot(centers[m], means[m], "-", color=colors[c], lw=1.7,
                alpha=alpha_c, label=lab, zorder=3)

    # reference case last, in bold black
    if reference_case is not None:
        means, se = binned[reference_case]
        m = ~np.isnan(means)
        r2 = aligned_r2[reference_case]
        lab = f"{reference_case}  (R²={r2:+.2f})   ← reference"
        if show_se_band:
            ax.fill_between(centers[m], (means - se)[m], (means + se)[m],
                            color="black", alpha=0.15, linewidth=0, zorder=4)
        ax.plot(centers[m], means[m], "-", color="black", lw=2.6,
                label=lab, zorder=5)

    ax.set_xlim(lo, hi)
    ax.set_xlabel(f"True {p_label} (physical space)")
    ax.set_ylabel(f"Mean predicted {p_label} per bin")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best", framealpha=0.9)

    pair_str = f"{obs_pair[0]} ↔ {obs_pair[1]}" if obs_pair else "unknown pair"
    ax.set_title(
        f"Attractor map for {p_label} — {n_cases} noise cases collapsed into one view\n"
        f"observables: {pair_str}   |   line color = case position in dual-clean-asym order",
        fontsize=11
    )

    annot_lines = ["closer to y=x → less shrinkage (more info)",
                   "closer to horizontal mean line → more shrinkage (less info)",
                   "per-case slope tells you how well each case recovers θ"]
    if weight_by == "aligned_r2":
        annot_lines.append("line alpha scaled by aligned R² (faded = noisy case)")
    ax.text(0.02, 0.98, "\n".join(annot_lines), transform=ax.transAxes,
            va="top", ha="left", fontsize=8, color="#3C3489",
            bbox=dict(facecolor="white", edgecolor="#B4B2A9",
                     alpha=0.9, boxstyle="round,pad=0.4"))

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    stats = {
        "case_sequence": case_sequence,
        "obs_pair": obs_pair,
        "reference_case": reference_case,
        "prior_mean": prior_mean,
        "bin_edges": edges,
        "bin_centers": centers,
        "binned_mean_pred": {c: binned[c][0] for c in case_sequence},
        "binned_se_pred":   {c: binned[c][1] for c in case_sequence},
        "aligned_r2":       aligned_r2,
    }
    return fig, stats


def plot_per_sim_accuracy_heatmap(param,
                                     *,
                                     case_sequence=None,
                                     obs_pair=None,
                                     results=None,
                                     param_labels=None,
                                     quantity="abs_error",     # "abs_error" | "signed_residual" | "prediction"
                                     cmap=None,
                                     vmin=None, vmax=None,
                                     n_quantile_ticks=6,
                                     show_summary_strip=True,
                                     figsize=(11, 9),
                                     perm=None,
                                     save_path=None):
    """
    Per-simulation × per-case heatmap: rows = individual val simulations
    sorted by true θ ascending, columns = cases in dual_clean_asym_order.

    Answers "which sims (in which part of the parameter range) improve as
    the noise regime shifts across cases" -- horizontal bands in the
    heatmap = groups of sims that share behavior, vertical patterns = case
    quality shifts, diagonal patterns = middle-of-range sims improving in
    middle-of-sequence cases (etc). Per-sim structure is preserved (no
    binning), sorted so the parameter-range axis is the y-axis.

    quantity:
      - "abs_error"        cell = |pred_case - true|.  Bright = high error
                           (bad), dark = low error (good).
      - "signed_residual"  cell = pred_case - true.  Diverging colormap
                           centered at 0.  Red = over-predicted here, blue =
                           under-predicted.
      - "prediction"       cell = pred_case value itself.

    A thin sidebar on the left shows the true θ gradient for each row so
    you can read which part of the parameter range a given row lives in.
    An optional summary strip along the bottom shows the median-across-
    sims value per case (a compact "how did this case do overall" view).

    Case-resolution and truth-agreement plumbing mirrors the sibling
    plotters. Returns (fig, stats).
    """
    if results is None: results = all_results
    if results is None:
        results = all_results

    # ---- parameter index
    default_labels = param_labels or _cfg.get("param_names") or [f"θ{i}" for i in range(output_dim)]
    label_to_idx = {label: i for i, label in enumerate(default_labels)}
    if isinstance(param, int):
        if not 0 <= param < output_dim:
            raise ValueError(f"Parameter index {param} out of range (0..{output_dim-1}).")
        p_idx = param
    elif isinstance(param, str):
        if param in label_to_idx:
            p_idx = label_to_idx[param]
        elif param.startswith("θ") and param[1:].isdigit():
            p_idx = int(param[1:])
        elif param.isdigit():
            p_idx = int(param)
        else:
            raise ValueError(f"Cannot interpret parameter identifier: {param!r}")
        if not 0 <= p_idx < output_dim:
            raise ValueError(f"Parameter index {p_idx} out of range.")
    else:
        raise ValueError(f"Cannot interpret parameter identifier: {param!r}")
    p_label = default_labels[p_idx]

    # ---- case sequence + obs_pair (same resolver as sibling functions)
    case_to_result = {r["case_name"]: r for r in results}
    if case_sequence is None:
        combo_pairs = set()
        for c in case_to_result:
            info = parse_case_name(c)
            if info["kind"] == "combo":
                combo_pairs.add(frozenset((info["obs1"], info["obs2"])))
        if obs_pair is None:
            if not combo_pairs:
                raise ValueError("No combo cases in results; cannot auto-detect an observable pair.")
            if len(combo_pairs) > 1:
                pretty = sorted(sorted(list(p)) for p in combo_pairs)
                raise ValueError(
                    f"Multiple observable pairs found in results: {pretty}. "
                    f"Pass obs_pair=(obsA, obsB) to disambiguate."
                )
            obs_pair = tuple(sorted(next(iter(combo_pairs))))
        else:
            wanted = frozenset(obs_pair)
            if wanted not in combo_pairs:
                pretty = sorted(sorted(list(p)) for p in combo_pairs)
                raise ValueError(
                    f"obs_pair={tuple(obs_pair)} not found in combo cases. Available pairs: {pretty}."
                )
            obs_pair = tuple(obs_pair)
        obs_set = set(obs_pair)
        filtered = []
        for c in case_to_result:
            info = parse_case_name(c)
            if info["kind"] == "combo" and {info["obs1"], info["obs2"]} == obs_set:
                filtered.append(c)
            elif info["kind"] == "single" and info["obs"] in obs_set:
                filtered.append(c)
        ordered_all, split_idx = dual_clean_asym_order(filtered)
        case_sequence = ordered_all[:split_idx]
    else:
        case_sequence = list(case_sequence)
        if obs_pair is None:
            for c in case_sequence:
                info = parse_case_name(c)
                if info["kind"] == "combo":
                    obs_pair = (info["obs1"], info["obs2"])
                    break

    if not case_sequence:
        raise ValueError("Resolved case_sequence is empty -- nothing to plot.")
    missing = [c for c in case_sequence if c not in case_to_result]
    if missing:
        raise ValueError(f"Cases not found in results: {missing}")

    # ---- collect predictions, verify truth alignment
    n_cases = len(case_sequence)
    preds_by_case = None
    true_ref = None
    for k, c in enumerate(case_sequence):
        preds, trues = get_case_predictions(case_to_result[c], mode="aligned", perm=perm)
        yp = preds[:, p_idx]; yt = trues[:, p_idx]
        if true_ref is None:
            true_ref = yt
            preds_by_case = np.zeros((len(yp), n_cases))
        else:
            if not np.allclose(yt, true_ref, rtol=1e-8, atol=1e-8):
                raise ValueError(
                    f"Truths for case {c!r} do not match the first case's truths row-for-row. "
                    f"get_case_predictions(mode='aligned') should share idx_val across cases."
                )
        preds_by_case[:, k] = yp

    n_sims = preds_by_case.shape[0]

    # ---- compute the cell quantity
    if quantity == "abs_error":
        Z = np.abs(preds_by_case - true_ref[:, None])
        default_cmap = "magma"
        cbar_label = f"|pred − true|   ({p_label})"
        center_diverging = False
    elif quantity == "signed_residual":
        Z = preds_by_case - true_ref[:, None]
        default_cmap = "RdBu_r"
        cbar_label = f"pred − true   ({p_label})"
        center_diverging = True
    elif quantity == "prediction":
        Z = preds_by_case
        default_cmap = "viridis"
        cbar_label = f"prediction   ({p_label})"
        center_diverging = False
    else:
        raise ValueError(f"quantity={quantity!r} not recognized. "
                         f"Use 'abs_error', 'signed_residual', or 'prediction'.")

    # ---- sort rows by true θ ascending
    order = np.argsort(true_ref)
    Z_sorted = Z[order]
    true_sorted = true_ref[order]

    # ---- vmin/vmax defaults
    if center_diverging:
        m = float(np.nanmax(np.abs(Z_sorted)))
        vmin_use = -m if vmin is None else vmin
        vmax_use = +m if vmax is None else vmax
    else:
        vmin_use = float(np.nanmin(Z_sorted)) if vmin is None else vmin
        vmax_use = float(np.nanmax(Z_sorted)) if vmax is None else vmax
    cmap_use = plt.get_cmap(cmap or default_cmap)

    # ---- figure layout: [true-θ sidebar] [main heatmap]  +  optional bottom summary strip
    if show_summary_strip:
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 3, width_ratios=[0.35, 20, 0.6],
                              height_ratios=[10, 0.6], hspace=0.05, wspace=0.06)
        ax_side = fig.add_subplot(gs[0, 0])
        ax_main = fig.add_subplot(gs[0, 1], sharey=ax_side)
        ax_cbar = fig.add_subplot(gs[0, 2])
        ax_summary = fig.add_subplot(gs[1, 1], sharex=ax_main)
    else:
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(1, 3, width_ratios=[0.35, 20, 0.6], wspace=0.06)
        ax_side = fig.add_subplot(gs[0, 0])
        ax_main = fig.add_subplot(gs[0, 1], sharey=ax_side)
        ax_cbar = fig.add_subplot(gs[0, 2])
        ax_summary = None

    # ---- true-θ sidebar (thin column of true values as a color gradient)
    ax_side.imshow(true_sorted[:, None], aspect="auto", cmap="cividis",
                   origin="lower", extent=[0, 1, 0, n_sims])
    ax_side.set_xticks([])
    # y-ticks at n_quantile_ticks evenly-spaced quantile positions of the sorted true θ
    tick_positions = np.linspace(0, n_sims - 1, n_quantile_ticks).astype(int)
    ax_side.set_yticks(tick_positions + 0.5)
    ax_side.set_yticklabels([f"{true_sorted[p]:.3g}" for p in tick_positions], fontsize=8)
    ax_side.set_ylabel(f"True {p_label} (sorted ascending)")
    for side in ("top", "right"):
        ax_side.spines[side].set_visible(False)

    # ---- main heatmap
    im = ax_main.imshow(Z_sorted, aspect="auto", cmap=cmap_use, origin="lower",
                        vmin=vmin_use, vmax=vmax_use,
                        extent=[0, n_cases, 0, n_sims], interpolation="nearest")
    ax_main.set_xticks(np.arange(n_cases) + 0.5)
    ax_main.set_xticklabels(case_sequence, rotation=45, ha="right", fontsize=9)
    ax_main.set_yticks([])
    for side in ("top", "right"):
        ax_main.spines[side].set_visible(False)

    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.set_label(cbar_label, fontsize=9)

    # ---- optional summary strip (median across sims per case)
    if ax_summary is not None:
        summary = np.median(Z_sorted, axis=0, keepdims=True)   # (1, n_cases)
        im2 = ax_summary.imshow(summary, aspect="auto", cmap=cmap_use, origin="lower",
                                vmin=vmin_use, vmax=vmax_use,
                                extent=[0, n_cases, 0, 1], interpolation="nearest")
        # overlay text values
        for k in range(n_cases):
            v = summary[0, k]
            ax_summary.text(k + 0.5, 0.5, f"{v:.3g}",
                            ha="center", va="center", fontsize=7.5, color="white",
                            path_effects=None)
        ax_summary.set_xticks(np.arange(n_cases) + 0.5)
        ax_summary.set_xticklabels(case_sequence, rotation=45, ha="right", fontsize=8)
        ax_summary.set_yticks([0.5])
        ax_summary.set_yticklabels(["median"], fontsize=8)
        for side in ("top", "right"):
            ax_summary.spines[side].set_visible(False)
        # hide the tick labels on the main axis if we have a summary strip
        ax_main.tick_params(axis="x", labelbottom=False)

    pair_str = f"{obs_pair[0]} ↔ {obs_pair[1]}" if obs_pair else "unknown pair"
    ax_side.set_title(
        f"Per-sim {quantity} across cases — {p_label}\n"
        f"observables: {pair_str}   |   rows sorted by true {p_label}",
        fontsize=11, loc="left", pad=10
    )

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    stats = {
        "case_sequence": case_sequence,
        "obs_pair": obs_pair,
        "quantity": quantity,
        "sort_index": order,
        "true_sorted": true_sorted,
        "Z_sorted": Z_sorted,
        "case_median": np.median(Z_sorted, axis=0),
        "case_mean":   np.nanmean(Z_sorted, axis=0),
    }
    return fig, stats


def _compute_all_chimera_preds(result, mode, x_dict, idx_val_, p_idx, batch_size, device_):
    """All chimera predictions for one case + one parameter.

    Returns a (n_val, n_val) matrix where entry [j, k] is the model's
    prediction (in physical space, matching get_case_predictions) when the
    input uses:
      - row j's KEPT observable(s), and
      - row k's SHUFFLED observable(s)

    j == k rows are still computed; the caller filters them if unwanted.
    Column layout in the model input matches the loader: features are
    concatenated in sorted(selected_observables.keys()) order. Uses the
    module-level stds/means/logflag to convert to physical space so this
    matches get_case_predictions bit-for-bit at j == k (aligned rows).
    """
    selected = result["selected_observables"]
    sel_keys_sorted = sorted(selected.keys())
    shuffle_keys, _ = resolve_shuffle(selected, mode)
    shuffle_set = set(shuffle_keys)

    idx_arr = np.asarray(idx_val_)
    n = len(idx_arr)

    # For each observable in sorted order, build (n*n, feat_dim) with the
    # convention that flat row (j*n + k) holds:
    #   arr[j] if the observable is KEPT
    #   arr[k] if the observable is SHUFFLED
    # np.repeat(arr, n, axis=0) → [arr[0]*n, arr[1]*n, ...]      → row j·n+k = arr[j]
    # np.tile(arr,   (n, 1))    → [arr[0..n-1], arr[0..n-1], ...] → row j·n+k = arr[k]
    cols = []
    for key in sel_keys_sorted:
        arr = x_dict[key][idx_arr]
        expanded = np.tile(arr, (n, 1)) if key in shuffle_set else np.repeat(arr, n, axis=0)
        cols.append(torch.from_numpy(expanded).float())
    x_all = torch.cat(cols, dim=1)

    model = result["model"].to(device_)
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, x_all.shape[0], batch_size):
            outs.append(model(x_all[i:i + batch_size].to(device_)).cpu())
    preds_flat = torch.cat(outs, dim=0).numpy()

    pred_p = preds_flat[:, p_idx] * stds[p_idx] + means[p_idx]
    if logflag[p_idx]:
        pred_p = np.exp(pred_p)
    return pred_p.reshape(n, n)


def plot_pair_normalized_shuffle_scatter(param,
                                            *,
                                            case=None,
                                            mode="obs1_vs_truth",
                                            x_pred_source="aligned",
                                            obs_pair=None,
                                            normalize_endpoints="obs1_to_obs2",
                                            n_pairs=None,
                                            pair_seed=0,
                                            all_pairs_batch_size=2048,
                                            results=None,
                                            param_labels=None,
                                            perm=None,
                                            drop_degenerate_pairs=True,
                                            min_pair_distance_frac=0.1,
                                            degenerate_eps=1e-12,
                                            clip_range=(-3.0, 3.0),
                                            show_diagonal=True,
                                            show_zero_lines=True,
                                            show_unit_lines=True,
                                            marker_alpha=None,
                                            marker_size=None,
                                            color_by_theta1=False,
                                            theta1_cmap="viridis",
                                            figsize=(7.5, 7.5),
                                            save_path=None):
    """
    Aligned vs shuffled prediction residuals for one parameter, one noise
    case, one dot per validation simulation, normalized by the pair
    truth-distance |θ_2 − θ_1|.

    For each val row j (paired with perm[j]):
        θ_1 = truth of sim_1 (the sim supplying the unshuffled channel)
        θ_2 = truth of sim_2 (the sim supplying the shuffled channel)
        θ̂_aligned  = model prediction using sim_1's full data (both obs from j)
        θ̂_shuffled = model prediction using the chimera
                     (row j's kept channel + row perm[j]'s shuffled channel)
    Plot:
        x = (θ̂_x − θ_1) / |θ_2 − θ_1|,  where θ̂_x is selected by x_pred_source:
            "aligned" (default) -> θ̂_aligned of the ANCHOR row j (both obs
                from j, unshuffled) -- the original behavior.
            "sim1"    -> the OWN aligned prediction of whichever row is
                sim_1 under this call's mode/normalize_endpoints convention
                (anchor or partner, whichever supplies θ_1's source obs).
            "sim2"    -> same, for sim_2.
        y = (θ̂_shuffled − θ_1) / |θ_2 − θ_1|

    Reads:
      - On y = x line: shuffling did not move the prediction relative to θ̂_x.
      - y > x: shuffled prediction is higher than the x-axis prediction (in
        units of pair-distance).
      - y = ±1: shuffled prediction lands one absolute pair-distance
        above (or below) sim_1's truth. When θ_2 > θ_1, y=+1 means the
        prediction has been pulled fully onto sim_2's truth.

    Reuses the truth-alignment machinery from plot_param_pair_normalized_values:
    sim_1 is ALWAYS the anchor row j (the sim supplying the kept/unshuffled
    channel) and sim_2 is ALWAYS the partner row perm[j] (supplies the
    shuffled channel) -- this role assignment is independent of `mode`.
    normalize_endpoints (obs1_to_obs2 by default) picks which role maps to
    the 0-endpoint: obs1_to_obs2 → anchor→0, partner→1; obs2_to_obs1 → the
    reverse. `mode` separately decides which OBSERVABLE the anchor/partner
    roles each supply (see Mode below) -- it does NOT change which physical
    sim is called sim_1 vs sim_2.

    Case: `case` defaults to the both-clean combo case for the observable
    pair (auto-detected via parse_case_name).  Only single-case at a time.

    Mode: `mode` selects which observable is shuffled, i.e. which observable
    the anchor keeps vs. takes from the partner (obs1_vs_truth → anchor
    keeps obs1, partner's obs2 is shuffled in; obs2_vs_truth → anchor keeps
    obs2, partner's obs1 is shuffled in). sim_1/sim_2 stay anchor/partner
    either way -- only which observable they supply changes.

    x_pred_source: which own (unshuffled/"aligned"-style) prediction feeds
    the x-axis numerator, still offset by θ_1 and normalized by |θ_2 − θ_1|
    exactly like y.
      - "aligned" (default): the ANCHOR row j's own prediction (pa[j]) --
        original behavior, backwards compatible. Always the anchor,
        regardless of normalize_endpoints.
      - "sim1": sim_1's own prediction (pa[j] if normalize_endpoints maps
        the anchor to sim_1, else pa[k] for the partner).
      - "sim2": sim_2's own prediction, the complementary choice.
      IMPORTANT for the vertical-cluster structure noted below: it only
      holds when x_pred_source resolves to the ANCHOR's own prediction. If
      it resolves to the PARTNER's, x varies across pairs sharing the same
      anchor (since the partner changes), so clusters collapse into a
      normal scatter.

    n_pairs: how many chimeras per anchor sim.
      - None (default): one fixed perm; each anchor j gets exactly one
        partner perm[j]. n_val dots total (backwards compatible).
      - int K: draw K random perms (seeded by pair_seed). Each anchor
        gets K different partners; K × n_val dots total. Uses
        get_case_predictions (K forward passes; cache misses since each
        perm is unique).
      - "all": every UNIQUE unordered pair {j, k}, j < k (not both (j,k)
        and (k,j) -- see note below). One batched forward pass builds the
        full n × n chimera matrix; only its upper triangle is used.
        n × (n − 1) / 2 dots total (~5k for n_val ≈ 102), the exhaustive
        picture without double-counting. Uses `all_pairs_batch_size` to
        chunk the forward pass.
      IMPORTANT: for a given anchor j, pred_aligned[j] and truth[j] are
      fixed, so all pairs sharing anchor j have the SAME x value when
      x_pred_source="aligned". With "all", j only ever appears as the
      anchor for pairs where j < k, so its vertical cluster is smaller than
      j's total number of partners -- expected structure, not a rendering
      artifact.
      NOTE on why "all" is unique pairs, not all ordered pairs: including
      both (j,k) and (k,j) is fine within a single `mode`/panel (they're
      different chimeras), but comparing the two mode panels
      (obs1_vs_truth vs obs2_vs_truth) side by side, (k,j) under one mode
      reproduces the exact same raw chimera value as (j,k) under the
      other mode -- so the two panels became redundant with each other.
      Restricting to unique pairs keeps each panel's dots non-duplicated
      internally and keeps the two panels independent of each other.

    marker_alpha / marker_size default to None → auto-scaled from the
    pair count so dense enriched scatters stay legible.

    color_by_theta1: if True, colors each dot by θ_1 (t0_arr, sim_1's true
    value -- the same truth the x/y normalization is anchored to), mapped
    with theta1_cmap over the parameter's full val-set true-value range
    (not just the range of kept pairs), so color is comparable across
    mode/case calls and across side-by-side panels sharing this axis.

    Degeneracy filter is SCALE-INVARIANT across parameters. A pair is dropped
    if |θ_2 − θ_1| < max(min_pair_distance_frac × range(true), degenerate_eps),
    where range(true) is the val-set range of the parameter (per call). So the
    effective threshold auto-adapts to whether the parameter lives in
    [1e-5, 1e-4] (tiny pair distances) or [10, 100] (large pair distances) --
    a fixed absolute eps like 1e-8 was meaningless for the small-range
    parameters and never fired for the large-range ones. degenerate_eps
    remains as a tiny absolute floor against literal-zero denominators.

    Returns (fig, stats).
    """
    if results is None: results = all_results
    if results is None:
        results = all_results

    # --- resolve parameter (mirrors sibling functions) ---
    default_labels = param_labels or _cfg.get("param_names") or [f"θ{i}" for i in range(output_dim)]
    label_to_idx = {label: i for i, label in enumerate(default_labels)}
    if isinstance(param, int):
        if not 0 <= param < output_dim:
            raise ValueError(f"Parameter index {param} out of range.")
        p_idx = param
    elif isinstance(param, str):
        if param in label_to_idx:
            p_idx = label_to_idx[param]
        elif param.startswith("θ") and param[1:].isdigit():
            p_idx = int(param[1:])
        elif param.isdigit():
            p_idx = int(param)
        else:
            raise ValueError(f"Cannot interpret parameter identifier: {param!r}")
        if not 0 <= p_idx < output_dim:
            raise ValueError(f"Parameter index {p_idx} out of range.")
    else:
        raise ValueError(f"Cannot interpret parameter identifier: {param!r}")
    p_label = default_labels[p_idx]

    if mode not in ("obs1_vs_truth", "obs2_vs_truth"):
        raise ValueError(f"mode must be 'obs1_vs_truth' or 'obs2_vs_truth', not {mode!r}.")

    if x_pred_source not in ("aligned", "sim1", "sim2"):
        raise ValueError(f"x_pred_source must be 'aligned', 'sim1', or 'sim2', not {x_pred_source!r}.")

    # --- resolve obs_pair (default: analysis-wide observable_1/observable_2) ---
    if obs_pair is None:
        obs_pair = (observable_1, observable_2)
    obs1_key, obs2_key = obs_pair
    obs_set = set(obs_pair)

    # --- resolve case (default: both-clean combo for this pair) ---
    # Case NAMES often use short labels ("sfr_0.0_ms_0.0") that don't literally
    # equal the full observable keys stored in selected_observables
    # ("SFR_Ms_s61", "Ms_Mh_s61").  Match observables via the case's actual
    # selected_observables dict; use parse_case_name only for kind + noise.
    case_to_result = {r["case_name"]: r for r in results}
    if case is None:
        both_clean_candidates = []
        for c, r in case_to_result.items():
            info = parse_case_name(c)
            if not (info["kind"] == "combo"
                    and info.get("noise1") == 0.0
                    and info.get("noise2") == 0.0):
                continue
            selected_c = set(r["selected_observables"])
            both_clean_candidates.append((c, sorted(selected_c)))
            if selected_c == obs_set:
                case = c
                break
        if case is None:
            if both_clean_candidates:
                raise ValueError(
                    f"No both-clean combo case whose selected_observables == "
                    f"{sorted(obs_pair)}. Both-clean combos found with other "
                    f"observables: {both_clean_candidates}. "
                    f"Pass `case` explicitly or adjust `obs_pair`."
                )
            raise ValueError(
                f"No both-clean combo case in results (no case with "
                f"parse_case_name kind='combo' and noise1=noise2=0). "
                f"Pass `case` explicitly. Cases available: "
                f"{sorted(case_to_result)}"
            )
    if case not in case_to_result:
        raise ValueError(f"Case {case!r} not in results.")
    result = case_to_result[case]
    selected = set(result["selected_observables"])
    if not obs_set.issubset(selected):
        missing = obs_set - selected
        raise ValueError(
            f"Case {case!r} is missing {sorted(missing)}; scatter needs both "
            f"{obs1_key} and {obs2_key} present so the shuffle produces a real chimera."
        )

    # --- perm (same fallback pattern as sibling) ---
    if perm is None:
        perm = _cfg.get("perm")
        if perm is None or len(perm) != len(_eval_idx()):
            perm = np.random.permutation(len(_eval_idx()))
    perm = np.asarray(perm)
    if len(perm) != len(_eval_idx()):
        raise ValueError("perm length must match len(_eval_idx()).")

    # --- aligned predictions from the shared cache (used regardless of n_pairs) ---
    preds_a, true_a = get_case_predictions(result, mode="aligned", perm=perm)
    pa = preds_a[:, p_idx]                # θ̂_aligned per anchor row (fixed)
    truth_base = true_a[:, p_idx]         # truth per row
    n_val = len(pa)

    # --- build pair arrays (anchors, partners, pred_shuf_per_pair) per n_pairs mode ---
    if n_pairs is None:
        # Single-perm mode (original behaviour). One pair per anchor.
        preds_s, true_s = get_case_predictions(result, mode=mode, perm=perm)
        if not np.allclose(true_a, true_s, rtol=1e-8, atol=1e-8):
            raise ValueError(
                "aligned vs shuffled truth vectors differ -- get_case_predictions returned "
                "inconsistent truths across modes."
            )
        anchors = np.arange(n_val)
        partners = perm
        pred_shuf_arr = preds_s[:, p_idx]
        n_pairs_desc = f"1 perm × {n_val} rows = {n_val}"
    elif isinstance(n_pairs, int) and not isinstance(n_pairs, bool):
        if n_pairs < 1:
            raise ValueError(f"n_pairs must be >= 1 (got {n_pairs}).")
        K = n_pairs
        rng_ = np.random.default_rng(pair_seed)
        perms = [rng_.permutation(n_val) for _ in range(K)]
        anchors = np.tile(np.arange(n_val), K)
        partners = np.concatenate(perms)
        shuf_chunks = []
        for pk in perms:
            preds_k, _ = get_case_predictions(result, mode=mode, perm=pk)
            shuf_chunks.append(preds_k[:, p_idx])
        pred_shuf_arr = np.concatenate(shuf_chunks)
        n_pairs_desc = f"{K} perms × {n_val} rows = {K * n_val}"
    elif isinstance(n_pairs, str) and n_pairs == "all":
        chimera = _compute_all_chimera_preds(
            result, mode, x_normalized_dict, idx_val, p_idx,
            batch_size=all_pairs_batch_size, device_=device,
        )  # (n_val, n_val); [j, k] = pred with anchor j + partner k
        # Unique unordered pairs only (j < k), not both (j,k) and (k,j).
        # Within one mode those two orderings are genuinely different
        # chimeras (different kept/borrowed assignment) -- but compared
        # ACROSS mode=obs1_vs_truth vs obs2_vs_truth, the (k,j) ordering
        # under one mode reproduces the exact same raw chimera value as the
        # (j,k) ordering under the other mode (same observables, same
        # sources), just renormalized against a different truth -- so
        # including both here made the two mode panels redundant with each
        # other. One ordering per unordered pair keeps them independent.
        anchors, partners = np.triu_indices(n_val, k=1)
        pred_shuf_arr = chimera[anchors, partners]
        n_pairs_desc = f"unique j<k pairs = {n_val}×{n_val - 1}/2 = {len(anchors)}"
    else:
        raise ValueError(
            f"n_pairs must be None, a positive int, or 'all'; got {n_pairs!r}."
        )

    # --- per-pair truths + own predictions (anchor + partner) ---
    pa_arr         = pa[anchors]          # anchor's own ("aligned") prediction
    pa_partner_arr = pa[partners]         # partner's own ("aligned") prediction
    truth_j        = truth_base[anchors]  # truth of the KEPT-obs sim (anchor)
    truth_k        = truth_base[partners] # truth of the SHUFFLED-obs sim (partner)

    # `mode` decides which OBSERVABLE each role supplies: anchor keeps obs1
    # intact (partner's obs2 gets shuffled in) under obs1_vs_truth; anchor
    # keeps obs2 intact (partner's obs1 gets shuffled in) under
    # obs2_vs_truth. This is descriptive only -- the actual shuffle already
    # happened upstream via resolve_shuffle inside get_case_predictions.
    if mode == "obs1_vs_truth":
        anchor_obs_key, partner_obs_key = obs1_key, obs2_key
    else:  # obs2_vs_truth
        anchor_obs_key, partner_obs_key = obs2_key, obs1_key

    # sim_1 = the ANCHOR (supplies the kept/unshuffled channel), sim_2 = the
    # PARTNER (supplies the shuffled channel) -- fixed roles, independent of
    # `mode` (mode only decides which observable each role supplies).
    # normalize_endpoints picks which role maps to the 0-endpoint.
    if normalize_endpoints == "obs1_to_obs2":
        t0_arr, t1_arr = truth_j, truth_k
        pred_sim1_arr, pred_sim2_arr = pa_arr, pa_partner_arr
        endpoint_desc = (f"sim_1 = anchor, supplies {anchor_obs_key} (→0);  "
                          f"sim_2 = partner, supplies {partner_obs_key} (→1)")
    elif normalize_endpoints == "obs2_to_obs1":
        t0_arr, t1_arr = truth_k, truth_j
        pred_sim1_arr, pred_sim2_arr = pa_partner_arr, pa_arr
        endpoint_desc = (f"sim_1 = partner, supplies {partner_obs_key} (→0);  "
                          f"sim_2 = anchor, supplies {anchor_obs_key} (→1)")
    else:
        raise ValueError("normalize_endpoints must be 'obs1_to_obs2' or 'obs2_to_obs1'.")

    denom_abs = np.abs(t1_arr - t0_arr)
    # Scale-invariant degeneracy threshold: FRACTION of the parameter's own true-
    # value range, not a fixed absolute epsilon. A hardcoded epsilon like 1e-8 is
    # meaningless for a parameter that lives in [1e-5, 1e-4] and irrelevant for
    # a parameter in [10, 100]. degenerate_eps stays as a tiny absolute floor.
    param_range = float(true_a[:, p_idx].max() - true_a[:, p_idx].min())
    frac_threshold = min_pair_distance_frac * param_range if drop_degenerate_pairs else 0.0
    threshold = max(frac_threshold, degenerate_eps)
    if drop_degenerate_pairs:
        keep = denom_abs > threshold
    else:
        keep = np.ones_like(denom_abs, dtype=bool)
    n_total = len(anchors)
    n_kept = int(keep.sum()); n_dropped = int((~keep).sum())
    if n_kept == 0:
        raise ValueError(
            f"All {n_total} pairs are degenerate (|θ_2 − θ_1| ≤ {threshold:g} = "
            f"max({min_pair_distance_frac}×range={frac_threshold:g}, "
            f"degenerate_eps={degenerate_eps:g})). "
            f"Try different pair_seed / lower min_pair_distance_frac."
        )

    # x source selectable via x_pred_source; still offset by θ_1 and
    # normalized by |θ_2 − θ_1|, same convention as y (user's Q5 convention:
    # numerator is (θ̂ − θ_1), i.e. prediction − sim_1 truth).
    pred_for_x = {"aligned": pa_arr, "sim1": pred_sim1_arr, "sim2": pred_sim2_arr}[x_pred_source]
    x = (pred_for_x - t0_arr)[keep] / denom_abs[keep]
    y = (pred_shuf_arr - t0_arr)[keep] / denom_abs[keep]

    # --- auto-scale marker alpha and size based on pair count ---
    if marker_alpha is None:
        if n_kept <= 200:      marker_alpha = 0.65
        elif n_kept <= 2000:   marker_alpha = 0.25
        else:                  marker_alpha = 0.10
    if marker_size is None:
        if n_kept <= 200:      marker_size = 22
        elif n_kept <= 2000:   marker_size = 8
        else:                  marker_size = 4

    # --- plot ---
    fig, ax = plt.subplots(figsize=figsize)

    if show_zero_lines:
        ax.axhline(0, color="#B4B2A9", lw=0.9, alpha=0.7, zorder=1)
        ax.axvline(0, color="#B4B2A9", lw=0.9, alpha=0.7, zorder=1)
    if show_unit_lines:
        for u in (-1.0, 1.0):
            ax.axhline(u, color="#993C1D", ls=":", lw=0.9, alpha=0.55, zorder=1)
            ax.axvline(u, color="#993C1D", ls=":", lw=0.9, alpha=0.55, zorder=1)
    if show_diagonal:
        lo, hi = clip_range
        ax.plot([lo, hi], [lo, hi], "--", color="#5F5E5A", lw=1.2, alpha=0.6,
                zorder=1, label="y = x (shuffling had no effect)")

    if color_by_theta1:
        theta1_vmin = float(true_a[:, p_idx].min())
        theta1_vmax = float(true_a[:, p_idx].max())
        sc = ax.scatter(x, y, s=marker_size, alpha=marker_alpha, c=t0_arr[keep],
                         cmap=theta1_cmap, vmin=theta1_vmin, vmax=theta1_vmax,
                         edgecolor="none", zorder=3, label=f"{n_kept} sims")
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(rf"$\theta_1$ true value ({p_label})", fontsize=8)
    else:
        ax.scatter(x, y, s=marker_size, alpha=marker_alpha, color="#3C3489",
                   edgecolor="none", zorder=3, label=f"{n_kept} sims")

    ax.set_xlim(*clip_range)
    ax.set_ylim(*clip_range)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)

    x_symbol = {"aligned": r"\hat{\theta}_1^{\rm aligned}",
                "sim1": r"\hat{\theta}_{\rm sim_1}",
                "sim2": r"\hat{\theta}_{\rm sim_2}"}[x_pred_source]
    ax.set_xlabel(rf"$({x_symbol} - \theta_1)\ /\ |\theta_2 - \theta_1|$")
    ax.set_ylabel(r"$(\hat{\theta}_1^{\rm shuffled} - \theta_1)\ /\ |\theta_2 - \theta_1|$")

    mode_desc = {"obs1_vs_truth": f"shuffle {obs2_key}  (kept: {obs1_key})",
                 "obs2_vs_truth": f"shuffle {obs1_key}  (kept: {obs2_key})"}[mode]
    ax.set_title(
        f"Pair-normalized aligned vs shuffled residuals — {p_label}\n"
        f"case: {case}   |   mode: {mode} ({mode_desc})   |   pairs: {n_pairs_desc}\n"
        f"{endpoint_desc}",
        fontsize=10
    )
    ax.legend(fontsize=8, loc="best", framealpha=0.9)

    annot_lines = [
        "each dot = one (anchor sim, partner sim) chimera",
        f"x_pred_source={x_pred_source!r}: same anchor → same x only if this "
        "resolves to the anchor's own prediction (see docstring)",
        "on y=x → shuffling didn't move the prediction",
        "y > x → shuffled pred is higher than the x-axis pred",
        "y = ±1 → shuffled pred is one |θ_2−θ_1| off sim_1's truth",
    ]
    if drop_degenerate_pairs and n_dropped:
        annot_lines.append(
            f"dropped {n_dropped}/{n_total} degenerate pairs "
            f"(|θ_2−θ_1| ≤ {threshold:.3g}, i.e. {min_pair_distance_frac*100:g}% of {p_label}'s range)"
        )
    ax.text(0.02, 0.98, "\n".join(annot_lines), transform=ax.transAxes,
            va="top", ha="left", fontsize=8, color="#3C3489",
            bbox=dict(facecolor="white", edgecolor="#B4B2A9",
                     alpha=0.9, boxstyle="round,pad=0.4"))

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    stats = {
        "case": case,
        "mode": mode,
        "x_pred_source": x_pred_source,
        "obs_pair": obs_pair,
        "normalize_endpoints": normalize_endpoints,
        "n_pairs": n_pairs,
        "n_val": n_val,
        "n_total_pairs": n_total,
        "n_kept": n_kept,
        "n_dropped": n_dropped,
        "x": x,
        "y": y,
        "anchors": anchors[keep],
        "partners": partners[keep],
        "denom_abs": denom_abs[keep],
        "truth_sim1": t0_arr[keep],
        "truth_sim2": t1_arr[keep],
        "pred_aligned":  pa_arr[keep],
        "pred_x_source": pred_for_x[keep],
        "pred_shuffled": pred_shuf_arr[keep],
    }
    return fig, stats


# ---------------------------------------------------------------------------
# Marginal-posterior plots — powered by pipeline.predict_with_uncertainty.
# The moment network (Jeffrey & Wandelt 2020) gives per-parameter Gaussian
# marginals (mu, sigma). These plots overlay obs1-only vs obs2-only vs both
# to show how each observable's information composes for a chosen parameter.
# ---------------------------------------------------------------------------


def _resolve_param_idx(param):
    """Accepts int, "theta{k}", "θ{k}", "k", or a param_names label."""
    labels = param_names or [f"θ{i}" for i in range(output_dim)]
    if isinstance(param, int):
        return param, labels[param]
    if isinstance(param, str):
        if param in labels:
            i = labels.index(param); return i, labels[i]
        for prefix in ("θ", "theta"):
            if param.startswith(prefix) and param[len(prefix):].isdigit():
                i = int(param[len(prefix):]); return i, labels[i]
        if param.isdigit():
            i = int(param); return i, labels[i]
    raise ValueError(f"could not resolve param {param!r}")


def _cases_with_moment(cases, results=None):
    """Filter to cases that actually have a moment_model attached."""
    if results is None:
        results = all_results
    name_to_r = {r["case_name"]: r for r in results}
    out = []
    for c in cases:
        r = name_to_r.get(c)
        if r is None:
            print(f"[warn] case {c!r} not in results — skipping")
            continue
        if "moment_model" not in r:
            print(f"[warn] case {c!r} has no moment_model; call fit_moment_head first — skipping")
            continue
        out.append(r)
    return out


def _gaussian_pdf(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def plot_marginal_posterior_1d(param, sim_idx, cases, *, space="log_partial",
                                results=None, figsize=(7.5, 4.5),
                                x_span_sigmas=4.0, show_true=True, ax=None,
                                colors=None, save_path=None):
    """Overlay p(θ|x_sim) Gaussian marginals for one validation sim, one parameter.

    Each case with a trained moment_model contributes a Gaussian; use this to
    compare obs1-only vs obs2-only vs both-clean posteriors for a chosen sim.

    Parameters
    ----------
    param : int or label
    sim_idx : int
        Index into the validation set (0..len(_eval_idx())-1), NOT a raw sim id.
    cases : list of case_name strings
    space : "log_partial" (default; recommended for log params), "normalized", or "physical"
    """
    pidx, plabel = _resolve_param_idx(param)
    res_list = _cases_with_moment(cases, results=results)
    if not res_list:
        raise ValueError("No cases have a trained moment_model.")

    # Collect (mu, sigma, truth) for this sim, per case
    entries = []
    for r in res_list:
        mu, sigma, truth = predict_with_uncertainty(r, indices=idx_val, space=space)
        entries.append({
            "case": r["case_name"],
            "mu": float(mu[sim_idx, pidx]),
            "sigma": float(sigma[sim_idx, pidx]),
            "truth": float(truth[sim_idx, pidx]),
        })

    # X-axis: span x_span_sigmas around the widest posterior, include truth
    lo = min(e["mu"] - x_span_sigmas * e["sigma"] for e in entries)
    hi = max(e["mu"] + x_span_sigmas * e["sigma"] for e in entries)
    true_val = entries[0]["truth"]
    lo = min(lo, true_val); hi = max(hi, true_val)
    pad = 0.1 * (hi - lo) if hi > lo else 1.0
    xs = np.linspace(lo - pad, hi + pad, 400)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    if colors is None:
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i) for i in range(len(entries))]

    for e, c in zip(entries, colors):
        ys = _gaussian_pdf(xs, e["mu"], e["sigma"])
        ax.plot(xs, ys, color=c, linewidth=2,
                label=f"{e['case']}  (μ={e['mu']:.2f}, σ={e['sigma']:.2f})")
        ax.axvline(e["mu"], color=c, linewidth=1, linestyle=":", alpha=0.5)

    if show_true:
        ax.axvline(true_val, color="red", linewidth=2, linestyle="--",
                   label=f"true = {true_val:.2f}")

    ax.set_xlabel(f"{plabel}  ({space})")
    ax.set_ylabel("p(θ | x)")
    ax.set_title(f"Marginal posterior — {plabel}, val sim #{sim_idx}")
    ax.legend(loc="best", fontsize=8, frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
    return fig


def plot_marginal_posterior_grid(param, cases, *, sim_indices=None, n_sims=6,
                                  space="log_partial", seed=0, results=None,
                                  figsize_per_panel=(4.5, 3.2), ncols=3,
                                  save_path=None):
    """Grid of Gaussian marginals: rows/cols = sims, panels overlay `cases`.

    Shows how one parameter's posterior narrows/shifts across a small sample of
    validation sims when different observable subsets feed the network.
    """
    pidx, plabel = _resolve_param_idx(param)
    res_list = _cases_with_moment(cases, results=results)
    if not res_list:
        raise ValueError("No cases have a trained moment_model.")

    n_val = len(np.asarray(idx_val))
    if sim_indices is None:
        rng = np.random.default_rng(seed)
        sim_indices = rng.choice(n_val, size=min(n_sims, n_val), replace=False)
    sim_indices = list(sim_indices)

    ncols = min(ncols, len(sim_indices))
    nrows = int(np.ceil(len(sim_indices) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per_panel[0] * ncols,
                                      figsize_per_panel[1] * nrows),
                             squeeze=False)

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(len(res_list))]
    case_names = [r["case_name"] for r in res_list]

    for i, s in enumerate(sim_indices):
        ax = axes.flat[i]
        plot_marginal_posterior_1d(pidx, int(s), case_names, space=space,
                                    results=results, ax=ax, colors=colors,
                                    show_true=True)
        ax.set_title(f"val sim #{int(s)}", fontsize=10)
        ax.legend().set_visible(i == 0)  # only show legend on first panel

    for j in range(len(sim_indices), nrows * ncols):
        fig.delaxes(axes.flat[j])

    fig.suptitle(f"{plabel} — marginal posteriors across cases and validation sims",
                 fontsize=13)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
    return fig


def plot_sigma_by_case_bars(params, cases, *, space="log_partial", results=None,
                             reducer="median", figsize=(11, 5.5), save_path=None):
    """Bar chart of population-level predicted σ per case per parameter.

    For each (param, case), aggregate σ across all validation sims (median or mean).
    Shows which observable subsets deliver tighter posteriors on average.
    """
    pidx_list, labels = zip(*[_resolve_param_idx(p) for p in params])
    res_list = _cases_with_moment(cases, results=results)
    if not res_list:
        raise ValueError("No cases have a trained moment_model.")

    reduce_fn = {"median": np.median, "mean": np.mean}[reducer]

    # sigma_by_case[case_name] -> [n_params] aggregated σ
    sigma_by_case = {}
    for r in res_list:
        _, sigma, _ = predict_with_uncertainty(r, indices=idx_val, space=space)
        sigma_by_case[r["case_name"]] = np.array(
            [reduce_fn(sigma[:, p]) for p in pidx_list]
        )

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(pidx_list))
    n_cases = len(res_list)
    bar_w = 0.8 / n_cases
    cmap = plt.get_cmap("tab10")
    for i, r in enumerate(res_list):
        c = r["case_name"]
        offset = (i - (n_cases - 1) / 2) * bar_w
        ax.bar(x + offset, sigma_by_case[c], width=bar_w, label=c, color=cmap(i))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"{reducer} predicted σ  ({space})")
    ax.set_title("Posterior width per parameter, by case")
    ax.legend(loc="best", fontsize=9, frameon=False)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
    return fig


def plot_pull_distribution(cases, *, space="normalized", results=None,
                            params=None, figsize=(11, 4.5), bins=30,
                            save_path=None):
    """Calibration diagnostic. Pull = (μ - true) / σ; well-calibrated → N(0, 1).

    One panel per case; overlays the standard normal for reference. Numbers
    printed on each panel:  mean(pull), std(pull). |mean| >> 0 => bias.
    std >> 1 => under-predicted σ (over-confident). std << 1 => over-predicted σ.

    space defaults to "normalized" because pull magnitudes are directly comparable
    across parameters there — physical space mixes units.
    """
    res_list = _cases_with_moment(cases, results=results)
    if not res_list:
        raise ValueError("No cases have a trained moment_model.")

    if params is None:
        param_idxs = list(range(output_dim))
    else:
        param_idxs = [_resolve_param_idx(p)[0] for p in params]

    n = len(res_list)
    ncols = min(4, n); nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize[0], figsize[1] * nrows / 1),
                             squeeze=False)

    xs = np.linspace(-4, 4, 200)
    ref = _gaussian_pdf(xs, 0, 1)

    for i, r in enumerate(res_list):
        ax = axes.flat[i]
        mu, sigma, truth = predict_with_uncertainty(r, indices=idx_val, space=space)
        pull = ((mu - truth) / sigma)[:, param_idxs].reshape(-1)
        pull_clean = pull[np.isfinite(pull)]
        ax.hist(pull_clean, bins=bins, density=True, alpha=0.6,
                color="tab:blue", edgecolor="none")
        ax.plot(xs, ref, "k--", linewidth=1.5, label="N(0, 1)")
        ax.set_xlim(-4, 4)
        ax.set_title(r["case_name"], fontsize=10)
        ax.text(0.02, 0.95,
                f"mean={pull_clean.mean():.2f}\nstd={pull_clean.std():.2f}",
                transform=ax.transAxes, va="top", ha="left",
                fontsize=9, family="monospace",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
        ax.axvline(0, color="red", linewidth=0.8, alpha=0.5)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8, frameon=False)

    for j in range(n, nrows * ncols):
        fig.delaxes(axes.flat[j])

    fig.suptitle("Moment-network calibration — pull = (μ − true) / σ",
                 fontsize=12)
    for ax in axes.flat[:n]:
        ax.set_xlabel("pull")
    axes.flat[0].set_ylabel("density")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
    return fig


# ---------------------------------------------------------------------------
# Covariance plots — need results with fit_moment_head applied.
# ---------------------------------------------------------------------------


def _find_case(case_name, results=None):
    if results is None:
        results = all_results
    for r in results:
        if r["case_name"] == case_name:
            return r
    raise KeyError(f"case {case_name!r} not in results")


def plot_covariance_matrix(case, sim_idx, *, results=None, space="log_partial",
                            show_correlation=False, figsize=(6.5, 5.5),
                            save_path=None):
    """Heatmap of the k×k covariance (or correlation) matrix for one case, one sim."""
    r = _find_case(case, results)
    if "moment_model" not in r:
        raise ValueError(f"case {case!r} has no moment_model — call fit_moment_head first.")
    mu, cov, truth = predict_covariance_matrix(r, space=space)
    C = cov[sim_idx]
    if show_correlation:
        C = cov_to_corr(C[None])[0]
    labels = r["moment_focus_params"]
    fig, ax = plt.subplots(figsize=figsize)
    kwargs = dict(annot=True, fmt=".2f", xticklabels=labels, yticklabels=labels,
                  linewidths=0.3, cbar_kws={"label": "correlation" if show_correlation else "covariance"})
    if show_correlation:
        import seaborn as sns
        sns.heatmap(C, ax=ax, vmin=-1, vmax=1, center=0.0, cmap="RdBu_r", **kwargs)
    else:
        import seaborn as sns
        sns.heatmap(C, ax=ax, center=0.0, cmap="RdBu_r", **kwargs)
    kind = "correlation" if show_correlation else "covariance"
    ax.set_title(f"{kind} — case {case}, val sim #{sim_idx}  ({space})")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
    return fig


def plot_median_correlation_grid(cases, *, results=None, space="log_partial",
                                   figsize_per_panel=(4.5, 4.0), save_path=None):
    """Per-case median correlation matrix across validation sims (one heatmap per case)."""
    import seaborn as sns
    res_list = _cases_with_moment(cases, results)
    if not res_list:
        raise ValueError("No cases with covariance heads.")
    n = len(res_list)
    ncols = min(3, n); nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
                              squeeze=False)
    for i, r in enumerate(res_list):
        ax = axes.flat[i]
        _, cov, _ = predict_covariance_matrix(r, space=space)
        corr = cov_to_corr(cov)               # [N, k, k]
        med = np.median(corr, axis=0)
        labels = r["moment_focus_params"]
        sns.heatmap(med, ax=ax, vmin=-1, vmax=1, center=0.0, cmap="RdBu_r",
                    annot=True, fmt=".2f", linewidths=0.3,
                    xticklabels=labels, yticklabels=labels,
                    cbar_kws={"label": "median correlation"})
        ax.set_title(r["case_name"], fontsize=10)
    for j in range(n, nrows * ncols):
        fig.delaxes(axes.flat[j])
    fig.suptitle(f"Median correlation across val sims  ({space})", fontsize=12)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
    return fig


def plot_marginal_ellipse_2d(param_i, param_j, sim_idx, cases, *, results=None,
                              space="log_partial", figsize=(7.5, 7.5),
                              n_sigma=(1, 2), show_true=True, colors=None,
                              save_path=None):
    """Overlay 2D Gaussian marginal ellipses for a param pair across cases, one sim.

    Each case's ellipse is drawn at each specified n-sigma level from the k×k
    covariance restricted to the (i, j) sub-block. Center = predicted mean.
    """
    from matplotlib.patches import Ellipse
    res_list = _cases_with_moment(cases, results)
    if not res_list:
        raise ValueError("No cases with covariance heads.")

    # Both params must be in every case's focus params
    for r in res_list:
        if param_i not in r["moment_focus_params"]:
            raise ValueError(f"param {param_i!r} not in moment_focus_params for {r['case_name']!r}: "
                             f"{r['moment_focus_params']}")
        if param_j not in r["moment_focus_params"]:
            raise ValueError(f"param {param_j!r} not in moment_focus_params for {r['case_name']!r}: "
                             f"{r['moment_focus_params']}")

    fig, ax = plt.subplots(figsize=figsize)
    if colors is None:
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i) for i in range(len(res_list))]

    truth_ij = None
    for r, c in zip(res_list, colors):
        mu, cov, truth = predict_covariance_matrix(r, space=space)
        li = r["moment_focus_params"].index(param_i)
        lj = r["moment_focus_params"].index(param_j)
        m = mu[sim_idx, [li, lj]]
        C = cov[sim_idx][np.ix_([li, lj], [li, lj])]
        w, V = np.linalg.eigh(C)
        # major axis is the larger eigenvalue
        order = np.argsort(w)[::-1]
        w = w[order]; V = V[:, order]
        angle_deg = np.degrees(np.arctan2(V[1, 0], V[0, 0]))
        for k, style in zip(n_sigma, ["-", "--", ":"]):
            width  = 2 * k * np.sqrt(max(w[0], 1e-30))
            height = 2 * k * np.sqrt(max(w[1], 1e-30))
            e = Ellipse(xy=m, width=width, height=height, angle=angle_deg,
                        edgecolor=c, facecolor="none", linewidth=1.6, linestyle=style,
                        label=f"{r['case_name']} {k}σ" if k == n_sigma[0] else None)
            ax.add_patch(e)
        ax.plot(m[0], m[1], "o", color=c, markersize=8)
        if truth_ij is None:
            truth_ij = truth[sim_idx, [li, lj]]

    if show_true and truth_ij is not None:
        ax.plot(truth_ij[0], truth_ij[1], "kx", markersize=14, markeredgewidth=2.5,
                label="true")

    ax.set_xlabel(f"{param_i}  ({space})")
    ax.set_ylabel(f"{param_j}  ({space})")
    ax.set_title(f"2D marginal — {param_i} vs {param_j},  val sim #{sim_idx}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, frameon=False)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
    return fig


# ---------------------------------------------------------------------------
# Summary plots — corner + prediction scatter + errorbar overview.
# Prefer these over the per-sim covariance/correlation/ellipse plots for
# the moment_example — they give the full picture without the per-sim clutter.
# ---------------------------------------------------------------------------


def plot_corner(sim_idx, cases, *, results=None, space="log_partial",
                 n_sigma=(1, 2), show_true=True, figsize_per_panel=1.7,
                 colors=None, save_path=None):
    """k×k corner plot: 1D Gaussian marginals on the diagonal, 2D Gaussian
    ellipses off-diagonal, cases overlaid, truth marked in red.

    For one validation-set sim, this shows the entire k×k joint posterior
    p(θ_1..θ_k | x_sim) structure at a glance: how each case constrains each
    parameter, and how the joint uncertainty ellipse rotates/shrinks across cases.
    """
    from matplotlib.patches import Ellipse
    res_list = _cases_with_moment(cases, results)
    if not res_list:
        raise ValueError("No cases with a moment_model.")

    focus_labels = res_list[0]["moment_focus_params"]
    for r in res_list:
        if r["moment_focus_params"] != focus_labels:
            raise ValueError("Cases have different focus_params; can't overlay.")
    k = len(focus_labels)

    fig, axes = plt.subplots(k, k, figsize=(figsize_per_panel * k, figsize_per_panel * k),
                              squeeze=False)
    if colors is None:
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i) for i in range(len(res_list))]

    # Precompute per-case mu/cov/truth for this sim
    per_case = []
    for r in res_list:
        mu, cov, truth = predict_covariance_matrix(r, space=space)
        per_case.append({"case": r["case_name"], "mu": mu[sim_idx], "cov": cov[sim_idx],
                          "truth": truth[sim_idx]})
    truth_vec = per_case[0]["truth"]

    # Axis ranges: span n_sigma[-1]+1 around widest posterior + include truth
    axis_ranges = []
    for i in range(k):
        lo = min(pc["mu"][i] - (n_sigma[-1] + 1) * np.sqrt(pc["cov"][i, i]) for pc in per_case)
        hi = max(pc["mu"][i] + (n_sigma[-1] + 1) * np.sqrt(pc["cov"][i, i]) for pc in per_case)
        if show_true:
            lo = min(lo, truth_vec[i]); hi = max(hi, truth_vec[i])
        pad = 0.1 * (hi - lo) if hi > lo else 1.0
        axis_ranges.append((lo - pad, hi + pad))

    for i in range(k):
        for j in range(k):
            ax = axes[i, j]
            if j > i:
                ax.set_visible(False); continue

            if i == j:
                # Diagonal: overlaid 1D Gaussian marginals
                xs = np.linspace(axis_ranges[i][0], axis_ranges[i][1], 300)
                for pc, c in zip(per_case, colors):
                    m = pc["mu"][i]; s = np.sqrt(max(pc["cov"][i, i], 1e-30))
                    ys = np.exp(-0.5 * ((xs - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
                    ax.plot(xs, ys, color=c, linewidth=1.5, label=pc["case"] if (i == 0) else None)
                if show_true:
                    ax.axvline(truth_vec[i], color="red", linewidth=1.5, linestyle="--")
                ax.set_xlim(axis_ranges[i])
                ax.set_yticks([])
                if i == 0:
                    ax.legend(loc="upper right", fontsize=7, frameon=False,
                              bbox_to_anchor=(1.02, 1.0))
            else:
                # Off-diagonal (i > j): 2D Gaussian ellipse per case
                for pc, c in zip(per_case, colors):
                    m2 = pc["mu"][[j, i]]
                    C2 = pc["cov"][np.ix_([j, i], [j, i])]
                    w, V = np.linalg.eigh(C2)
                    order = np.argsort(w)[::-1]
                    w = w[order]; V = V[:, order]
                    angle = np.degrees(np.arctan2(V[1, 0], V[0, 0]))
                    for sig, style in zip(n_sigma, ["-", "--", ":"]):
                        width  = 2 * sig * np.sqrt(max(w[0], 1e-30))
                        height = 2 * sig * np.sqrt(max(w[1], 1e-30))
                        ax.add_patch(Ellipse(xy=m2, width=width, height=height,
                                              angle=angle, edgecolor=c, facecolor="none",
                                              linewidth=1.2, linestyle=style))
                    ax.plot(m2[0], m2[1], "o", color=c, markersize=3)
                if show_true:
                    ax.plot(truth_vec[j], truth_vec[i], "x", color="red",
                             markersize=10, markeredgewidth=1.8)
                ax.set_xlim(axis_ranges[j])
                ax.set_ylim(axis_ranges[i])
                ax.grid(True, alpha=0.2)

            if i == k - 1:
                ax.set_xlabel(focus_labels[j], fontsize=8)
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(focus_labels[i], fontsize=8)
            elif j > 0:
                ax.set_yticklabels([])

    fig.suptitle(f"Corner plot — val sim #{sim_idx}  ({space})", fontsize=11, y=0.995)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_prediction_scatter_corner(case, *, results=None, space="log_partial",
                                     show_median_ellipse=True, ellipse_n_sigma=1,
                                     figsize_per_panel=1.8, save_path=None):
    """k×k scatter matrix across ALL test sims for ONE case.

    Diagonal (i, i): overlaid histograms — predicted μ_i (blue) vs truth_i (grey).
    Off-diagonal (i, j) with i > j: scatter of (μ_j, μ_i) points across sims,
      colored by |μ − truth| magnitude; median-covariance 1σ ellipse overlaid.

    Aggregate view — how well the predictions fill the parameter space, and
    what the typical joint uncertainty looks like for this case.
    """
    from matplotlib.patches import Ellipse
    r = _find_case(case, results)
    if "moment_model" not in r:
        raise ValueError(f"case {case!r} has no moment_model.")

    mu, cov, truth = predict_covariance_matrix(r, space=space)
    labels = r["moment_focus_params"]
    k = len(labels)
    # Population "typical" cov = median across sims for the ellipse
    typical_cov = np.median(cov, axis=0)
    residual_norm = np.linalg.norm(mu - truth, axis=1)   # coloring per sim

    fig, axes = plt.subplots(k, k, figsize=(figsize_per_panel * k, figsize_per_panel * k),
                              squeeze=False)

    for i in range(k):
        for j in range(k):
            ax = axes[i, j]
            if j > i:
                ax.set_visible(False); continue

            if i == j:
                # 1D marginals — histogram of predicted μ_i vs truth_i
                ax.hist(truth[:, i], bins=25, color="gray", alpha=0.4, label="truth")
                ax.hist(mu[:, i], bins=25, color="tab:blue", alpha=0.6, label="predicted")
                ax.set_yticks([])
                if i == 0:
                    ax.legend(loc="upper right", fontsize=7, frameon=False)
            else:
                # 2D scatter (μ_j vs μ_i), color by residual
                sc = ax.scatter(mu[:, j], mu[:, i], c=residual_norm, s=8, alpha=0.6,
                                 cmap="viridis")
                ax.plot(truth[:, j], truth[:, i], "x", color="red", markersize=4,
                         markeredgewidth=0.7, alpha=0.3, label="truth" if (i == 1 and j == 0) else None)
                if show_median_ellipse:
                    med_center = np.median(mu[:, [j, i]], axis=0)
                    C2 = typical_cov[np.ix_([j, i], [j, i])]
                    w, V = np.linalg.eigh(C2)
                    order = np.argsort(w)[::-1]
                    w = w[order]; V = V[:, order]
                    angle = np.degrees(np.arctan2(V[1, 0], V[0, 0]))
                    ax.add_patch(Ellipse(xy=med_center,
                                          width=2 * ellipse_n_sigma * np.sqrt(max(w[0], 1e-30)),
                                          height=2 * ellipse_n_sigma * np.sqrt(max(w[1], 1e-30)),
                                          angle=angle, edgecolor="black", facecolor="none",
                                          linewidth=1.2, linestyle="--"))
                ax.grid(True, alpha=0.2)

            if i == k - 1:
                ax.set_xlabel(labels[j], fontsize=8)
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(labels[i], fontsize=8)
            elif j > 0:
                ax.set_yticklabels([])

    fig.suptitle(f"Prediction scatter matrix — case {case}  ({space})", fontsize=11, y=0.995)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_predictions_with_errorbars(param, cases, *, results=None,
                                      space="log_partial", ncols=None,
                                      figsize_per_panel=(4.0, 4.0),
                                      save_path=None):
    """For one param: predicted μ ± σ vs true θ across all test sims, per case.

    One panel per case. y=x reference line. Points inside 1σ error bars = well-calibrated.
    Answers "does the moment head produce reasonable point-estimates AND uncertainties?"
    """
    pidx, plabel = _resolve_param_idx(param)
    res_list = _cases_with_moment(cases, results)
    if not res_list:
        raise ValueError("No cases with a moment_model.")

    # Only cases whose moment_focus_params include this param
    res_list = [r for r in res_list if plabel in r["moment_focus_params"]]
    if not res_list:
        raise ValueError(f"param {plabel!r} not in any case's moment_focus_params.")

    n = len(res_list)
    if ncols is None:
        ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
                              squeeze=False)

    for i, r in enumerate(res_list):
        ax = axes.flat[i]
        mu, cov, truth = predict_covariance_matrix(r, space=space)
        local_i = r["moment_focus_params"].index(plabel)
        mu_p = mu[:, local_i]
        sig_p = np.sqrt(np.clip(cov[:, local_i, local_i], 0.0, None))
        true_p = truth[:, local_i]

        ax.errorbar(true_p, mu_p, yerr=sig_p, fmt="o", markersize=4,
                     alpha=0.6, capsize=1.5, elinewidth=0.7, color="tab:blue",
                     ecolor="gray")
        lo = min(true_p.min(), mu_p.min())
        hi = max(true_p.max(), mu_p.max())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.2, label="y = x")
        # Pull calibration diagnostic
        pull = (mu_p - true_p) / sig_p
        pull_clean = pull[np.isfinite(pull)]
        ax.text(0.03, 0.97, f"pull std = {pull_clean.std():.2f}\nmean = {pull_clean.mean():+.2f}",
                 transform=ax.transAxes, va="top", ha="left", fontsize=8,
                 family="monospace",
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.75))
        ax.set_xlabel(f"true {plabel}")
        ax.set_ylabel(f"predicted {plabel}")
        ax.set_title(r["case_name"], fontsize=9)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="lower right", fontsize=8, frameon=False)

    for j in range(n, nrows * ncols):
        fig.delaxes(axes.flat[j])

    fig.suptitle(f"{plabel} — predicted μ ± σ vs true across test sims  ({space})",
                  fontsize=11, y=1.0)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# ---------------------------------------------------------------------------
# Exact analytic Gaussian-from-moments plots.
#
# These plot the Gaussian approximation implied by (μ, Σ) — NOT the true
# posterior. The moment network gives us only the first two moments; if the
# true posterior is non-Gaussian (multi-modal, skewed) those features are lost.
# All titles/labels reflect this.
#
# No sampling, no KDE, no chainconsumer / corner.py — exact PDF curves and
# eigendecomposition-derived ellipses only.
# ---------------------------------------------------------------------------


def _validate_psd(cov, atol_neg=1e-8, name="cov_pred"):
    """Symmetrize and check PSD. Raises ValueError with eigenvalues if not."""
    C = 0.5 * (np.asarray(cov) + np.asarray(cov).T)
    w = np.linalg.eigvalsh(C)
    if w.min() < -atol_neg:
        raise ValueError(
            f"{name} is not positive semi-definite (moment head G may not "
            f"guarantee PSD by construction). Eigenvalues: {w.tolist()}"
        )
    return C


def plot_moment_corner(mean_pred, cov_pred, theta_true=None,
                        param_labels=None, fig=None, color="C0",
                        n_std=(1, 2), label="Gaussian approx (moments)",
                        figsize_per_panel=1.7):
    """Corner plot of the Gaussian approximation implied by (mean_pred, cov_pred).

    Diagonal panels: exact analytic PDF N(mean_pred[j], cov_pred[j, j]) — no sampling.
    Lower off-diagonal panels: exact 1σ / 2σ confidence ellipses from eigendecomposition
    of each 2×2 sub-block of cov_pred (matplotlib Ellipse patches).
    Upper triangle: hidden.

    IMPORTANT: What this plots is the Gaussian family with matching first two moments.
    It is NOT the true marginal posterior unless the posterior is Gaussian. The
    moment network gives only (μ, Σ); the shape assumption is on you.

    Parameters
    ----------
    mean_pred : (k,) array_like     — posterior mean
    cov_pred  : (k, k) array_like   — posterior covariance
    theta_true : (k,) array_like, optional — overlays truth (line on diag, marker off-diag)
    param_labels : list of k str, optional
    fig : matplotlib.figure.Figure, optional
        None → create new k×k grid. Otherwise reuses the axes stored on the fig by
        a previous plot_moment_corner call (enables overlay).
    color : matplotlib color for THIS call's Gaussian
    n_std : tuple of ints — sigma levels for off-diagonal ellipses
    label : legend text for THIS call

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    from matplotlib.patches import Ellipse

    mean_pred = np.asarray(mean_pred, dtype=float)
    C = _validate_psd(cov_pred, name="cov_pred")
    k = len(mean_pred)
    if C.shape != (k, k):
        raise ValueError(f"cov_pred shape {C.shape} incompatible with mean_pred length {k}")
    if param_labels is None:
        param_labels = [f"θ{i}" for i in range(k)]
    elif len(param_labels) != k:
        raise ValueError(f"param_labels length {len(param_labels)} != k={k}")
    if theta_true is not None:
        theta_true = np.asarray(theta_true, dtype=float)

    # Retrieve or create the axes grid
    if fig is None:
        fig, axes = plt.subplots(k, k, figsize=(figsize_per_panel * k, figsize_per_panel * k),
                                  squeeze=False)
        for i in range(k):
            for j in range(k):
                if j > i:
                    axes[i, j].set_visible(False)
        fig._moment_corner_axes = axes
        fig._moment_corner_labels_drawn = False
    else:
        if not hasattr(fig, "_moment_corner_axes"):
            raise ValueError("fig was not created by plot_moment_corner — cannot overlay.")
        axes = fig._moment_corner_axes

    # Axis ranges: enlarge existing if any, or set fresh
    axis_ranges = getattr(fig, "_moment_corner_ranges", [None] * k)
    for i in range(k):
        lo_i = mean_pred[i] - (max(n_std) + 0.5) * np.sqrt(max(C[i, i], 1e-30))
        hi_i = mean_pred[i] + (max(n_std) + 0.5) * np.sqrt(max(C[i, i], 1e-30))
        if theta_true is not None:
            lo_i = min(lo_i, theta_true[i]); hi_i = max(hi_i, theta_true[i])
        if axis_ranges[i] is not None:
            lo_i = min(lo_i, axis_ranges[i][0])
            hi_i = max(hi_i, axis_ranges[i][1])
        axis_ranges[i] = (lo_i, hi_i)
    fig._moment_corner_ranges = axis_ranges

    # Draw
    for i in range(k):
        for j in range(k):
            if j > i:
                continue
            ax = axes[i, j]

            if i == j:
                lo, hi = axis_ranges[i]
                xs = np.linspace(lo, hi, 400)
                s = np.sqrt(max(C[i, i], 1e-30))
                pdf = np.exp(-0.5 * ((xs - mean_pred[i]) / s) ** 2) / (s * np.sqrt(2 * np.pi))
                ax.plot(xs, pdf, color=color, linewidth=1.5, label=label if i == 0 else None)
                if theta_true is not None:
                    ax.axvline(theta_true[i], color=color, linewidth=1, linestyle=":", alpha=0.6)
                ax.set_xlim(axis_ranges[i]); ax.set_yticks([])
            else:
                m2 = mean_pred[[j, i]]
                C2 = C[np.ix_([j, i], [j, i])]
                w, V = np.linalg.eigh(C2)
                order = np.argsort(w)[::-1]
                w = w[order]; V = V[:, order]
                angle = np.degrees(np.arctan2(V[1, 0], V[0, 0]))
                for sig, style in zip(n_std, ["-", "--", ":"]):
                    width  = 2 * sig * np.sqrt(max(w[0], 1e-30))
                    height = 2 * sig * np.sqrt(max(w[1], 1e-30))
                    ax.add_patch(Ellipse(xy=m2, width=width, height=height, angle=angle,
                                          edgecolor=color, facecolor="none",
                                          linewidth=1.1, linestyle=style))
                ax.plot(m2[0], m2[1], "o", color=color, markersize=3)
                if theta_true is not None:
                    ax.plot(theta_true[j], theta_true[i], "x", color=color,
                             markersize=8, markeredgewidth=1.4, alpha=0.7)
                ax.set_xlim(axis_ranges[j]); ax.set_ylim(axis_ranges[i])
                ax.grid(True, alpha=0.2)

            if i == k - 1 and not fig._moment_corner_labels_drawn:
                ax.set_xlabel(param_labels[j], fontsize=9)
            elif i < k - 1:
                ax.set_xticklabels([])
            if j == 0 and i > 0 and not fig._moment_corner_labels_drawn:
                ax.set_ylabel(param_labels[i], fontsize=9)
            elif j > 0:
                ax.set_yticklabels([])
    fig._moment_corner_labels_drawn = True

    # Aggregate legend on the top-right corner (upper triangle is hidden anyway)
    handles, labels = [], []
    seen = set()
    for a in axes.flat:
        for h, l in zip(*a.get_legend_handles_labels()):
            if l and l not in seen:
                handles.append(h); labels.append(l); seen.add(l)
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=9,
                   bbox_to_anchor=(0.98, 0.98))
    fig.suptitle("Corner plot — Gaussian approx (moments); NOT the true posterior",
                  fontsize=10, y=0.995)
    fig.tight_layout()
    return fig


def plot_pull_calibration(theta_true_all, mean_pred_all, cov_pred_all,
                            param_labels=None, ncols=None,
                            figsize_per_panel=(3.2, 2.8),
                            compute_joint=True, joint_jitter=1e-10):
    """Per-parameter and joint pull-calibration diagnostic for a Gaussian approximation.

    Per parameter i, computes z_i = (θ_true_i − μ_pred_i) / √Σ_pred_ii across all
    n_test cases, plots the histogram against N(0, 1). Reports empirical coverage
    against expected 68.27% / 95.45%.

    If compute_joint, also computes the joint whitened residual per test case:
        z_full = L^{-1} @ (θ_true − μ_pred)
    where L is the Cholesky factor of Σ_pred. Under a well-calibrated Gaussian,
    ‖z_full‖² is χ²_k distributed; empirical coverage vs. χ²_k CDF is reported.

    All results interpret (μ, Σ) as defining a Gaussian approximation; the diagnostic
    checks whether that Gaussian is calibrated — NOT whether the true posterior is
    Gaussian. Poor calibration could come from either mis-estimated moments OR
    non-Gaussian posterior shape.

    Parameters
    ----------
    theta_true_all : (n_test, k) array
    mean_pred_all  : (n_test, k) array
    cov_pred_all   : (n_test, k, k) array
    param_labels   : list of k str, optional
    compute_joint  : bool — run Cholesky-based joint χ² diagnostic
    joint_jitter   : PSD jitter added to Σ before Cholesky (numerical stability)

    Returns
    -------
    fig, summary_dict
        summary_dict has keys: 'per_param' (list of dicts) and, if compute_joint,
        'joint' (dict).
    """
    theta_true_all = np.asarray(theta_true_all, dtype=float)
    mean_pred_all = np.asarray(mean_pred_all, dtype=float)
    cov_pred_all = np.asarray(cov_pred_all, dtype=float)
    n, k = theta_true_all.shape
    if mean_pred_all.shape != (n, k):
        raise ValueError(f"mean_pred_all shape {mean_pred_all.shape} != ({n}, {k})")
    if cov_pred_all.shape != (n, k, k):
        raise ValueError(f"cov_pred_all shape {cov_pred_all.shape} != ({n}, {k}, {k})")
    if param_labels is None:
        param_labels = [f"θ{i}" for i in range(k)]
    elif len(param_labels) != k:
        raise ValueError(f"param_labels length {len(param_labels)} != k={k}")

    # Per-parameter pull
    sigma_diag = np.sqrt(np.clip(np.diagonal(cov_pred_all, axis1=1, axis2=2), 0.0, None))
    z_per = (theta_true_all - mean_pred_all) / np.where(sigma_diag > 0, sigma_diag, np.nan)

    per_param_stats = []
    print("=" * 72)
    print("Per-parameter pull calibration  (Gaussian approx from estimated moments)")
    print("=" * 72)
    print(f"{'param':<8}  {'|z|<1':>8}  {'|z|<2':>8}  {'mean':>8}  {'std':>8}  {'n':>6}")
    print(f"{'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*6}")
    for i, lbl in enumerate(param_labels):
        z = z_per[:, i]; z = z[np.isfinite(z)]
        cov_1 = 100 * np.mean(np.abs(z) < 1) if z.size else np.nan
        cov_2 = 100 * np.mean(np.abs(z) < 2) if z.size else np.nan
        stats = dict(param=lbl, cov_1sigma_pct=cov_1, cov_2sigma_pct=cov_2,
                     mean=float(z.mean()) if z.size else np.nan,
                     std=float(z.std()) if z.size else np.nan, n=int(z.size))
        per_param_stats.append(stats)
        print(f"{lbl:<8}  {cov_1:>7.1f}%  {cov_2:>7.1f}%  {z.mean():>+8.2f}  {z.std():>8.2f}  {z.size:>6d}")
    print(f"{'ideal':<8}  {'68.3%':>8}  {'95.5%':>8}  {'0.00':>8}  {'1.00':>8}")

    ncols = ncols or min(4, k)
    nrows = int(np.ceil(k / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
                              squeeze=False)
    xs = np.linspace(-4, 4, 400)
    pdf = np.exp(-0.5 * xs ** 2) / np.sqrt(2 * np.pi)
    for i, lbl in enumerate(param_labels):
        ax = axes.flat[i]
        z = z_per[:, i]; z = z[np.isfinite(z)]
        if z.size:
            ax.hist(z, bins=25, density=True, alpha=0.55, color="steelblue",
                     edgecolor="none")
        ax.plot(xs, pdf, "k--", linewidth=1.4, label="N(0, 1)")
        ax.axvline(0, color="red", linewidth=0.6, alpha=0.7)
        ax.set_xlim(-4, 4)
        ax.set_title(lbl, fontsize=9)
        s = per_param_stats[i]
        ax.text(0.03, 0.97,
                 f"mean={s['mean']:+.2f}\nstd={s['std']:.2f}\n|z|<1:{s['cov_1sigma_pct']:.0f}% (68)",
                 transform=ax.transAxes, va="top", ha="left", fontsize=7.5,
                 family="monospace",
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
        if i == 0:
            ax.legend(loc="upper right", fontsize=7, frameon=False)
    for j in range(k, nrows * ncols):
        fig.delaxes(axes.flat[j])

    summary = {"per_param": per_param_stats}

    if compute_joint:
        try:
            from scipy.stats import chi2
        except ImportError:
            print("[joint diagnostic skipped — scipy not installed]")
            fig.suptitle("Per-parameter pull — Gaussian approx (moments)",
                          fontsize=11, y=1.0)
            fig.tight_layout()
            return fig, summary

        chi_sq = np.empty(n); chi_sq.fill(np.nan)
        for m in range(n):
            C = 0.5 * (cov_pred_all[m] + cov_pred_all[m].T)
            r = theta_true_all[m] - mean_pred_all[m]
            try:
                L = np.linalg.cholesky(C + joint_jitter * np.eye(k))
                z_full = np.linalg.solve(L, r)
                chi_sq[m] = float(np.sum(z_full ** 2))
            except np.linalg.LinAlgError:
                continue
        chi_finite = chi_sq[np.isfinite(chi_sq)]
        n_ok = chi_finite.size

        q68 = chi2.ppf(0.68, k)
        q95 = chi2.ppf(0.95, k)
        cov68 = 100 * np.mean(chi_finite < q68) if n_ok else np.nan
        cov95 = 100 * np.mean(chi_finite < q95) if n_ok else np.nan

        print()
        print("=" * 72)
        print(f"Joint whitened residual  (χ² with k={k} dof; Cholesky-whitened)")
        print("=" * 72)
        print(f"  n_test with valid Cholesky: {n_ok}/{n}")
        print(f"  mean ‖z_full‖²  = {chi_finite.mean():>7.2f}   (expected: {k})")
        print(f"  median          = {np.median(chi_finite):>7.2f}   (expected: {chi2.ppf(0.5, k):.2f})")
        print(f"  frac < χ²_0.68 = {q68:5.2f} : {cov68:>5.1f}%   (expected 68%)")
        print(f"  frac < χ²_0.95 = {q95:5.2f} : {cov95:>5.1f}%   (expected 95%)")

        summary["joint"] = dict(
            n_valid=n_ok, mean=float(chi_finite.mean()) if n_ok else np.nan,
            median=float(np.median(chi_finite)) if n_ok else np.nan,
            cov_68_pct=cov68, cov_95_pct=cov95, k_dof=k,
            expected_mean=k, expected_median=float(chi2.ppf(0.5, k)),
        )

    fig.suptitle("Per-parameter pull — Gaussian approx (moments); not a Gaussianity test",
                  fontsize=10, y=1.0)
    fig.tight_layout()
    return fig, summary
