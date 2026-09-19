"""
Plot functions for the CAMELS SBI noise-mixing experiment.

Call configure() once after training to register shared state
(run_sweep.py does this per pair):

    import pipeline, plots
    pipeline.configure(observable_1=..., ...)
    plots.configure(all_results=all_results, output_dim=output_dim, ...)

Then all plot functions work with just the parameter name:

    fig, stats = plots.plot_prediction_attractor_map(param="θ4")
"""
import sys as _sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

from pipeline import (
    get_case_predictions, make_pair_val_loader_fn,
    resolve_shuffle, sample_unique_unordered_pairs,
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

    # choose space for normalization/plotting: "processed" (default, the
    # model's own training space -- mean 0/std 1), "log" (processed
    # un-standardized; gives the SAME p_norm ratio as "processed" since
    # standardization is affine and cancels out of a ratio-of-differences,
    # so this only exists for callers who want log-space axis units), or
    # "physical" (exp() applied to logflag params too).
    space="processed",

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

    The 'space' argument controls which space pred/t0/t1 are computed in:
    "processed" (default, model's training space), "log", or "physical"
    (exp applied to logflag params). "processed" and "log" give the same
    p_norm ratio (standardization cancels out of it); "physical" differs
    for logflag parameters.
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


def plot_prediction_attractor_map(param,
                                    *,
                                    case_sequence=None,
                                    obs_pair=None,
                                    space="processed",
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

    space: "processed" (default) bins/averages in the model's own training
    space (mean 0, std 1) -- every parameter is uniformly distributed there
    (see the parameter_distributions notebook), so bins stay evenly
    populated regardless of whether this parameter is logflagged, and
    y = mean(true) collapses to the trivial y = 0 line. "log" or "physical"
    bin/average in those spaces instead -- physical space in particular
    reintroduces the skew that motivated switching to "processed": bins are
    equal-WIDTH, not equal-count, so a long-tailed logflagged parameter
    packs most sims into a few low bins and leaves the tail sparse (bins
    under min_per_bin silently drop out as gaps in the curve).

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
        preds, trues = get_case_predictions(case_to_result[c], mode="aligned", perm=perm, space=space)
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
    ax.set_xlabel(f"True {p_label} ({space} space)")
    ax.set_ylabel(f"Mean predicted {p_label} per bin")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best", framealpha=0.9)

    # "collapsed into one view" and the line-color/ordering explainer dropped
    # -- the legend already names each case, that was just clutter.
    pair_str = f"{obs_pair[0]} ↔ {obs_pair[1]}" if obs_pair else "unknown pair"
    ax.set_title(f"Attractor map — {p_label}   |   {pair_str}", fontsize=11)

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


def _compute_all_chimera_preds(result, mode, x_dict, eval_idx_, p_idx, batch_size, device_, space="physical"):
    """All chimera predictions for one case + one parameter.

    Returns a (n_val, n_val) matrix where entry [j, k] is the model's
    prediction (in the requested space, matching get_case_predictions) when
    the input uses:
      - row j's KEPT observable(s), and
      - row k's SHUFFLED observable(s)

    space: "processed" (raw model output), "log" (processed un-standardized),
    or "physical" (log undone too, via exp() when this parameter is
    logflagged) -- same convention and same module-level stds/means/logflag
    as get_case_predictions.

    j == k rows are still computed; the caller filters them if unwanted.
    Column layout in the model input matches the loader: features are
    concatenated in sorted(selected_observables.keys()) order. Space
    conversion here matches get_case_predictions bit-for-bit at j == k
    (aligned rows) -- PROVIDED the caller passes the same evaluation indices
    get_case_predictions uses internally (_eval_idx(), i.e. idx_test if
    configured else idx_val). Passing plain idx_val when idx_test is a
    different, disjoint set silently pairs each row against the wrong
    simulation's truth everywhere downstream.
    """
    selected = result["selected_observables"]
    sel_keys_sorted = sorted(selected.keys())
    shuffle_keys, _ = resolve_shuffle(selected, mode)
    shuffle_set = set(shuffle_keys)

    idx_arr = np.asarray(eval_idx_)
    n = len(idx_arr)

    # For each observable in sorted order, build (n*n, feat_dim) with the
    # convention that flat row (j*n + k) holds:
    #   arr[j] if the observable is KEPT
    #   arr[k] if the observable is SHUFFLED
    # np.repeat(arr, n, axis=0) → [arr[0]*n, arr[1]*n, ...]      → row j·n+k = arr[j]
    # np.tile(arr,   (n, 1))    → [arr[0..n-1], arr[0..n-1], ...] → row j·n+k = arr[k]
    # The forward pass is parameter-independent (it produces all output_dim
    # columns), so cache the whole grid on the result and slice per parameter —
    # otherwise every one of the 35 params repeats the same n² forward pass.
    cache_key = ("_chimera_grid_v1", mode, tuple(idx_arr.tolist()))
    if cache_key not in result:
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
        result[cache_key] = torch.cat(outs, dim=0).numpy()  # (n*n, output_dim), processed space
    preds_flat = result[cache_key]

    pred_p = preds_flat[:, p_idx]
    if space != "processed":
        pred_p = pred_p * stds[p_idx] + means[p_idx]
        if space == "physical":
            if logflag[p_idx]:
                pred_p = np.exp(pred_p)
        elif space != "log":
            raise ValueError("space must be 'processed', 'log', or 'physical'.")
    return pred_p.reshape(n, n)


def plot_pair_normalized_shuffle_scatter(param,
                                            *,
                                            case=None,
                                            space="processed",
                                            mode="obs1_vs_truth",
                                            x_pred_source="aligned",
                                            obs_pair=None,
                                            normalize_endpoints="anchor_to_partner",
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
                                            color_by_theta_diff=False,
                                            theta_diff_cmap="RdBu_r",
                                            theta_diff_clip_quantile=0.98,
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
    normalize_endpoints (anchor_to_partner by default) picks which ROLE maps
    to the 0-endpoint: anchor_to_partner → anchor→0, partner→1;
    partner_to_anchor → the reverse. This is deliberately named by ROLE, not
    by observable: unlike plot_param_pair_normalized_values (where obs1/obs2
    ARE the sim1/sim2 labels), here `mode` decides which OBSERVABLE the
    anchor/partner roles each supply, so "the obs1-supplying sim" is the
    anchor under mode="obs1_vs_truth" but the PARTNER under
    mode="obs2_vs_truth" -- an "obs1_to_obs2"-style name would silently mean
    different things depending on `mode`. anchor_to_partner avoids that: it
    always means anchor→0 regardless of mode, which is also why x (built
    from the anchor's own prediction) comes out identical for both modes.

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

    color_by_theta_diff: if True, colors each dot by the SIGNED truth
    difference θ_sim2 − θ_sim1 (t1_arr − t0_arr, i.e. the same pair-distance
    that normalizes x/y, but signed rather than absolute) using
    theta_diff_cmap (a diverging map, symmetric about 0) clipped to the
    theta_diff_clip_quantile of |θ_sim2 − θ_sim1| so a few extreme pairs
    don't wash out the color scale for the rest.

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
    preds_a, true_a = get_case_predictions(result, mode="aligned", perm=perm, space=space)
    pa = preds_a[:, p_idx]                # θ̂_aligned per anchor row (fixed)
    truth_base = true_a[:, p_idx]         # truth per row
    n_val = len(pa)

    # --- build pair arrays (anchors, partners, pred_shuf_per_pair) per n_pairs mode ---
    if n_pairs is None:
        # Single-perm mode (original behaviour). One pair per anchor.
        preds_s, true_s = get_case_predictions(result, mode=mode, perm=perm, space=space)
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
            preds_k, _ = get_case_predictions(result, mode=mode, perm=pk, space=space)
            shuf_chunks.append(preds_k[:, p_idx])
        pred_shuf_arr = np.concatenate(shuf_chunks)
        n_pairs_desc = f"{K} perms × {n_val} rows = {K * n_val}"
    elif isinstance(n_pairs, str) and n_pairs == "all":
        chimera = _compute_all_chimera_preds(
            result, mode, x_normalized_dict, _eval_idx(), p_idx,
            batch_size=all_pairs_batch_size, device_=device, space=space,
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
    # normalize_endpoints picks which ROLE maps to the 0-endpoint (not which
    # observable -- that's `mode`'s job; see docstring for why these must stay
    # decoupled).
    if normalize_endpoints == "anchor_to_partner":
        t0_arr, t1_arr = truth_j, truth_k
        pred_sim1_arr, pred_sim2_arr = pa_arr, pa_partner_arr
        endpoint_desc = (f"sim_1 = anchor, supplies {anchor_obs_key} (→0);  "
                          f"sim_2 = partner, supplies {partner_obs_key} (→1)")
    elif normalize_endpoints == "partner_to_anchor":
        t0_arr, t1_arr = truth_k, truth_j
        pred_sim1_arr, pred_sim2_arr = pa_partner_arr, pa_arr
        endpoint_desc = (f"sim_1 = partner, supplies {partner_obs_key} (→0);  "
                          f"sim_2 = anchor, supplies {anchor_obs_key} (→1)")
    else:
        raise ValueError("normalize_endpoints must be 'anchor_to_partner' or 'partner_to_anchor'.")

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
                zorder=1, label="y = x  →  swap changed nothing")

    # --- name endpoints/predictions by observable, not generic sim1/sim2 ---
    # sim1/sim2 mean "whichever sim supplies the 0/1 endpoint" per
    # normalize_endpoints -- see docstring for how that maps to anchor/partner.
    # short_of(key) identifies a SIM by the observable that anchors it (e.g.
    # "Mg" for Mg_Mh_s61) -- theta_Mg below means "this parameter's truth at
    # the sim supplying Mg", not the Mg observable's own reading.
    def short_of(obs_key):
        return obs_key.split("_")[0]

    obs_for_sim1 = anchor_obs_key if normalize_endpoints == "anchor_to_partner" else partner_obs_key
    obs_for_sim2 = partner_obs_key if normalize_endpoints == "anchor_to_partner" else anchor_obs_key
    short1, short2 = short_of(obs_for_sim1), short_of(obs_for_sim2)
    short_anchor, short_partner = short_of(anchor_obs_key), short_of(partner_obs_key)

    if color_by_theta_diff:
        theta_diff = (t1_arr - t0_arr)[keep]   # signed: θ_{short2} - θ_{short1}
        vlim = float(np.quantile(np.abs(theta_diff), theta_diff_clip_quantile)) if n_kept else 1.0
        vlim = max(vlim, 1e-12)
        sc = ax.scatter(x, y, s=marker_size, alpha=marker_alpha, c=theta_diff,
                         cmap=theta_diff_cmap, vmin=-vlim, vmax=vlim,
                         edgecolor="none", zorder=3, label=f"{n_kept} pairs")
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(f"θ_{short2} − θ_{short1}  ({p_label} truth, signed)", fontsize=8)
    else:
        ax.scatter(x, y, s=marker_size, alpha=marker_alpha, color="#3C3489",
                   edgecolor="none", zorder=3, label=f"{n_kept} pairs")

    ax.set_xlim(*clip_range)
    ax.set_ylim(*clip_range)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)

    # Explicit equations, spelled out with the same observable-named theta_X
    # subscripts as the title/colorbar/annotations -- no bare theta_1/theta_2.
    x_eq = (rf"$x = (\hat\theta_{{\mathrm{{pred}}}} - \theta_{{\mathrm{{{short1}}}}})"
            rf"\,/\,|\theta_{{\mathrm{{{short2}}}}} - \theta_{{\mathrm{{{short1}}}}}|$")
    y_eq = (rf"$y = (\hat\theta_{{\mathrm{{shuffled}}}} - \theta_{{\mathrm{{{short1}}}}})"
            rf"\,/\,|\theta_{{\mathrm{{{short2}}}}} - \theta_{{\mathrm{{{short1}}}}}|$")

    x_desc = {"aligned": f"prediction from {short_anchor}-sim's data",
              "sim1": f"prediction from {short1}-sim's data",
              "sim2": f"prediction from {short2}-sim's data"}[x_pred_source]
    ax.set_xlabel(f"{x_eq}\n{x_desc}\n(0 = θ_{short1} truth, 1 = θ_{short2} truth)", fontsize=9)
    ax.set_ylabel(f"{y_eq}\nprediction with {short_partner} swapped\n"
                  f"(0 = θ_{short1} truth, 1 = θ_{short2} truth)", fontsize=9)

    # Endpoint mapping (0=.../1=...) deliberately left off the title -- it's
    # already on both axis labels, repeating it here was just clutter.
    ax.set_title(
        f"{p_label}  —  shuffle test: keep {short_anchor}, swap in {short_partner}"
        f"   |   case: {case}   |   n={n_kept}",
        fontsize=9.5)
    # Pinned bottom-right (not "best"): the annotation box below is hardcoded
    # top-left, and "best" occasionally also picks top-left when that corner
    # is data-sparse, colliding with it.
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)

    # Kept minimal on purpose: the diagonal's own legend entry already covers
    # "y=x -> no effect"; only show caveats that apply to this specific call.
    annot_lines = [f"y = ±1  →  θ_shuffled matches θ_{short2} truth"]
    if x_pred_source != "aligned":
        annot_lines.append(
            f"x_pred_source={x_pred_source!r}: same-anchor x only repeats if "
            "this resolves to the anchor's own prediction (see docstring)"
        )
    if drop_degenerate_pairs and n_dropped:
        annot_lines.append(
            f"dropped {n_dropped}/{n_total} degenerate pairs "
            f"(|θ_{short2}−θ_{short1}| ≤ {threshold:.3g}, i.e. {min_pair_distance_frac*100:g}% of {p_label}'s range)"
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
        "n_pairs_desc": n_pairs_desc,
        "endpoint_desc": endpoint_desc,
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


def plot_param_pair_truth_scatter(param,
                                    *,
                                    anchor_obs="obs1",
                                    obs1_key=None,
                                    obs2_key=None,
                                    case=None,
                                    space="physical",
                                    n_pairs="all",
                                    pair_seed=0,
                                    all_pairs_batch_size=4096,
                                    diff_cmap="RdBu_r",
                                    diff_clip_quantile=0.98,
                                    marker_alpha=None,
                                    marker_size=None,
                                    star_size = 20,
                                    show_diagonal_line=True,
                                    show_aligned_points=True,
                                    results=None,
                                    x_dict=None,
                                    y_vector=None,
                                    idx=None,
                                    batch_size=None,
                                    param_labels=None,
                                    device=None,
                                    figsize=(7.5, 7.0),
                                    save_path=None):
    """
    Chimera predictions vs. the anchor sim's truth, over every unique pair
    of validation simulations.

    anchor_obs picks WHICH observable the anchor (x-axis) sim keeps intact:
      - "obs1" (default): anchor keeps obs1_key, partner supplies obs2_key.
      - "obs2": anchor keeps obs2_key, partner supplies obs1_key.
    Call this function twice (once per anchor_obs) to see both directions
    for one parameter -- side_by_side_pair_truth_scatter in the analysis
    notebooks does exactly that. Same anchor/partner-role vocabulary as
    plot_pair_normalized_shuffle_scatter: "anchor" is always the smaller-
    index validation sim in an unordered pair {i, j} (i < j); "partner" is
    the other one. anchor_obs only decides which observable each supplies.

    For unordered pair {i, j} (i < j, same convention as
    sample_unique_unordered_pairs / make_pair_val_loader_fn):
        x = θ_anchor        (anchor sim's true value)
        y = model prediction on the chimera input (anchor keeps anchor_obs,
            partner supplies the other observable)
        color = θ_partner − θ_anchor   (signed — NOT |θ_partner − θ_anchor|)
    The aligned pair (i, i) — obs1 AND obs2 both from the same sim, i.e. the
    model's ordinary own prediction — is included too (same for both
    anchor_obs values) and drawn as a marker; it always has color 0 by
    construction (θ_i − θ_i = 0).

    y = x is the reference: predictions that land exactly on the anchor's
    truth.

    n_pairs: "all" (default) computes every i<j pair via one batched n×n
    forward pass (n = eval-set size, ~100 here, so n² is cheap) and plots
    all C(n,2) of them. An int instead randomly subsamples that many
    off-diagonal pairs (seeded by pair_seed) for a lighter-weight plot; the
    n×n forward pass still runs once since it's needed to know the diagonal
    (and is shared/cached across both anchor_obs calls).

    Case: `case` defaults to the both-clean combo case for the observable
    pair (auto-detected via parse_case_name), same convention as
    plot_pair_normalized_shuffle_scatter.

    Chimera caveat: off-diagonal points feed the model observables from two
    different simulations — no such galaxy exists in the training
    distribution. This is a saliency probe ("which channel does the output
    track?"), not posterior inference — see the chimera caveat in CLAUDE.md.

    Returns (fig, stats).
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

    if space not in ("processed", "log", "physical"):
        raise ValueError("space must be 'processed', 'log', or 'physical'.")

    if anchor_obs not in ("obs1", "obs2"):
        raise ValueError(f"anchor_obs must be 'obs1' or 'obs2', not {anchor_obs!r}.")

    # --- resolve case (default: both-clean combo for this obs pair) ---
    obs_set = {obs1_key, obs2_key}
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
                    f"{sorted(obs_set)}. Both-clean combos found with other "
                    f"observables: {both_clean_candidates}. "
                    f"Pass `case` explicitly or adjust obs1_key/obs2_key."
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
            f"Case {case!r} is missing {sorted(missing)}; the chimera needs "
            f"both {obs1_key} and {obs2_key} present."
        )

    idx_arr = np.asarray(idx)
    n_val = len(idx_arr)

    # --- full n_val x n_val chimera grid: one batched forward pass ---
    # Row i*n_val+j takes obs1_key from sim i and obs2_key from sim j; any
    # other observable in the case (there normally isn't one, for these
    # 2-observable pair sweeps) comes from sim i, matching
    # make_pair_val_loader_fn's convention. [i, i] reduces to the ordinary
    # aligned prediction for sim i.
    cache_key = ("_pair_truth_grid_v1", case, obs1_key, obs2_key, tuple(idx_arr.tolist()))
    if cache_key not in result:
        sel_keys_sorted = sorted(result["selected_observables"].keys())
        cols = []
        for key in sel_keys_sorted:
            arr = x_dict[key][idx_arr]
            if key == obs2_key:
                expanded = np.tile(arr, (n_val, 1))       # row i*n+j -> arr[j]
            else:
                expanded = np.repeat(arr, n_val, axis=0)  # row i*n+j -> arr[i]
            cols.append(torch.from_numpy(expanded).float())
        x_all = torch.cat(cols, dim=1)

        model = result["model"].to(device)
        model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, x_all.shape[0], all_pairs_batch_size):
                outs.append(model(x_all[i:i + all_pairs_batch_size].to(device)).cpu())
        result[cache_key] = torch.cat(outs, dim=0).numpy()  # (n_val*n_val, output_dim), processed space

    preds_flat = result[cache_key]

    def _convert_space(arr):
        arr = np.array(arr, copy=True)
        if space == "processed":
            return arr
        arr = arr * stds + means
        if space == "log":
            return arr
        arr[:, logflag] = np.exp(arr[:, logflag])
        return arr

    pred_grid = _convert_space(preds_flat)[:, p_idx].reshape(n_val, n_val)  # [i, j]

    truth_raw = y_vector[idx_arr]
    if hasattr(truth_raw, "numpy"):
        truth_raw = truth_raw.numpy()
    truth = _convert_space(np.asarray(truth_raw, dtype=np.float64))[:, p_idx]

    # --- off-diagonal unique pairs i<j ---
    i_all, j_all = np.triu_indices(n_val, k=1)
    if isinstance(n_pairs, str) and n_pairs == "all":
        i_off, j_off = i_all, j_all
        n_pairs_desc = f"all unique pairs = {n_val}x{n_val - 1}/2 = {len(i_all)}"
    elif isinstance(n_pairs, int) and not isinstance(n_pairs, bool):
        if n_pairs < 1:
            raise ValueError(f"n_pairs must be >= 1 (got {n_pairs}).")
        n_avail = len(i_all)
        if n_pairs > n_avail:
            raise ValueError(f"n_pairs={n_pairs} exceeds C(n_val,2)={n_avail}.")
        rng_ = np.random.default_rng(pair_seed)
        sel = rng_.choice(n_avail, size=n_pairs, replace=False)
        i_off, j_off = i_all[sel], j_all[sel]
        n_pairs_desc = f"{n_pairs} sampled pairs (of {n_avail})"
    else:
        raise ValueError(f"n_pairs must be 'all' or a positive int, got {n_pairs!r}.")

    # anchor is always the smaller-index sim i; anchor_obs picks which
    # observable it keeps. pred_grid[row, col] = obs1 from row, obs2 from
    # col, so "obs1" anchor reads pred_grid[i, j] (anchor keeps obs1) and
    # "obs2" anchor reads pred_grid[j, i] (anchor keeps obs2, partner j
    # supplies obs1) -- both already sit in the one n_val x n_val grid
    # computed above, so switching anchor_obs is free (no extra forward pass).
    if anchor_obs == "obs1":
        anchor_obs_key, partner_obs_key = obs1_key, obs2_key
        y_off = pred_grid[i_off, j_off]
    else:
        anchor_obs_key, partner_obs_key = obs2_key, obs1_key
        y_off = pred_grid[j_off, i_off]

    def short_of(obs_key):
        return obs_key.split("_")[0]

    short_anchor, short_partner = short_of(anchor_obs_key), short_of(partner_obs_key)

    x_off = truth[i_off]
    diff_off = truth[j_off] - truth[i_off]     # signed: θ_partner - θ_anchor, NOT absolute

    x_diag = truth
    y_diag = pred_grid[np.arange(n_val), np.arange(n_val)]
    n_off = len(x_off)

    # --- auto-scale marker style from point count ---
    if marker_alpha is None:
        if n_off <= 200:      marker_alpha = 0.85
        elif n_off <= 2000:   marker_alpha = 0.55
        else:                 marker_alpha = 0.35
    if marker_size is None:
        if n_off <= 200:      marker_size = 26
        elif n_off <= 2000:   marker_size = 10
        else:                 marker_size = 5

    # --- plot ---
    fig, ax = plt.subplots(figsize=figsize)

    lo = float(min(truth.min(), y_off.min() if n_off else truth.min(), y_diag.min()))
    hi = float(max(truth.max(), y_off.max() if n_off else truth.max(), y_diag.max()))
    pad = 0.03 * (hi - lo if hi > lo else 1.0)
    lo, hi = lo - pad, hi + pad

    if show_diagonal_line:
        ax.plot([lo, hi], [lo, hi], "--", color="#5F5E5A", lw=1.2, alpha=0.6,
                zorder=1, label=f"y = x  →  prediction matches θ_{short_anchor} truth")

    vlim = float(np.quantile(np.abs(diff_off), diff_clip_quantile)) if n_off else 1.0
    vlim = max(vlim, 1e-12)
    sc = ax.scatter(x_off, y_off, s=marker_size, alpha=marker_alpha, c=diff_off,
                     cmap=diff_cmap, vmin=-vlim, vmax=vlim, edgecolor="none",
                     zorder=2, label=f"{n_off} chimera pairs")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"θ_{short_partner} − θ_{short_anchor}  ({p_label} truth, signed)")

    if show_aligned_points:
        ax.scatter(x_diag, y_diag, s=star_size, marker="P", facecolor="#F2C14E",
                   edgecolor="#3C3489", linewidth=0.5, zorder=3,
                   label=f"{n_val} aligned (own data)")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.set_xlabel(f"θ_{short_anchor} truth  ({p_label}, space={space})")
    ax.set_ylabel(f"prediction — keep {short_anchor}, swap in {short_partner}")
    # x-axis-is-anchor-truth and the endpoint mapping are already on the
    # x/y labels -- kept off the title to cut repetition.
    ax.set_title(
        f"{p_label}  —  pair-truth scatter: keep {short_anchor}, swap in {short_partner}"
        f"   |   case: {case}   |   n={n_off}",
        fontsize=10)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    stats = {
        "case": case,
        "anchor_obs": anchor_obs,
        "anchor_obs_key": anchor_obs_key,
        "partner_obs_key": partner_obs_key,
        "obs1_key": obs1_key,
        "obs2_key": obs2_key,
        "space": space,
        "n_val": n_val,
        "n_pairs_desc": n_pairs_desc,
        "n_off_diag": n_off,
        "n_diag": n_val,
        "x_off": x_off,
        "y_off": y_off,
        "diff_off": diff_off,
        "i_off": i_off,
        "j_off": j_off,
        "x_diag": x_diag,
        "y_diag": y_diag,
    }
    return fig, stats
