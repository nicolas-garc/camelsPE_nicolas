"""Helpers for loading a saved pair's results.pt and getting a live plotting env.

Usage in a notebook:
    from _analyze_helpers import load_pair, pretty_case_names
    ctx = load_pair(pair_idx=0)             # or pair_dir="pair_01_..."
    # ctx.plots.plot_predictions_vs_true("θ4", ctx.cases[0])
    # ctx.plots.plot_marginal_posterior_grid("θ4", cases=ctx.cases)  # only if moment run

The load also re-registers observable names into pipeline + plots modules so all
plot functions work as they did during training. `pretty_case_names(ctx)` returns
a dict mapping stored case names (e.g. 'B_1.0_A_0.0') to human-readable strings
using the actual observable names (e.g. 'SFR=1.0, Mg=0.0').
"""
import os
import sys
import glob
import re
import types
import numpy as np
import h5py
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = "/Users/nicolasgarcia/Downloads/GAL_SBI/camelsPE/src"
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import pipeline
import plots

# Local copy of _load_data so we don't import _pair_pipeline (which forces
# matplotlib.use("Agg") at module import, breaking interactive backends in
# Jupyter). Must stay in sync with _pair_pipeline._load_data.
DATAFILE = "/Users/nicolasgarcia/Downloads/GAL_SBI/DATA/data_L50_TNG_v3.hdf5"
LOGFLAG_MASK = np.array([False, False, True, True, True, True, False, False, False, True, True,
                         False, False, True, False, True, False, True, True, False, False, True,
                         True, True, True, True, True, False, True, False, True, False, False,
                         False, True])


def _load_data():
    with h5py.File(DATAFILE, "r") as f:
        Parameters = f["Parameters"][:, :1024].T
    logflag = LOGFLAG_MASK[:Parameters.shape[1]]
    if not np.all(Parameters[:, logflag] > 0):
        raise ValueError("Non-positive value in logflag columns")
    Pll = Parameters.copy()
    Pll[:, logflag] = np.log(Pll[:, logflag])
    means = Pll.mean(axis=0)
    stds = Pll.std(axis=0)
    Pnorm = (Pll - means) / stds
    n_sims = Pnorm.shape[0]
    with h5py.File(DATAFILE, "r") as f:
        observable_block = {
            key: torch.from_numpy(f[key][:].T).float()
            for key in sorted(f.keys())
            if key != "Parameters" and f[key].shape[-1] == n_sims
        }
    y = torch.from_numpy(Pnorm).float()
    return y, logflag, means, stds, observable_block


def list_pairs():
    """Return sorted list of pair directory names in sweep/."""
    dirs = sorted(glob.glob(os.path.join(_HERE, "pair_*")))
    return [os.path.basename(d) for d in dirs]


def _short_obs_name(name):
    """First token of an observable name (before first '_'). 'Mg_Mh_s90' -> 'Mg'."""
    return name.split("_")[0]


def pretty_case_names(saved):
    """Return {stored_name: 'display_name'} for the pair's case dict.

    Structural case keys look like 'B_val_A_val' or 'A_clean'/'B_clean' where
    A = observable_1 (alphabetically first) and B = observable_2. This function
    substitutes the real short observable names.
    """
    obs1, obs2 = saved["obs1"], saved["obs2"]
    A_short = _short_obs_name(obs1)
    B_short = _short_obs_name(obs2)
    out = {}
    for name in saved["noise_cases"].keys():
        if name == "A_clean":
            out[name] = f"{A_short} alone"
        elif name == "B_clean":
            out[name] = f"{B_short} alone"
        else:
            # Expected 'B_<b_noise>_A_<a_noise>'
            m = re.match(r"^B_([0-9.]+)_A_([0-9.]+)$", name)
            if m:
                b_val, a_val = m.group(1), m.group(2)
                out[name] = f"{B_short}={b_val}, {A_short}={a_val}"
            else:
                out[name] = name  # unknown pattern → passthrough
    return out


class PairContext(types.SimpleNamespace):
    """Container for a loaded pair. Fields:
      obs_a, obs_b, obs1, obs2, all_results, cases, r2_matrix, ...
      noise_cases, pipeline, plots, display_names, pair_dir, results_dir
    """
    def summary(self):
        A, B = _short_obs_name(self.obs1), _short_obs_name(self.obs2)
        print(f"Pair: {self.obs_a}  ×  {self.obs_b}")
        print(f"  observable_1 (A) = {self.obs1}")
        print(f"  observable_2 (B) = {self.obs2}")
        print(f"  {len(self.cases)} cases:")
        for c in self.cases:
            print(f"    {c:20s}  →  {self.display_names.get(c, c)}")
        print(f"  Focus params for plots: θ0, θ1, θ2, θ4, θ7, θ11 (adjust as needed)")


def load_pair(pair_idx=None, pair_dir=None, results_path=None,
              val_fraction=None, test_fraction=None, seed=None):
    """Load a saved pair's results.pt and reconfigure pipeline + plots modules.

    Provide one of:
      pair_idx: 0-based index into list_pairs()
      pair_dir: directory name like "pair_01_Mg_Mh_s90__Rs_Ms_s90"
      results_path: full path to results.pt

    val_fraction/test_fraction/seed default to whatever was saved in results.pt
    (falling back to 0.1/0.1/0 for older files without the metadata).

    Returns a PairContext. All plots from plots.py are then callable.
    """
    if results_path is None:
        if pair_dir is None:
            pairs = list_pairs()
            if pair_idx is None:
                pair_idx = 0
            if pair_idx < 0 or pair_idx >= len(pairs):
                raise IndexError(f"pair_idx {pair_idx} out of range [0,{len(pairs)})")
            pair_dir = pairs[pair_idx]
        results_path = os.path.join(_HERE, pair_dir, "results", "results.pt")
    if not os.path.exists(results_path):
        raise FileNotFoundError(results_path)
    saved = torch.load(results_path, weights_only=False)

    # Rebuild the shared data — must match the training-time split
    y, logflag, means, stds, observable_block = _load_data()
    val_fraction = val_fraction if val_fraction is not None else saved.get("val_fraction", 0.1)
    test_fraction = test_fraction if test_fraction is not None else saved.get("test_fraction", 0.0)
    seed = seed if seed is not None else saved.get("seed", 0)
    n_val = int(len(y) * val_fraction)
    n_test = int(len(y) * test_fraction)
    torch.manual_seed(seed); np.random.seed(seed)
    split_perm = torch.randperm(len(y))
    if n_test > 0:
        idx_test  = split_perm[:n_test]
        idx_val   = split_perm[n_test:n_test + n_val]
        idx_train = split_perm[n_test + n_val:]
        perm = np.random.permutation(len(idx_test))
    else:
        # Older 2-way-split results.pt (test_fraction=0). Preserve original layout.
        idx_test  = None
        idx_train = split_perm[:-n_val]
        idx_val   = split_perm[-n_val:]
        perm = np.random.permutation(len(idx_val))

    obs1, obs2 = saved["obs1"], saved["obs2"]
    x_raw_dict = {k: observable_block[k].numpy() for k in (obs1, obs2)}
    x_normalized_dict = {k: pipeline.normalize(observable_block[k].numpy()) for k in (obs1, obs2)}

    # Guess device from whichever trained module the case happens to have.
    # SBI results carry "sbi_density_estimator" instead of "model"; hetero has "model".
    all_results = saved["all_results"]
    _r0 = all_results[0]
    _mod = None
    for _k in ("model", "hetero_model", "moment_model", "sbi_density_estimator"):
        if _k in _r0 and hasattr(_r0[_k], "parameters"):
            _mod = _r0[_k]; break
    device = next(_mod.parameters()).device if _mod is not None else torch.device("cpu")
    hidden_dims = [128, 64]
    dropout_rate = 0.2
    batch_size = 64
    output_dim = y.shape[1]
    param_names = [f"θ{j}" for j in range(output_dim)]

    pipeline.configure(
        observable_1=obs1, observable_2=obs2,
        x_normalized_dict=x_normalized_dict, x_raw_dict=x_raw_dict,
        y=y, idx_val=idx_val, idx_test=idx_test, idx_train=idx_train,
        batch_size=batch_size, device=device,
        logflag=logflag, means=means, stds=stds, output_dim=output_dim,
        hidden_dims=hidden_dims, dropout_rate=dropout_rate, epochs=750,
        perm=perm, all_results=all_results,
    )

    # r2_matrix is optional — SBI results don't compute it (no F* mean net)
    r2_matrix = saved.get("r2_matrix")
    plots.configure(
        all_results=all_results, output_dim=output_dim,
        observable_1=obs1, observable_2=obs2,
        logflag=logflag, means=means, stds=stds,
        x_normalized_dict=x_normalized_dict, y=y,
        idx_val=idx_val, idx_test=idx_test,
        param_names=param_names, noise_cases=saved["noise_cases"],
        batch_size=batch_size, device=device, perm=perm,
        r2_matrix=r2_matrix,
        r2_matrix_shifted_observable_only=saved.get("r2_matrix_shifted_observable_only"),
        r2_matrix_shifted_both=saved.get("r2_matrix_shifted_both"),
    )

    ctx = PairContext(
        obs_a=saved["obs_a"], obs_b=saved["obs_b"],
        obs1=obs1, obs2=obs2,
        noise_cases=saved["noise_cases"],
        cases=list(saved["noise_cases"].keys()),
        all_results=all_results,
        r2_matrix=r2_matrix,
        r2_matrix_shifted_obs=saved.get("r2_matrix_shifted_observable_only"),
        r2_matrix_shifted_both=saved.get("r2_matrix_shifted_both"),
        r2_std_matrix_shifted_obs=saved.get("r2_std_matrix_observable_only"),
        r2_std_matrix_shifted_both=saved.get("r2_std_matrix_both"),
        param_names=param_names,
        display_names=pretty_case_names(saved),
        pair_dir=os.path.dirname(results_path),
        results_dir=os.path.dirname(results_path),
        pipeline=pipeline,
        plots=plots,
        has_moment=any("moment_model" in r for r in all_results),
        # backward-compat aliases so existing notebook cells that reference the
        # old flags keep working
        has_variance=any("moment_model" in r for r in all_results),
        has_covariance=any("moment_model" in r for r in all_results),
    )
    return ctx


def load_all_pairs(include_moment=False):
    """Load every saved pair into a list of PairContexts (in pair-directory order)."""
    ctxs = []
    for d in sorted(glob.glob(os.path.join(_HERE, "pair_*"))):
        rp = os.path.join(d, "results", "results.pt")
        if os.path.exists(rp):
            try:
                ctxs.append(load_pair(results_path=rp))
            except Exception as e:
                print(f"[warn] skipped {rp}: {e}")
    if include_moment:
        rp = os.path.join(_HERE, "moment_example", "results", "results.pt")
        if os.path.exists(rp):
            try:
                ctxs.append(load_pair(results_path=rp))
            except Exception as e:
                print(f"[warn] skipped moment_example: {e}")
    return ctxs


def compare_across_pairs(ctxs, param, *, case_pattern="B_0.0_A_0.0",
                          metric="r2_aligned", figsize=(9, 4.5)):
    """Bar chart of one param's metric across pairs for a chosen case.

    metric: 'r2_aligned' or 'r2_shifted_obs' or 'r2_shifted_both' or 'sigma_median'
    case_pattern: which case per pair to compare (structural name).
    Returns the figure.
    """
    import matplotlib.pyplot as plt
    values = []
    labels = []
    for ctx in ctxs:
        # resolve param index
        if param in ctx.param_names:
            pidx = ctx.param_names.index(param)
        else:
            print(f"[warn] {param} not in {ctx.pair_dir}"); continue
        # resolve case index
        if case_pattern not in ctx.cases:
            print(f"[warn] case {case_pattern!r} not in {ctx.pair_dir}"); continue
        cidx = ctx.cases.index(case_pattern)

        if metric == "r2_aligned":
            v = float(ctx.r2_matrix[cidx, pidx])
        elif metric == "r2_shifted_obs":
            v = float(ctx.r2_matrix_shifted_obs[cidx, pidx]) if ctx.r2_matrix_shifted_obs is not None else np.nan
        elif metric == "r2_shifted_both":
            v = float(ctx.r2_matrix_shifted_both[cidx, pidx]) if ctx.r2_matrix_shifted_both is not None else np.nan
        elif metric == "sigma_median":
            if not ctx.has_variance:
                print(f"[warn] no variance for {ctx.pair_dir}"); continue
            r = ctx.all_results[cidx]
            _, sigma, _ = ctx.pipeline.predict_with_uncertainty(r, space="log_partial")
            v = float(np.median(sigma[:, pidx]))
        else:
            raise ValueError(f"unknown metric {metric!r}")

        values.append(v)
        labels.append(f"{ctx.obs1[:6]}×{ctx.obs2[:6]}")

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(len(values)), values, color="tab:blue")
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(metric)
    ax.set_title(f"{param}  |  case={case_pattern}")
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(0, color="k", linewidth=0.5)
    fig.tight_layout()
    return fig


def build_dual_r2_df(ctx):
    """Long-format DataFrame with per-cell mean + std for both shuffle modes.

    Convenience for plot_param_curve_dual which reads the r2_shuf_{obs1,obs2}_std
    columns to draw ±σ shaded bands. If stds weren't saved in the .pt, they fall
    back to 0.0.
    """
    import pandas as pd
    std_obs = getattr(ctx, "r2_std_matrix_shifted_obs", None)
    std_both = getattr(ctx, "r2_std_matrix_shifted_both", None)
    rows = []
    for i, r in enumerate(ctx.all_results):
        for j, p in enumerate(ctx.param_names):
            rows.append({
                "case": r["case_name"], "param": p,
                "r2_aligned": float(ctx.r2_matrix[i, j]),
                "r2_shuf_obs1": (float(ctx.r2_matrix_shifted_obs[i, j])
                                 if ctx.r2_matrix_shifted_obs is not None else np.nan),
                "r2_shuf_obs2": (float(ctx.r2_matrix_shifted_both[i, j])
                                 if ctx.r2_matrix_shifted_both is not None else np.nan),
                "r2_shuf_obs1_std": float(std_obs[i, j]) if std_obs is not None else 0.0,
                "r2_shuf_obs2_std": float(std_both[i, j]) if std_both is not None else 0.0,
            })
    return pd.DataFrame(rows)


def label_cases(ctx, case_list):
    """Return list of display names for a list of stored case names."""
    return [ctx.display_names.get(c, c) for c in case_list]


def suggest_params(ctx, top_n=6, min_aligned_r2=0.1):
    """Rank parameters by heatmap 'interestingness'. Print + return dict of lists.

    Three rankings — a parameter interesting under any of them is worth plotting:

      well_constrained     : max_case aligned R² is high  (worth studying at all)
      information_sharing  : max_case |max ΔR²| is large  (shuffle actually kills R²)
      shuffle_asymmetry    : |ΔR²_obs2_shuf − ΔR²_obs1_shuf| is large per case,
                             maxed across cases  (obs1 vs obs2 contribute differently
                             → candidates for Case 3/4/6 distinctions)

    Requires ctx.r2_matrix_shifted_obs and ctx.r2_matrix_shifted_both to be non-None
    for the second and third rankings.
    """
    n_cases, n_params = ctx.r2_matrix.shape

    max_aligned = ctx.r2_matrix.max(axis=0)   # [n_params]
    ranked_wc = np.argsort(-max_aligned)      # descending
    ranked_wc = [i for i in ranked_wc if max_aligned[i] >= min_aligned_r2]

    result = {
        "well_constrained": [(ctx.param_names[i], float(max_aligned[i]))
                             for i in ranked_wc[:top_n]],
    }

    if ctx.r2_matrix_shifted_obs is not None and ctx.r2_matrix_shifted_both is not None:
        delta_obs2 = ctx.r2_matrix_shifted_obs - ctx.r2_matrix   # negative = info lost when obs2 shuffled
        delta_obs1 = ctx.r2_matrix_shifted_both - ctx.r2_matrix
        # Information sharing: strongest drop under either shuffle, worst-case per param
        max_drop = np.minimum(delta_obs2.min(axis=0), delta_obs1.min(axis=0))
        ranked_is = np.argsort(max_drop)   # ascending = most-negative first
        result["information_sharing"] = [
            (ctx.param_names[i], float(max_drop[i])) for i in ranked_is[:top_n]
            if max_drop[i] < -0.05
        ]
        # Shuffle asymmetry per (case, param); take max abs across cases per param
        asym = np.abs(delta_obs2 - delta_obs1)  # [n_cases, n_params]
        max_asym = asym.max(axis=0)
        ranked_as = np.argsort(-max_asym)
        result["shuffle_asymmetry"] = [
            (ctx.param_names[i], float(max_asym[i])) for i in ranked_as[:top_n]
            if max_asym[i] > 0.05
        ]

    # Print
    print(f"Suggested parameters (top {top_n}):")
    print(f"  well-constrained   (highest max-case aligned R²):")
    for name, v in result["well_constrained"]:
        print(f"    {name:6s}  {v:+.2f}")
    if "information_sharing" in result:
        print(f"  information-sharing (biggest R² drop under best shuffle):")
        for name, v in result["information_sharing"]:
            print(f"    {name:6s}  ΔR² = {v:+.2f}")
        print(f"  shuffle-asymmetric  (obs1-shuf vs obs2-shuf disagree most):")
        for name, v in result["shuffle_asymmetry"]:
            print(f"    {name:6s}  |Δ diff| = {v:+.2f}")
    # A convenient union — dedup, preserve order
    all_names = []
    for cat in ("well_constrained", "information_sharing", "shuffle_asymmetry"):
        for name, _ in result.get(cat, []):
            if name not in all_names:
                all_names.append(name)
    result["union"] = all_names
    print(f"\n  Union of the above (paste into FOCUS_PARAMS):")
    print(f"    FOCUS_PARAMS = {all_names}")
    return result


def relabel_figure(ctx, fig=None, ax=None):
    """Post-process a plot: rewrite legend labels + x-tick labels + title
    substituting display case names for structural ones. Call after any plot.
    """
    import matplotlib.pyplot as plt
    if fig is None:
        fig = plt.gcf()
    for a in fig.axes:
        # legend labels
        leg = a.get_legend()
        if leg is not None:
            for t in leg.get_texts():
                for k, v in ctx.display_names.items():
                    if k in t.get_text():
                        t.set_text(t.get_text().replace(k, v))
        # x-tick labels
        labels = [t.get_text() for t in a.get_xticklabels()]
        if any(k in lab for k in ctx.display_names for lab in labels):
            new_labels = []
            for lab in labels:
                new_lab = lab
                for k, v in ctx.display_names.items():
                    if k in new_lab:
                        new_lab = new_lab.replace(k, v)
                new_labels.append(new_lab)
            a.set_xticklabels(new_labels, rotation=45, ha="right", fontsize=9)
        # title
        title = a.get_title()
        for k, v in ctx.display_names.items():
            if k in title:
                title = title.replace(k, v)
        a.set_title(title)
    fig.tight_layout()
    return fig
