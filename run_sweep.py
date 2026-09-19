"""Full observable-pair sweep in one continuous run: train, featurize, plot.

For every unordered pair of observables in the HDF5 file (14 observables ->
91 pairs), train one SimpleMLP per noise case (7 cases, see noise_cases_for),
compute aligned + shuffle-test R², save the models, extract a feature vector
per (pair, parameter), and write the consolidated summary figure for every
parameter. The run ends by collecting everything into analysis-ready tables.

    python run_sweep.py                        # all 91 pairs
    python run_sweep.py --epochs 5 --pairs 0   # smoke test

Resumable: a pair whose models/<pair>.pt exists is loaded instead of retrained
(features and plots are regenerated), so resubmitting a timed-out job
continues into the same output. --overwrite forces retraining.

Output layout (--out, default ./sweep_output):
    models/<pair>.pt            state_dicts, losses, noise cases, R² matrices
    tables/<pair>/*.csv         aligned / shuffled R², long-format dual table
    features/<pair>.csv         one feature row per parameter (src/features.py)
    plots/<pair>/*.png          heatmaps, loss curves, per-parameter summaries
    plots/summary/*.png         cross-pair heatmaps
    features_all.csv            every (pair, parameter) row + path to its figure
    feature_columns.csv         one-line meaning of every feature column
    run_config.json             hyperparameters, data file, git commit, timing

Downstream (dimensionality reduction, clustering) starts from
features_all.csv and is not run here.

Design notes (see CLAUDE.md): noise is added to the normalized observable and
NOT renormalized; noise is resampled every epoch; val/test inputs are clean.
Case names use A = observable_1 (alphabetically first), B = observable_2.
"""
import os
import sys
import time
import json
import argparse
import subprocess
import textwrap
import contextlib
from itertools import combinations

import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
import models
import train
import pipeline
import plots
import features as features_mod

plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"

# Which of the 35 parameters are log-transformed before standardization
LOGFLAG_MASK = np.array([False, False, True, True, True, True, False, False, False, True, True,
                         False, False, True, False, True, False, True, True, False, False, True,
                         True, True, True, True, True, False, True, False, True, False, False,
                         False, True])
N_SIMS = 1024
N_SHUFFLE_PERMS = 10
# Searched in order when --data is not given: the copy shipped with the repo
# first, then the layout used in the original working tree.
DATA_CANDIDATES = ("data/data_L50_TNG_v3.hdf5", "../DATA/data_L50_TNG_v3.hdf5")


def resolve_data(path):
    """Return the data file, with an actionable message if it isn't there."""
    if path:
        if not os.path.exists(path):
            raise SystemExit(f"--data file not found: {path}")
        return path
    for cand in DATA_CANDIDATES:
        full = os.path.join(HERE, cand)
        if os.path.exists(full):
            return full
    raise SystemExit(
        "Training data not found. Looked for:\n  "
        + "\n  ".join(os.path.join(HERE, c) for c in DATA_CANDIDATES)
        + "\nPass --data /path/to/data_L50_TNG_v3.hdf5")


def noise_cases_for(obs1, obs2):
    """7 cases per pair: noise sweep through both-clean + 2 single-observable refs.
    Name format 'B_<noise on obs2>_A_<noise on obs1>'."""
    return {
        "B_5.0_A_0.0": {obs2: 5.0, obs1: 0.0},
        "B_2.5_A_0.0": {obs2: 2.5, obs1: 0.0},
        "B_0.0_A_0.0": {obs2: 0.0, obs1: 0.0},
        "B_0.0_A_2.5": {obs2: 0.0, obs1: 2.5},
        "B_0.0_A_5.0": {obs2: 0.0, obs1: 5.0},
        "B_clean": {obs2: 0.0},
        "A_clean": {obs1: 0.0},
    }


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Data + split (shared by every pair)
# ---------------------------------------------------------------------------

def load_data(datafile):
    with h5py.File(datafile, "r") as f:
        params = f["Parameters"][:, :N_SIMS].T
        # every per-sim array except Parameters; bin-center arrays are filtered out by shape
        obs = {k: f[k][:].T.astype(np.float32) for k in sorted(f.keys())
               if k != "Parameters" and f[k].shape[-1] == params.shape[0]}
    logflag = LOGFLAG_MASK[:params.shape[1]]
    if not np.all(params[:, logflag] > 0):
        raise ValueError("Non-positive value in logflag columns")
    p_log = params.copy()
    p_log[:, logflag] = np.log(p_log[:, logflag])
    means, stds = p_log.mean(axis=0), p_log.std(axis=0)
    y = torch.from_numpy((p_log - means) / stds).float()
    x_norm = {k: pipeline.normalize(v) for k, v in obs.items()}
    return y, logflag, means, stds, obs, x_norm


def make_split(n, val_fraction, test_fraction, seed):
    """test = final reporting + shuffle test; val = best-weights selection."""
    n_val, n_test = int(n * val_fraction), int(n * test_fraction)
    torch.manual_seed(seed)
    np.random.seed(seed)
    split_perm = torch.randperm(n)
    idx_test = split_perm[:n_test]
    idx_val = split_perm[n_test:n_test + n_val]
    idx_train = split_perm[n_test + n_val:]
    perm = np.random.permutation(n_test)
    return idx_train, idx_val, idx_test, perm


# ---------------------------------------------------------------------------
# Per-pair: configure, train, evaluate, save
# ---------------------------------------------------------------------------

def configure_modules(D, obs1, obs2, noise_cases, all_results=None, r2=None):
    x_norm = {k: D["x_norm"][k] for k in (obs1, obs2)}
    x_raw = {k: D["x_raw"][k] for k in (obs1, obs2)}
    a = D["args"]
    pipeline.configure(
        observable_1=obs1, observable_2=obs2,
        x_normalized_dict=x_norm, x_raw_dict=x_raw,
        y=D["y"], idx_val=D["idx_val"], idx_test=D["idx_test"], idx_train=D["idx_train"],
        batch_size=a.eval_batch_size, device=D["device"],
        logflag=D["logflag"], means=D["means"], stds=D["stds"], output_dim=D["output_dim"],
        hidden_dims=a.hidden_dims, dropout_rate=a.dropout, epochs=a.epochs,
        perm=D["perm"], all_results=all_results,
    )
    r2 = r2 or {}
    plots.configure(
        all_results=all_results, output_dim=D["output_dim"],
        observable_1=obs1, observable_2=obs2,
        logflag=D["logflag"], means=D["means"], stds=D["stds"],
        x_normalized_dict=x_norm, y=D["y"], idx_val=D["idx_val"], idx_test=D["idx_test"],
        param_names=D["param_names"], noise_cases=noise_cases,
        batch_size=a.eval_batch_size, device=D["device"], perm=D["perm"],
        r2_matrix=r2.get("aligned"),
        r2_matrix_shifted_observable_only=r2.get("shuf_obs2"),
        r2_matrix_shifted_both=r2.get("shuf_obs1"),
    )


def case_tensors(D, selected):
    """Concatenated (train, val) inputs and the per-column noise std for one case.

    Columns follow sorted(selected) — the same layout the eval loaders build.
    Everything moves to the device once and stays there for training.
    """
    dev = D["device"]
    tr, va, noise = [], [], []
    for key in sorted(selected):
        arr = D["x_norm"][key]
        tr.append(torch.from_numpy(arr[D["idx_train"]]).float())
        va.append(torch.from_numpy(arr[D["idx_val"]]).float())
        noise.append(torch.full((arr.shape[1],), float(selected[key])))
    return torch.cat(tr, 1).to(dev), torch.cat(va, 1).to(dev), torch.cat(noise).to(dev)


def train_pair(D, noise_cases):
    a, dev = D["args"], D["device"]
    torch.manual_seed(a.seed)  # same init stream for every pair regardless of run order
    y_train = D["y"][D["idx_train"]].to(dev)
    y_val = D["y"][D["idx_val"]].to(dev)
    all_results = []
    for i, (case_name, sel) in enumerate(noise_cases.items()):
        t0 = time.time()
        x_tr, x_va, noise_std = case_tensors(D, sel)
        model = models.SimpleMLP(x_tr.shape[1], a.hidden_dims, D["output_dim"], a.dropout).to(dev)
        tl, vl = train.fit_with_epoch_noise(
            model=model, x_train=x_tr, y_train=y_train, x_val=x_va, y_val=y_val,
            noise_std=noise_std,
            optimizer=optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.wd),
            criterion=nn.MSELoss(), epochs=a.epochs, batch_size=a.batch_size,
            restore_best_weights=True, best_weights_smoothing_window=50,
            log_every=a.log_every)
        all_results.append({"case_name": case_name, "selected_observables": sel,
                            "model": model, "train_losses": tl, "val_losses": vl})
        print(f"    [{i+1}/{len(noise_cases)}] {case_name:12s} {(time.time()-t0)/60:5.1f} min  "
              f"min val loss={min(vl):.4f}", flush=True)
    return all_results


def evaluate_pair(D, all_results):
    """Aligned R² + both shuffle modes (averaged over N_SHUFFLE_PERMS perms), on the test set."""
    r2 = {"aligned": np.zeros((len(all_results), D["output_dim"]))}
    for ri, r in enumerate(all_results):
        p, t = pipeline.get_case_predictions(r, mode="aligned")
        r2["aligned"][ri] = r2_score(t, p, multioutput="raw_values")
    rng = np.random.default_rng(0)
    perms = [rng.permutation(len(D["idx_test"])) for _ in range(N_SHUFFLE_PERMS)]
    # S1: shuffle obs2 -> survives only if the model reads obs1 (and vice versa)
    r2["shuf_obs2"], r2["shuf_obs2_std"] = pipeline.average_r2_over_perms("obs1_vs_truth", perms)
    r2["shuf_obs1"], r2["shuf_obs1_std"] = pipeline.average_r2_over_perms("obs2_vs_truth", perms)
    return r2


def save_pair(path, obs1, obs2, noise_cases, all_results, r2, args):
    torch.save({
        "obs1": obs1, "obs2": obs2, "noise_cases": noise_cases,
        "cases": [{"case_name": r["case_name"],
                   "selected_observables": r["selected_observables"],
                   "state_dict": {k: v.cpu() for k, v in r["model"].state_dict().items()},
                   "train_losses": r["train_losses"], "val_losses": r["val_losses"]}
                  for r in all_results],
        "r2": r2,
        "hparams": {k: getattr(args, k) for k in
                    ("epochs", "batch_size", "hidden_dims", "dropout", "wd", "lr",
                     "val_fraction", "test_fraction", "seed")},
    }, path)


def load_pair(D, path):
    saved = torch.load(path, weights_only=False, map_location="cpu")
    hp = saved["hparams"]
    all_results = []
    for c in saved["cases"]:
        input_dim = sum(D["x_raw"][k].shape[1] for k in c["selected_observables"])
        model = models.SimpleMLP(input_dim, hp["hidden_dims"], D["output_dim"], hp["dropout"])
        model.load_state_dict(c["state_dict"])
        model.to(D["device"]).eval()
        all_results.append({**{k: c[k] for k in ("case_name", "selected_observables",
                                                 "train_losses", "val_losses")},
                            "model": model})
    return saved, all_results


def write_tables(tdir, cases, param_names, r2):
    os.makedirs(tdir, exist_ok=True)
    for key, fname in [("aligned", "r2_aligned"), ("shuf_obs2", "r2_shuffled_obs2"),
                       ("shuf_obs1", "r2_shuffled_obs1")]:
        pd.DataFrame(r2[key], index=cases, columns=param_names).to_csv(
            os.path.join(tdir, fname + ".csv"))
    dual = dual_r2_df(cases, param_names, r2)
    dual.to_csv(os.path.join(tdir, "dual_r2.csv"), index=False)
    return dual


def dual_r2_df(cases, param_names, r2):
    """Long format for plots.plot_param_curve_dual. Column names follow the
    shuffle-mode naming: r2_shuf_obs1 = S1 (obs1_vs_truth, obs2 shuffled),
    r2_shuf_obs2 = S2 (obs2_vs_truth, obs1 shuffled)."""
    rows = []
    for i, c in enumerate(cases):
        for j, p in enumerate(param_names):
            rows.append({"case": c, "param": p,
                         "r2_aligned": r2["aligned"][i, j],
                         "r2_shuf_obs1": r2["shuf_obs2"][i, j],
                         "r2_shuf_obs2": r2["shuf_obs1"][i, j],
                         "r2_shuf_obs1_std": r2["shuf_obs2_std"][i, j],
                         "r2_shuf_obs2_std": r2["shuf_obs1_std"][i, j]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def display_names(obs1, obs2, cases):
    """'B_2.5_A_0.0' -> 'SFR=2.5, Mg=0.0' using the real observable names."""
    out = {}
    for c in cases:
        if c == "A_clean":
            out[c] = f"{obs1} alone"
        elif c == "B_clean":
            out[c] = f"{obs2} alone"
        else:
            _, b, _, a = c.split("_")
            out[c] = f"{obs2}={b}, {obs1}={a}"
    return out


def relabel(fig, names):
    """Swap structural case names for display names in legends, ticks, titles.
    Longest keys first so 'B_clean' isn't clobbered by a shorter match."""
    keys = sorted(names, key=len, reverse=True)

    def sub(s):
        for k in keys:
            s = s.replace(k, names[k])
        return s
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is not None:
            for t in leg.get_texts():
                t.set_text(sub(t.get_text()))
        ticks = [t.get_text() for t in ax.get_xticklabels()]
        if any(sub(t) != t for t in ticks):
            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels([sub(t) for t in ticks], rotation=45, ha="right", fontsize=8)
        ax.set_title(sub(ax.get_title()))
    return fig


def save(fig, path, names=None, dpi=110):
    if names:
        relabel(fig, names)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close("all")


@contextlib.contextmanager
def onto_axes(fig, axes):
    """Redirect the plt.subplots() call inside each plots.* function onto the
    next axis of a shared grid, and mute its plt.show()."""
    it = iter(np.ravel(axes))
    orig_subplots, orig_show = plt.subplots, plt.show
    plt.subplots = lambda *a, **k: (fig, next(it))
    plt.show = lambda *a, **k: None
    try:
        yield
    finally:
        plt.subplots, plt.show = orig_subplots, orig_show


def heatmap(df, title, path, **kw):
    fig = plt.figure(figsize=(max(16, 0.45 * df.shape[1]), max(4, 0.45 * df.shape[0] + 2)))
    opts = dict(annot=df.size <= 400, fmt=".2f", cmap="Spectral", linewidths=0.2)
    opts.update(kw)
    sns.heatmap(df, **opts)
    plt.title(title)
    plt.tight_layout()
    save(fig, path)


def consolidated_summary(dual, param):
    """Dual R²-vs-noise curve, attractor map, both shuffle scatters, both
    pair-truth scatters — one 3×2 figure. The scatter panels reuse the cached
    chimera grids, so after the first parameter the cost is rendering only."""
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    with onto_axes(fig, axes):
        plots.plot_param_curve_dual(dual, param, show_band=True)
        plots.plot_prediction_attractor_map(param=param)
        for mode in ("obs1_vs_truth", "obs2_vs_truth"):
            plots.plot_pair_normalized_shuffle_scatter(
                param=param, n_pairs="all", case=None, mode=mode, color_by_theta_diff=True)
        for anchor in ("obs1", "obs2"):
            plots.plot_param_pair_truth_scatter(param, anchor_obs=anchor, n_pairs="all")
    for ax in fig.axes:  # long per-panel titles collide in the grid
        ax.set_title(textwrap.fill(ax.get_title(), 60), fontsize=9)
        for coll in ax.collections:  # keep the ~5k-point scatters light in the PNG
            coll.set_rasterized(True)
    fig.suptitle(f"{param} — consolidated summary", fontsize=13, y=1.005)
    fig.tight_layout()
    return fig


def plot_pair(D, obs1, obs2, noise_cases, all_results, r2, dual, pdir, params):
    os.makedirs(pdir, exist_ok=True)
    cases = list(noise_cases)
    names = display_names(obs1, obs2, cases)
    labels = [names[c] for c in cases]
    pn = D["param_names"]
    tag = f"{obs1} × {obs2}"

    # Loss curves — train is on noisy input, val on clean, so they aren't comparable
    fig, axes = plt.subplots(1, len(all_results), figsize=(3.2 * len(all_results), 3), sharey=True)
    for ax, r in zip(axes, all_results):
        ax.plot(r["train_losses"], label="train (noisy input)", lw=0.8)
        ax.plot(r["val_losses"], label="val (clean input)", lw=0.8)
        ax.set_title(names[r["case_name"]], fontsize=8)
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
    axes[0].set_ylabel("MSE (normalized)")
    axes[0].legend(fontsize=7)
    fig.suptitle(tag)
    fig.tight_layout()
    save(fig, os.path.join(pdir, "00_loss_curves.png"))

    # Heatmaps (ΔR² = shifted − aligned; negative = information lost)
    kw = dict(vmin=-1, vmax=1)
    heatmap(pd.DataFrame(r2["aligned"], labels, pn), f"aligned R²  |  {tag}",
            os.path.join(pdir, "01_r2_aligned.png"), **kw)
    heatmap(pd.DataFrame(r2["shuf_obs2"], labels, pn),
            f"R², {obs2} shuffled (tests {obs1})  |  {tag}",
            os.path.join(pdir, "02_r2_shuffled_obs2.png"), **kw)
    heatmap(pd.DataFrame(r2["shuf_obs1"], labels, pn),
            f"R², {obs1} shuffled (tests {obs2})  |  {tag}",
            os.path.join(pdir, "03_r2_shuffled_obs1.png"), **kw)
    kw = dict(vmin=-0.5, center=0.0, linewidths=0.3)
    heatmap(pd.DataFrame(r2["shuf_obs2"] - r2["aligned"], labels, pn),
            f"ΔR², {obs2} shuffled  |  {tag}",
            os.path.join(pdir, "04_delta_r2_shuffled_obs2.png"), **kw)
    heatmap(pd.DataFrame(r2["shuf_obs1"] - r2["aligned"], labels, pn),
            f"ΔR², {obs1} shuffled  |  {tag}",
            os.path.join(pdir, "05_delta_r2_shuffled_obs1.png"), **kw)

    # Consolidated summary per parameter — one figure per row of features/<pair>.csv
    for p in params:
        try:
            consolidated_summary(dual, p)
            save(plt.gcf(), os.path.join(pdir, f"{p}_summary.png"), names)
        except Exception as e:
            plt.close("all")
            print(f"    [warn] summary {p}: {type(e).__name__}: {e}", flush=True)


# ---------------------------------------------------------------------------
# Cross-pair summary
# ---------------------------------------------------------------------------

def plot_summary(D, pair_list, out):
    """Pairs × params tables: both-clean R², gain from combining, shuffle drops."""
    pn = D["param_names"]
    both, gain, shuf2, shuf1, single = {}, {}, {}, {}, {}
    for obs1, obs2 in pair_list:
        path = os.path.join(out, "models", f"{obs1}__{obs2}.pt")
        if not os.path.exists(path):
            continue
        saved = torch.load(path, weights_only=False, map_location="cpu")
        cases = [c["case_name"] for c in saved["cases"]]
        r2 = saved["r2"]["aligned"]
        bc = r2[cases.index("B_0.0_A_0.0")]
        ra, rb = r2[cases.index("A_clean")], r2[cases.index("B_clean")]
        key = f"{obs1} × {obs2}"
        both[key] = bc
        gain[key] = bc - np.maximum(ra, rb)
        shuf2[key] = saved["r2"]["shuf_obs2"][cases.index("B_0.0_A_0.0")] - bc
        shuf1[key] = saved["r2"]["shuf_obs1"][cases.index("B_0.0_A_0.0")] - bc
        single.setdefault(obs1, []).append(ra)
        single.setdefault(obs2, []).append(rb)
    if not both:
        print("[summary] no trained pairs found")
        return

    tdir, sdir = os.path.join(out, "tables"), os.path.join(out, "plots", "summary")
    os.makedirs(tdir, exist_ok=True)
    os.makedirs(sdir, exist_ok=True)
    tables = {
        "r2_both_clean": (both, "aligned R², both observables clean", dict(vmin=-1, vmax=1)),
        "combination_gain": (gain, "R²(both clean) − max(R² single)",
                             dict(center=0.0, cmap="RdBu_r")),
        "delta_r2_shuffled_obs2_both_clean": (shuf2, "ΔR², observable_2 shuffled (both-clean case)",
                                              dict(vmin=-1, center=0.0)),
        "delta_r2_shuffled_obs1_both_clean": (shuf1, "ΔR², observable_1 shuffled (both-clean case)",
                                              dict(vmin=-1, center=0.0)),
    }
    for name, (d, title, kw) in tables.items():
        df = pd.DataFrame.from_dict(d, orient="index", columns=pn)
        df.to_csv(os.path.join(tdir, f"summary_{name}.csv"))
        heatmap(df, title, os.path.join(sdir, f"{name}.png"), **kw)

    # Single-observable R² (each observable appears in several pairs; mean + spread)
    s_mean = pd.DataFrame({k: np.mean(v, axis=0) for k, v in single.items()}, index=pn).T
    pd.DataFrame({k: np.std(v, axis=0) for k, v in single.items()}, index=pn).T.to_csv(
        os.path.join(tdir, "summary_r2_single_observable_std.csv"))
    s_mean.to_csv(os.path.join(tdir, "summary_r2_single_observable_mean.csv"))
    heatmap(s_mean, "single-observable R² (mean over pairs containing it)",
            os.path.join(sdir, "r2_single_observable.png"), vmin=-1, vmax=1)

    # Observable × observable, averaged over params: where does combining help most?
    obs = sorted(single)
    for name, d in [("r2_both_clean", both), ("combination_gain", gain)]:
        mat = pd.DataFrame(np.nan, index=obs, columns=obs)
        for key, v in d.items():
            a, b = key.split(" × ")
            mat.loc[a, b] = mat.loc[b, a] = float(np.mean(v))
        heatmap(mat, f"{name}, mean over parameters", os.path.join(sdir, f"{name}_obs_matrix.png"),
                annot=True, center=0.0 if name == "combination_gain" else None,
                cmap="RdBu_r" if name == "combination_gain" else "Spectral")
    print(f"[summary] {len(both)} pairs -> {sdir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None,
                    help=f"HDF5 training data (default: first of {', '.join(DATA_CANDIDATES)})")
    ap.add_argument("--out", default=os.path.join(HERE, "sweep_output"))
    ap.add_argument("--pairs", type=int, nargs="*", default=None,
                    help="pair indices to run (default: all); printed at startup")
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--eval-batch-size", type=int, default=4096,
                    help="batch size for the n² chimera forward passes")
    ap.add_argument("--hidden-dims", type=int, nargs="+", default=[128, 64])
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--test-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=500, help="epoch logging interval (0 = off)")
    ap.add_argument("--overwrite", action="store_true", help="retrain pairs that have saved models")
    ap.add_argument("--threads", type=int, default=None,
                    help="torch CPU threads (default: SLURM allocation)")
    return ap.parse_args(argv)


def git_commit():
    try:
        return subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def collect_features(out, pair_list, param_names):
    """Stack every pair's feature CSV into one analysis-ready table, with the
    path of each row's summary figure so downstream results map back to plots."""
    frames = []
    for obs1, obs2 in pair_list:
        path = os.path.join(out, "features", f"{obs1}__{obs2}.csv")
        if os.path.exists(path):
            frames.append(pd.read_csv(path))
    if not frames:
        print("[features] no feature files found")
        return
    df = pd.concat(frames, ignore_index=True)
    df.insert(4, "figure", [os.path.join("plots", pr, f"{pa}_summary.png")
                            for pr, pa in zip(df["pair"], df["param"])])
    df.to_csv(os.path.join(out, "features_all.csv"), index=False)
    features_mod.describe_columns(df.columns).to_csv(
        os.path.join(out, "feature_columns.csv"), index=False)
    n_feat = df.select_dtypes(include=[np.number]).shape[1]
    print(f"[features] features_all.csv: {len(df)} rows "
          f"({df['pair'].nunique()} pairs × {len(param_names)} params), {n_feat} numeric features")


def main(argv=None):
    args = parse_args(argv)
    threads = args.threads or int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or None
    if threads:
        torch.set_num_threads(threads)

    t_start = time.time()
    device = get_device()
    args.data = resolve_data(args.data)
    y, logflag, means, stds, x_raw, x_norm = load_data(args.data)
    idx_train, idx_val, idx_test, perm = make_split(len(y), args.val_fraction,
                                                    args.test_fraction, args.seed)
    D = dict(args=args, device=device, y=y, logflag=logflag, means=means, stds=stds,
             x_raw=x_raw, x_norm=x_norm, output_dim=y.shape[1],
             param_names=[f"θ{j}" for j in range(y.shape[1])],
             idx_train=idx_train, idx_val=idx_val, idx_test=idx_test, perm=perm)

    all_pairs = list(combinations(sorted(x_raw), 2))   # sorted -> (obs1, obs2)
    todo = all_pairs if args.pairs is None else [all_pairs[i] for i in args.pairs]
    print(f"data={args.data}")
    print(f"device={device}  threads={threads or 'default'}  sims={len(y)}  "
          f"train/val/test={len(idx_train)}/{len(idx_val)}/{len(idx_test)}")
    print(f"{len(x_raw)} observables -> {len(all_pairs)} pairs; running {len(todo)}")
    for sub in ("models", "tables", "features", "plots"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    failed = []
    for n_i, (obs1, obs2) in enumerate(todo, 1):
        pair = f"{obs1}__{obs2}"
        mpath = os.path.join(args.out, "models", pair + ".pt")
        t0 = time.time()
        print(f"\n=== [{n_i}/{len(todo)}] {obs1} × {obs2}  ({time.strftime('%H:%M:%S')}) ===",
              flush=True)
        try:
            noise_cases = noise_cases_for(obs1, obs2)
            if os.path.exists(mpath) and not args.overwrite:
                saved, all_results = load_pair(D, mpath)
                noise_cases, r2 = saved["noise_cases"], saved["r2"]
                configure_modules(D, obs1, obs2, noise_cases, all_results, r2)
                print("    loaded saved models (--overwrite to retrain)")
            else:
                configure_modules(D, obs1, obs2, noise_cases)
                all_results = train_pair(D, noise_cases)
                configure_modules(D, obs1, obs2, noise_cases, all_results)
                r2 = evaluate_pair(D, all_results)
                configure_modules(D, obs1, obs2, noise_cases, all_results, r2)
                save_pair(mpath, obs1, obs2, noise_cases, all_results, r2, args)
                print(f"    saved {mpath}")

            dual = write_tables(os.path.join(args.out, "tables", pair),
                                list(noise_cases), D["param_names"], r2)

            # features before plots: it warms the chimera-grid cache the plots reuse
            feats = features_mod.pair_features(all_results, r2, D["param_names"], obs1, obs2)
            feats.to_csv(os.path.join(args.out, "features", pair + ".csv"), index=False)

            plot_pair(D, obs1, obs2, noise_cases, all_results, r2, dual,
                      os.path.join(args.out, "plots", pair), D["param_names"])
            print(f"    features + {len(D['param_names'])} summaries in "
                  f"{(time.time()-t0)/60:.1f} min total", flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            failed.append(pair)
            plt.close("all")
            print(f"    [FAIL] {pair}: {type(e).__name__}: {e}", flush=True)

    plot_summary(D, all_pairs, args.out)
    collect_features(args.out, all_pairs, D["param_names"])

    hours = (time.time() - t_start) / 3600
    with open(os.path.join(args.out, "run_config.json"), "w") as f:
        json.dump({**{k: v for k, v in vars(args).items()},
                   "git_commit": git_commit(), "device": str(device),
                   "n_pairs": len(all_pairs), "pairs_run": len(todo),
                   "failed_pairs": failed, "hours": round(hours, 3),
                   "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
    print(f"\nFinished in {hours:.2f} h. Failed pairs: {failed or 'none'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
