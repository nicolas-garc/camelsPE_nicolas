"""Regenerate the same plot set for every saved pair (+ moment example) using the
real observable names in place of the structural A/B tokens. No retraining —
loads each pair's results.pt.

Run:
    python3 analyze_all.py                 # all 4 pairs + moment_example
    python3 analyze_all.py pair_02_...     # one pair
    python3 analyze_all.py --skip-moment   # sweep pairs only

Outputs land in each pair's results/plots/ dir, prefixed 're_' to avoid
overwriting the training-time plots.
"""
import os
import sys
import glob
import time
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from _analyze_helpers import load_pair, list_pairs, relabel_figure, label_cases, build_dual_r2_df

FOCUS_PARAMS = ("θ0", "θ1", "θ2", "θ4", "θ7", "θ11")


def _save(fig, out_dir, name):
    p = os.path.join(out_dir, name + ".png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved {p}")


def _r2_heatmaps(ctx, out_dir):
    labels = label_cases(ctx, ctx.cases)
    for arr, title, fname in [
        (ctx.r2_matrix, "aligned R²", "re_01_r2_aligned"),
        (ctx.r2_matrix_shifted_obs, "R² — obs2 shuffled", "re_02_r2_shuffled_obs2"),
        (ctx.r2_matrix_shifted_both, "R² — obs1 shuffled", "re_03_r2_shuffled_obs1"),
    ]:
        if arr is None:
            continue
        df = pd.DataFrame(arr, index=labels, columns=ctx.param_names)
        fig = plt.figure(figsize=(16, 5))
        sns.heatmap(df, vmin=-1, vmax=1, annot=True, fmt=".2f", cmap="Spectral",
                    linewidths=0.2, cbar_kws={"label": title})
        plt.title(f"{title}  |  {ctx.obs1}  ×  {ctx.obs2}")
        plt.tight_layout()
        _save(fig, out_dir, fname)

    # ΔR²
    for shifted, name in [
        (ctx.r2_matrix_shifted_obs, "re_04_delta_r2_shuffled_obs2"),
        (ctx.r2_matrix_shifted_both, "re_05_delta_r2_shuffled_obs1"),
    ]:
        if shifted is None:
            continue
        delta = shifted - ctx.r2_matrix
        df = pd.DataFrame(delta, index=labels, columns=ctx.param_names)
        fig = plt.figure(figsize=(16, 5))
        sns.heatmap(df, center=0.0, vmin=-0.5, annot=True, fmt=".2f", cmap="Spectral",
                    linewidths=0.3, cbar_kws={"label": "ΔR² shifted − aligned"})
        plt.title(f"ΔR² — {name}  |  {ctx.obs1}  ×  {ctx.obs2}")
        plt.tight_layout()
        _save(fig, out_dir, name)


def _param_all_val(ctx, out_dir):
    for p in FOCUS_PARAMS:
        try:
            ctx.plots.plot_param_all_val(p, mode="aligned")
            relabel_figure(ctx)
            _save(plt.gcf(), out_dir, f"re_06_param_all_val_{p}")
        except Exception as e:
            print(f"    [warn] plot_param_all_val({p}) failed: {e}")


def _dual_curves(ctx, out_dir):
    if ctx.r2_matrix_shifted_obs is None or ctx.r2_matrix_shifted_both is None:
        return
    dual_r2_df = build_dual_r2_df(ctx)   # real ±σ bands from 10-perm averaging
    for p in FOCUS_PARAMS:
        try:
            ctx.plots.plot_param_curve_dual(dual_r2_df, p, figsize=(8, 5))
            relabel_figure(ctx)
            _save(plt.gcf(), out_dir, f"re_07_param_curve_dual_{p}")
        except Exception as e:
            print(f"    [warn] plot_param_curve_dual({p}) failed: {e}")


def _moment_plots(ctx, out_dir):
    """Summary-only moment-network plots — no per-sim clutter.

    Layout:
      10  predicted-μ ± σ vs true, per param              (n_focus files)
      11  σ_median by case per param — bar chart          (1 file)
      12  pull calibration histogram, per case            (1 file)
      13  median correlation matrix per case              (1 file)
      20  corner plot, all cases overlaid — one rep sim   (1 file)
      21  prediction-scatter corner per case              (n_cases files)
    """
    if not getattr(ctx, "has_moment", False) and not ctx.has_variance:
        return

    # 10. predicted μ ± σ vs true, per focus param — the "does it work?" plot
    for p in FOCUS_PARAMS:
        try:
            ctx.plots.plot_predictions_with_errorbars(p, cases=ctx.cases, space="log_partial")
            relabel_figure(ctx)
            _save(plt.gcf(), out_dir, f"re_10_predictions_with_errorbars_{p}")
        except Exception as e:
            print(f"    [warn] predictions_with_errorbars({p}): {e}")

    # 11. Median σ bar chart
    try:
        ctx.plots.plot_sigma_by_case_bars(list(FOCUS_PARAMS), ctx.cases,
                                           space="log_partial", reducer="median")
        relabel_figure(ctx)
        _save(plt.gcf(), out_dir, "re_11_sigma_by_case")
    except Exception as e:
        print(f"    [warn] sigma_by_case: {e}")

    # 12. Pull calibration
    try:
        ctx.plots.plot_pull_distribution(ctx.cases, space="normalized",
                                          params=list(FOCUS_PARAMS))
        relabel_figure(ctx)
        _save(plt.gcf(), out_dir, "re_12_pull_distribution")
    except Exception as e:
        print(f"    [warn] pull: {e}")

    # 13. Median correlation per case
    try:
        ctx.plots.plot_median_correlation_grid(ctx.cases, space="log_partial")
        _save(plt.gcf(), out_dir, "re_13_median_correlation_grid")
    except Exception as e:
        print(f"    [warn] median_correlation_grid: {e}")

    # 20. Corner plot — all cases overlaid, one representative sim
    try:
        ctx.plots.plot_corner(sim_idx=0, cases=ctx.cases, space="log_partial")
        relabel_figure(ctx)
        _save(plt.gcf(), out_dir, "re_20_corner_sim0")
    except Exception as e:
        print(f"    [warn] corner(sim=0): {e}")

    # 21. Prediction-scatter corner — one per case
    for c in ctx.cases:
        try:
            ctx.plots.plot_prediction_scatter_corner(c, space="log_partial")
            _save(plt.gcf(), out_dir, f"re_21_scatter_corner_{c}")
        except Exception as e:
            print(f"    [warn] scatter_corner({c}): {e}")


def analyze_one(results_path):
    t0 = time.time()
    print(f"\n=== {results_path} ===")
    ctx = load_pair(results_path=results_path)
    ctx.summary()

    out_dir = os.path.join(ctx.results_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    # Write a legend file so anyone browsing plots knows what the display names mean
    with open(os.path.join(out_dir, "_case_legend.txt"), "w") as f:
        f.write(f"observable_1 (A) = {ctx.obs1}\n")
        f.write(f"observable_2 (B) = {ctx.obs2}\n\n")
        for stored, display in ctx.display_names.items():
            f.write(f"  {stored:20s}  →  {display}\n")

    _r2_heatmaps(ctx, out_dir)
    _param_all_val(ctx, out_dir)
    _dual_curves(ctx, out_dir)
    _moment_plots(ctx, out_dir)

    print(f"    done in {time.time() - t0:.1f} s")


def _discover(skip_moment):
    targets = []
    for d in sorted(glob.glob(os.path.join(_HERE, "pair_*"))):
        rp = os.path.join(d, "results", "results.pt")
        if os.path.exists(rp):
            targets.append(rp)
    if not skip_moment:
        rp = os.path.join(_HERE, "moment_example", "results", "results.pt")
        if os.path.exists(rp):
            targets.append(rp)
    return targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subset", nargs="*",
                    help="Optional pair directory names or paths to results.pt. Default: all pairs + moment.")
    ap.add_argument("--skip-moment", action="store_true", help="Skip moment_example.")
    args = ap.parse_args()

    if args.subset:
        targets = []
        for s in args.subset:
            if os.path.isfile(s):
                targets.append(s)
            elif os.path.isdir(os.path.join(_HERE, s)):
                targets.append(os.path.join(_HERE, s, "results", "results.pt"))
            else:
                print(f"[warn] cannot resolve {s!r} — skipping")
    else:
        targets = _discover(args.skip_moment)

    if not targets:
        print("No results.pt found. Did you run the sweep yet?")
        return 1

    print(f"Analyzing {len(targets)} run(s):")
    for t in targets:
        print(f"  {t}")

    for t in targets:
        try:
            analyze_one(t)
        except Exception as e:
            print(f"[fail] {t}: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()

    return 0


if __name__ == "__main__":
    sys.exit(main())
