# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: py311-main
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Interactive analysis of a saved sweep pair
#
# **Workflow**:
# 1. Pick a pair (change `PAIR_IDX`)
# 2. Run the heatmap cells — aligned R², shuffle R², ΔR²
# 3. Run the `suggest_params(ctx)` cell — ranks parameters by three heuristics
# 4. Edit `FOCUS_PARAMS` based on what you saw and want to dig into
# 5. Run the parameter-specific plot cells

# %%
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from _analyze_helpers import load_pair, list_pairs, relabel_figure, label_cases, suggest_params, build_dual_r2_df
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# %% [markdown]
# ## 1. Pick a pair

# %%
print("Available:")
for i, name in enumerate(list_pairs()):
    print(f"  {i}: {name}")
print(f"  moment_example/results/results.pt  (moment network — has σ)")

# %%
PAIR_IDX = 1   # 0..3 for the sweep pairs
# Or load moment_example:
#   ctx = load_pair(results_path="moment_example/results/results.pt")
ctx = load_pair(pair_idx=PAIR_IDX)
ctx.summary()

# %% [markdown]
# ## 2. Heatmaps — look here first
#
# Read them to answer:
# - Which parameters have any signal at all? (aligned R² > threshold in *some* case)
# - Which parameters collapse under shuffling? (large negative ΔR²)
# - Which parameters respond differently to obs1-shuffle vs obs2-shuffle?
#   (asymmetric ΔR² between the two shuffle heatmaps)

# %% [markdown]
# ### 2a. Aligned R² — the baseline (no shuffling)

# %%
labels = label_cases(ctx, ctx.cases)
df = pd.DataFrame(ctx.r2_matrix, index=labels, columns=ctx.param_names)
plt.figure(figsize=(16, 5))
sns.heatmap(df, vmin=-1, vmax=1, annot=True, fmt=".2f", cmap="Spectral",
            linewidths=0.2, cbar_kws={"label": "aligned R²"})
plt.title(f"Aligned R²  |  {ctx.obs1}  ×  {ctx.obs2}")
plt.tight_layout(); plt.show()

# %% [markdown]
# ### 2b. Shuffled R² — obs2 shuffled (survives only if model reads obs1)

# %%
if ctx.r2_matrix_shifted_obs is not None:
    df = pd.DataFrame(ctx.r2_matrix_shifted_obs, index=labels, columns=ctx.param_names)
    plt.figure(figsize=(16, 5))
    sns.heatmap(df, vmin=-1, vmax=1, annot=True, fmt=".2f", cmap="Spectral",
                linewidths=0.2, cbar_kws={"label": "R² (obs2 shuffled)"})
    plt.title(f"obs2 shuffled  |  {ctx.obs1}  ×  {ctx.obs2}")
    plt.tight_layout(); plt.show()

# %% [markdown]
# ### 2c. Shuffled R² — obs1 shuffled (survives only if model reads obs2)

# %%
if ctx.r2_matrix_shifted_both is not None:
    df = pd.DataFrame(ctx.r2_matrix_shifted_both, index=labels, columns=ctx.param_names)
    plt.figure(figsize=(16, 5))
    sns.heatmap(df, vmin=-1, vmax=1, annot=True, fmt=".2f", cmap="Spectral",
                linewidths=0.2, cbar_kws={"label": "R² (obs1 shuffled)"})
    plt.title(f"obs1 shuffled  |  {ctx.obs1}  ×  {ctx.obs2}")
    plt.tight_layout(); plt.show()

# %% [markdown]
# ### 2d. ΔR² — obs2 shuffled (negative = information lost)

# %%
if ctx.r2_matrix_shifted_obs is not None:
    delta = ctx.r2_matrix_shifted_obs - ctx.r2_matrix
    df = pd.DataFrame(delta, index=labels, columns=ctx.param_names)
    plt.figure(figsize=(16, 5))
    sns.heatmap(df, center=0.0, vmin=-0.5, annot=True, fmt=".2f", cmap="Spectral",
                linewidths=0.3, cbar_kws={"label": "ΔR² (obs2 shuffled)"})
    plt.title(f"ΔR² — obs2 shuffled  |  {ctx.obs1}  ×  {ctx.obs2}")
    plt.tight_layout(); plt.show()

# %% [markdown]
# ### 2e. ΔR² — obs1 shuffled

# %%
if ctx.r2_matrix_shifted_both is not None:
    delta = ctx.r2_matrix_shifted_both - ctx.r2_matrix
    df = pd.DataFrame(delta, index=labels, columns=ctx.param_names)
    plt.figure(figsize=(16, 5))
    sns.heatmap(df, center=0.0, vmin=-0.5, annot=True, fmt=".2f", cmap="Spectral",
                linewidths=0.3, cbar_kws={"label": "ΔR² (obs1 shuffled)"})
    plt.title(f"ΔR² — obs1 shuffled  |  {ctx.obs1}  ×  {ctx.obs2}")
    plt.tight_layout(); plt.show()

# %% [markdown]
# ## 3. Ranked parameter suggestions
#
# Three rankings:
#   - **well-constrained**: high aligned R² in any case (worth looking at at all)
#   - **information-sharing**: R² drops sharply under best shuffle (obs matters)
#   - **shuffle-asymmetric**: obs1-shuffle and obs2-shuffle disagree most (differential contribution)
#
# The `union` list combines all three. Paste it (or a subset) into `FOCUS_PARAMS` below.

# %%
suggestions = suggest_params(ctx, top_n=6, min_aligned_r2=0.1)

# %% [markdown]
# ## 4. Set focus params — the ones you want deep plots for
#
# Either paste from `suggestions['union']` above, or handpick from the heatmaps.

# %%
FOCUS_PARAMS = [1,4,7,11,16,34] #suggestions["union"]     # or e.g. ["θ4", "θ7", "θ11"]
print(f"Will make deep plots for: {FOCUS_PARAMS}")

# %% [markdown]
# ## 5. Deep plots — for each param in FOCUS_PARAMS

# %% [markdown]
# ### 5a. Predictions overlaid across cases (per parameter)

# %%
for p in FOCUS_PARAMS:
    ctx.plots.plot_param_unordered_pair_normalized_values(p, space = "processed", min_abs_denom=1.5,)
    relabel_figure(ctx)
    plt.show()

# %% [markdown]
# ### 5b. Dual-curve plot (aligned + both shuffle modes)

# %%
if ctx.r2_matrix_shifted_obs is not None and ctx.r2_matrix_shifted_both is not None:
    dual_r2_df = build_dual_r2_df(ctx)   # includes real ±σ from 10-perm averaging
    for p in FOCUS_PARAMS:
        try:
            ctx.plots.plot_param_curve_dual(dual_r2_df, p, figsize=(8, 5), show_band=True)
            relabel_figure(ctx)
            plt.show()
        except Exception as e:
            print(f"[warn] {p}: {e}")

# %% [markdown]
# ### 5c. Bias-progression overlay (per parameter)
# For each param, overlays (pred - true) vs true bias curves across cases sorted
# along the dual clean→asym→both-clean→asym→clean sequence.

# %%
for p in FOCUS_PARAMS:
    try:
        fig, stats = ctx.plots.plot_prediction_attractor_map(param=p, )
        relabel_figure(ctx)
        plt.show()
    except Exception as e:
        print(f"[warn] plot_bias_progression_overlay({p}): {e}")

# %% [markdown]
# ### 5d. Pair-normalized values (one param, one mode)
# Shows p_norm = (pred - t0) / (t1 - t0) where t0/t1 are the pair endpoint truths.
# `mode="obs2_vs_truth"` shuffles observable_1; `normalize_endpoints="obs1_to_obs2"`
# maps obs1's truth → 0 and obs2's truth → 1 on the y-axis.

# %%
PAIR_PARAM = FOCUS_PARAMS[0] if FOCUS_PARAMS else "θ4"
try:
    ctx.plots.plot_param_pair_normalized_values(
        PAIR_PARAM,
        space="log",
        mode="obs2_vs_truth",
        normalize_endpoints="obs1_to_obs2",
    )
    relabel_figure(ctx)
    plt.show()
except Exception as e:
    print(f"[warn] plot_param_pair_normalized_values({PAIR_PARAM}): {e}")

# %% [markdown]
# ### 5e. Side-by-side pair-normalized shuffle scatter (obs1 vs obs2 modes)
# Two panels: obs1_vs_truth and obs2_vs_truth for the same param.
# Reveals which observable pulls predictions off-line under each shuffle.

# %%
def _side_by_side_shuffle_scatter(param):
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
    _ax_iter = iter(axes)
    _orig_subplots = plt.subplots
    plt.subplots = lambda *a, **k: (fig, next(_ax_iter))
    stats_obs1 = stats_obs2 = None
    try:
        _, stats_obs1 = ctx.plots.plot_pair_normalized_shuffle_scatter(
            param=param, n_pairs="all", case=None,
            mode="obs1_vs_truth", color_by_theta1=True,
        )
        _, stats_obs2 = ctx.plots.plot_pair_normalized_shuffle_scatter(
            param=param, n_pairs="all", case=None,
            mode="obs2_vs_truth", color_by_theta1=True,
        )
    finally:
        plt.subplots = _orig_subplots
    fig.tight_layout()
    return fig, stats_obs1, stats_obs2

SHUFFLE_SCATTER_PARAM = FOCUS_PARAMS[0] if FOCUS_PARAMS else "θ4"
try:
    fig, s1, s2 = _side_by_side_shuffle_scatter(SHUFFLE_SCATTER_PARAM)
    #relabel_figure(ctx, fig)
    plt.show()
except Exception as e:
    print(f"[warn] shuffle_scatter({SHUFFLE_SCATTER_PARAM}): {e}")

# Loop across FOCUS_PARAMS if you want one per param — uncomment:
for p in FOCUS_PARAMS:
    fig, s1, s2 = _side_by_side_shuffle_scatter(p)
    #relabel_figure(ctx, fig)
    plt.show()

# %% [markdown]
# ## 6. Moment-network plots (only if this pair has a variance head)
#
# `has_variance` is True only for `moment_example`. Load it via:
#   `ctx = load_pair(results_path="moment_example/results/results.pt")`

# %%
if ctx.has_variance:
    for p in FOCUS_PARAMS:
        ctx.plots.plot_marginal_posterior_grid(p, cases=ctx.cases, n_sims=6, seed=42,
                                                space="log_partial")
        relabel_figure(ctx)
        plt.show()
    ctx.plots.plot_sigma_by_case_bars(FOCUS_PARAMS, ctx.cases, space="log_partial")
    relabel_figure(ctx)
    plt.show()
    ctx.plots.plot_pull_distribution(ctx.cases, space="normalized", params=FOCUS_PARAMS)
    plt.show()
else:
    print(f"Pair '{ctx.pair_dir}' has no variance head.")
    print("Reload with:  ctx = load_pair(results_path='moment_example/results/results.pt')")

# %% [markdown]
# ## 7. Custom cell — one sim, one param (moment example only)

# %%
if ctx.has_variance and FOCUS_PARAMS:
    ctx.plots.plot_marginal_posterior_1d(FOCUS_PARAMS[0], sim_idx=3,
                                          cases=ctx.cases, space="log_partial")
    relabel_figure(ctx)
    plt.show()

# %%

# %%
