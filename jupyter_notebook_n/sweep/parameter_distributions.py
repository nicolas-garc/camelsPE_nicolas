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
#     name: py311-main
# ---

# %% [markdown]
# # Parameter distributions across spaces
#
# All 35 parameters' distributions, one grid per space:
#
# 1. **Physical** -- raw simulation values, straight from the HDF5 file.
# 2. **Log (logflag)** -- the 21 logflagged parameters log-transformed, the
#    other 14 left as-is. This is the space training happens in, before
#    standardization.
# 3. **Log + normalized** -- (2) z-score standardized. The model's actual
#    training target space.
#
# Motivation: logflag parameters are log-transformed specifically because
# their physical-space distributions are long-tailed / span a large dynamic
# range; the non-logflag parameters are assumed to already be reasonably
# uniform in physical space. This notebook checks that assumption directly
# instead of assuming it, and feeds into whether attractor-map binning
# should also switch to log-space for logflag parameters (see CLAUDE.md's
# noise section and the chat discussion this was pulled from).

# %%
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import matplotlib.pyplot as plt
from _analyze_helpers import _load_data

# %% [markdown]
# ## Load and derive the three spaces
# `_load_data()` already returns `y` in log+normalized space (identical to
# what the model trains on) plus `means`/`stds`/`logflag` -- reusing those
# to reconstruct the other two spaces instead of re-deriving from the HDF5
# file, so this stays byte-identical to what `run_pair` actually trains on.

# %%
y, logflag, means, stds, observable_block = _load_data()

Pnorm = y.numpy()                              # log + normalized (training target space)
Pll = Pnorm * stds + means                     # log space only (undo standardization)
Pphys = Pll.copy()
Pphys[:, logflag] = np.exp(Pll[:, logflag])    # physical/raw space

output_dim = Pnorm.shape[1]
param_names = [f"θ{i}" for i in range(output_dim)]
print(f"{output_dim} parameters ({int(logflag.sum())} logflagged, "
      f"{int((~logflag).sum())} not), {Pnorm.shape[0]} sims")

# %% [markdown]
# ## Grid plotting helper
# One grid per space: all 35 parameters as histograms, arranged 5 columns
# wide. Each subplot is tagged (log)/(lin) so you can immediately see which
# parameters are logflagged when comparing spread across the three grids.

# %%
def plot_param_grid(data, title, n_cols=5, bins=30, color="#3C3489"):
    n_rows = int(np.ceil(output_dim / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = axes.flatten()
    for i in range(output_dim):
        ax = axes[i]
        ax.hist(data[:, i], bins=bins, color=color, alpha=0.75, edgecolor="none")
        tag = "log" if logflag[i] else "lin"
        ax.set_title(f"{param_names[i]}  ({tag})", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)
    for j in range(output_dim, len(axes)):
        axes[j].axis("off")
    fig.suptitle(title, fontsize=15, y=1.0)
    fig.tight_layout()
    return fig

# %% [markdown]
# ## 1. Physical space (raw values)
# Logflagged parameters should visibly show long right tails / concentrated
# low-value clumps here -- that skew is exactly why they get logflagged.

# %%
fig = plot_param_grid(Pphys, "Physical space -- raw parameter values")
plt.show()

# %% [markdown]
# ## 2. Log (logflag) space
# Logflagged parameters should look much more symmetric/uniform here; the
# non-logflag parameters are untouched (identical to space 1 for those).

# %%
fig = plot_param_grid(Pll, "Log space -- logflagged parameters log-transformed")
plt.show()

# %% [markdown]
# ## 3. Log + normalized space (model's training target space)
# Same shapes as space 2, just recentered/rescaled to mean 0, std 1 -- this
# is exactly what the network's MSE loss operates on.

# %%
fig = plot_param_grid(Pnorm, "Log + normalized space -- model's training target space")
plt.show()
