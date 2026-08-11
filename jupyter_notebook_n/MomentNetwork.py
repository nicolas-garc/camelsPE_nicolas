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
# # Moment Network — posterior mean + variance per parameter
#
# Follows Jeffrey & Wandelt 2020 (arXiv:2011.05991). For each case:
#   1. Train F(x) with MSE — posterior mean μ (already done by test-noise pipeline).
#   2. K-fold OOF residuals from F → variance targets v_i = log(r_i²), z-scored.
#      (OOF because train_loss < val_loss in this project — real overfitting means
#       naive same-set residuals would understate σ.)
#   3. Train G(x) on (x_train, v_z) with MSE — G predicts log-variance z-score.
#   4. At eval: μ = F(x_val); σ = √exp(G(x_val)·std + mean).
#
# Marginal posterior plots then show how obs1-only vs obs2-only vs both compare
# per parameter — the probabilistic version of the R² comparison in test-noise.

# %%
import sys
import os
import importlib
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd

base_path = "../src/"
sys.path.append(base_path)
import models
import train
from losses import *
import pipeline
import plots

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# ## Data loading (identical to test-noise-Copy1)

# %%
datafilename = "../../DATA/data_L50_TNG_v3.hdf5"

with h5py.File(datafilename, "r") as f:
    Parameters = f["Parameters"][:, :1024].T

logflag = np.array([False, False, True, True, True, True, False, False, False, True, True, False, False, True, False, True, False, True, True, False, False, True, True, True, True, True, True, False, True, False, True, False, False, False, True])
logflag = logflag[:Parameters.shape[1]]
if not np.all(Parameters[:, logflag] > 0):
    raise ValueError("Some values to be logged are non-positive.")
PartiallyLoggedParameters = Parameters.copy()
PartiallyLoggedParameters[:, logflag] = np.log(PartiallyLoggedParameters[:, logflag])
means = PartiallyLoggedParameters.mean(axis=0)
stds = PartiallyLoggedParameters.std(axis=0)
Parameters = (PartiallyLoggedParameters - means) / stds

n_sims = Parameters.shape[0]
with h5py.File(datafilename, "r") as f:
    observable_block = {
        key: torch.from_numpy(f[key][:].T).float()
        for key in sorted(f.keys())
        if key != "Parameters" and f[key].shape[-1] == n_sims
    }

# %% [markdown]
# ## Noise cases — start with the three we need for marginal-posterior comparison

# %%
noise_cases = {
    # Reference: obs2 alone (SFR)
    "sfr_clean": {"SFR_Ms_s61": 0.0},
    # Reference: obs1 alone (Ms)
    "ms_clean":  {"Ms_Mh_s61": 0.0},
    # Both observables, both clean
    "sfr_0.0_ms_0.0": {"SFR_Ms_s61": 0.0, "Ms_Mh_s61": 0.0},
    # A noisy pair (optional; comment out to speed up the run)
    "sfr_1.0_ms_1.0": {"SFR_Ms_s61": 1.0, "Ms_Mh_s61": 1.0},
}

all_observables = set()
for case in noise_cases.values():
    all_observables.update(case.keys())
x_raw_dict = {key: observable_block[key].numpy() for key in all_observables}

_sorted_obs = sorted(all_observables)
observable_1, observable_2 = _sorted_obs[0], _sorted_obs[1]
print(f"observable_1 = {observable_1}   observable_2 = {observable_2}")

x_normalized_dict = {k: pipeline.normalize(observable_block[k].numpy()) for k in all_observables}
y = torch.from_numpy(Parameters).float()

# %% [markdown]
# ## Hyperparameters & train/val split

# %%
output_dim   = y.shape[1]
hidden_dims  = [128, 64]
dropout_rate = 0.4      # was 0.2 — bumped to counter overfitting
wd           = 1e-3     # was 1e-5 — stronger weight decay
epochs       = 2000
val_fraction = 0.1        # for training-time model selection (best-weights)
test_fraction = 0.1       # held out for final unbiased R² / pull calibration
batch_size   = 64

# Best-weights restoration (see src/train.py::fit_with_epoch_noise)
RESTORE_BEST_WEIGHTS = True
BEST_WEIGHTS_WINDOW  = 50

# Moment-network specific
K_FOLDS      = 5
FOLD_EPOCHS  = epochs      # partial-F folds train as thoroughly as the main F
VAR_EPOCHS   = epochs

n_val = int(n_sims * val_fraction)
n_test = int(n_sims * test_fraction)
torch.manual_seed(0); np.random.seed(0)
split_perm = torch.randperm(n_sims)
idx_test  = split_perm[:n_test]                                     # held out for final reporting
idx_val   = split_perm[n_test:n_test + n_val]                        # model selection during training
idx_train = split_perm[n_test + n_val:]                              # gradient updates
perm = np.random.permutation(len(idx_test))

# %% [markdown]
# ## Configure pipeline and plots modules

# %%
importlib.reload(train); importlib.reload(models); importlib.reload(pipeline); importlib.reload(plots)

pipeline.configure(
    observable_1=observable_1, observable_2=observable_2,
    x_normalized_dict=x_normalized_dict, x_raw_dict=x_raw_dict,
    y=y, idx_val=idx_val, idx_test=idx_test, idx_train=idx_train,
    batch_size=batch_size, device=device,
    logflag=logflag, means=means, stds=stds, output_dim=output_dim,
    hidden_dims=hidden_dims, dropout_rate=dropout_rate, epochs=epochs,
    perm=perm,
)

# %% [markdown]
# ## Train F* per case (mean net — same as test-noise pipeline)

# %%
all_results = []
for case_name, selected_observables in noise_cases.items():
    print(f"\n=== Training F* for {case_name} ===")
    input_dim = sum(x_raw_dict[k].shape[1] for k in selected_observables)
    model = models.SimpleMLP(input_dim, hidden_dims, output_dim, dropout_rate).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=wd)
    criterion = nn.MSELoss()

    train_loader_fn = pipeline.make_train_loader_fn(
        selected_observables, x_normalized_dict, y, idx_train, batch_size)
    val_loader = pipeline.make_val_loader_fn(
        selected_observables, x_normalized_dict, y, idx_val, batch_size)()

    train_losses, val_losses = train.fit_with_epoch_noise(
        model=model, train_loader=None, train_loader_fn=train_loader_fn,
        val_loader=val_loader, optimizer=optimizer, criterion=criterion,
        device=device, epochs=epochs,
        restore_best_weights=RESTORE_BEST_WEIGHTS,
        best_weights_smoothing_window=BEST_WEIGHTS_WINDOW,
    )
    all_results.append({
        "case_name": case_name,
        "selected_observables": selected_observables,
        "model": model,
        "train_losses": train_losses,
        "val_losses": val_losses,
    })

pipeline.configure(all_results=all_results)

# %% [markdown]
# ## Diagnostic — R² per case (sanity check before the variance net)

# %%
r2_matrix = np.zeros((len(all_results), output_dim))
for ri, r in enumerate(all_results):
    preds, trues = pipeline.get_case_predictions(r, mode="aligned")
    r2_matrix[ri] = r2_score(trues, preds, multioutput="raw_values")

param_names = [f"θ{j}" for j in range(output_dim)]
print(pd.DataFrame(r2_matrix,
                   index=[r["case_name"] for r in all_results],
                   columns=param_names).round(2))

# %% [markdown]
# ## Fit G (variance head) per case — this is the moment-network step
#
# Each call does K-fold retraining of F on train subsets (for OOF residuals), then
# trains G on (x_train, log-variance z-scored). Attaches `var_model`, `var_target_mean`,
# `var_target_std` to the result dict.
#
# Time budget: K × fold_epochs mean-net trainings + 1 G training per case.

# %%
FOCUS_PARAMS_FOR_MOMENT = ["θ0", "θ1", "θ2", "θ4", "θ7", "θ11"]

for r in all_results:
    print(f"\n=== Fitting moment head M for {r['case_name']} ===")
    pipeline.fit_moment_head(
        r, FOCUS_PARAMS_FOR_MOMENT,
        models_module=models, train_module=train,
        K=K_FOLDS, fold_epochs=FOLD_EPOCHS, moment_epochs=VAR_EPOCHS,
        wd=wd,
        restore_best_weights=RESTORE_BEST_WEIGHTS,
        best_weights_smoothing_window=BEST_WEIGHTS_WINDOW,
    )

# %% [markdown]
# ## Configure plots module (needs the fitted var_models on all_results)

# %%
plots.configure(
    all_results=all_results, output_dim=output_dim,
    observable_1=observable_1, observable_2=observable_2,
    logflag=logflag, means=means, stds=stds,
    x_normalized_dict=x_normalized_dict, y=y,
    idx_val=idx_val, idx_test=idx_test,
    param_names=param_names, noise_cases=noise_cases,
    batch_size=batch_size, device=device, perm=perm,
    r2_matrix=r2_matrix,
)

# %% [markdown]
# ## Marginal posteriors — the payoff plots
#
# For a single validation sim, overlay p(θ|x) Gaussians from each case. Shows
# how the obs1-only, obs2-only, and both-clean models constrain each parameter
# differently. `space="log_partial"` = standardized log space (Gaussianity is
# honest here for log parameters).

# %%
CASES = list(noise_cases.keys())
FOCUS_PARAMS = ["θ0", "θ1", "θ2", "θ4", "θ7", "θ11"]

# Single-sim, single-parameter comparison
for p in FOCUS_PARAMS[:3]:
    plots.plot_marginal_posterior_1d(p, sim_idx=0, cases=CASES, space="log_partial")
    plt.show()

# %% [markdown]
# ### Grid: multiple sims × chosen cases, one param per figure

# %%
for p in FOCUS_PARAMS:
    plots.plot_marginal_posterior_grid(p, cases=CASES, n_sims=6, seed=42,
                                        space="log_partial")
    plt.show()

# %% [markdown]
# ### Population σ per case per parameter (bar chart)

# %%
plots.plot_sigma_by_case_bars(FOCUS_PARAMS, CASES, space="log_partial",
                               reducer="median")
plt.show()

# %% [markdown]
# ### Calibration diagnostic — pull = (μ − true) / σ
# Well-calibrated → histogram matches N(0,1). std >> 1 → over-confident (σ too small).

# %%
plots.plot_pull_distribution(CASES, space="normalized", params=FOCUS_PARAMS)
plt.show()

# %%
