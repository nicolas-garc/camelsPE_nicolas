# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: py311-main
#     language: python
#     name: python3
# ---

# %%
import sys
import os
import importlib
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import seaborn as sns

base_path = "../src/"
sys.path.append(base_path)
import models
import train
from losses import *
import pipeline
import plots

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
datafilename='../../DATA/data_L50_TNG_v3.hdf5'
with h5py.File(datafilename, 'r') as f:
    print(f"File: {datafilename}")
    print("Top-level attrs:", dict(f.attrs))
    print("\nDatasets:")
    for key in f.keys():
        dset = f[key]
        print(f"  {key:<14} shape={dset.shape}  dtype={dset.dtype}  attrs={dict(dset.attrs)}")


with h5py.File(datafilename, 'r') as f:
    Parameters = f['Parameters'][:, :1024].T
logflag = np.array([False, False, True, True, True, True, False, False, False, True, True, False, False, True, False, True, False, True, True, False, False, True, True, True, True, True, True, False, True, False, True, False, False, False, True])
logflag = logflag[:Parameters.shape[1]]
origParameters = Parameters
if not np.all(Parameters[:, logflag] > 0):
    raise ValueError("Some values to be logged are non-positive.")
PartiallyLoggedParameters = Parameters.copy()
PartiallyLoggedParameters[:, logflag] = np.log(PartiallyLoggedParameters[:, logflag])
means = PartiallyLoggedParameters.mean(axis=0)
stds = PartiallyLoggedParameters.std(axis=0)
Parameters = (PartiallyLoggedParameters - means) / stds

n_sims = Parameters.shape[0]

with h5py.File(datafilename, 'r') as f:
    observable_block = {
        key: torch.from_numpy(f[key][:].T).float()
        for key in sorted(f.keys())
        if key != 'Parameters' and f[key].shape[-1] == n_sims
    }

print(f"Observables available for noise_cases ({n_sims} sims):")
for key, arr in observable_block.items():
    print(f"  {key:<14} {tuple(arr.shape)}")

# %%
noise_cases = {
    # SFR noisy, Ms clean
    "sfr_5.0_ms_0.0": {
        "SFR_Ms_s61": 5.0,
        "Ms_Mh_s61": 0.0,
    },
    "sfr_2.5_ms_0.0": {
        "SFR_Ms_s61": 2.5,
        "Ms_Mh_s61": 0.0,
    },
    "sfr_1.0_ms_0.0": {
        "SFR_Ms_s61": 1.0,
        "Ms_Mh_s61": 0.0,
    },

    # Both clean
    "sfr_0.0_ms_0.0": {
        "SFR_Ms_s61": 0.0,
        "Ms_Mh_s61": 0.0,
    },

    # SFR clean, Ms noisy
    "sfr_0.0_ms_1.0": {
        "SFR_Ms_s61": 0.0,
        "Ms_Mh_s61": 1.0,
    },
    "sfr_0.0_ms_2.5": {
        "SFR_Ms_s61": 0.0,
        "Ms_Mh_s61": 2.5,
    },
    "sfr_0.0_ms_5.0": {
        "SFR_Ms_s61": 0.0,
        "Ms_Mh_s61": 5.0,
    },

    # Both noisy — diagonal cases
    "sfr_0.5_ms_0.5": {
        "SFR_Ms_s61": 0.5,
        "Ms_Mh_s61": 0.5,
    },
    "sfr_1.0_ms_1.0": {
        "SFR_Ms_s61": 1.0,
        "Ms_Mh_s61": 1.0,
    },
    "sfr_2.5_ms_2.5": {
        "SFR_Ms_s61": 2.5,
        "Ms_Mh_s61": 2.5,
    },
    "sfr_5.0_ms_5.0": {
        "SFR_Ms_s61": 5.0,
        "Ms_Mh_s61": 5.0,
    },
    "sfr_10.0_ms_10.0": {
        "SFR_Ms_s61": 10.0,
        "Ms_Mh_s61": 10.0,
    },

    # SFR-only clean reference
    "sfr_clean": {
        "SFR_Ms_s61": 0.0,
    },

    # Ms-only clean reference
    "ms_clean": {
        "Ms_Mh_s61": 0.0,
    },
}

# %%
all_observables = set()
for case in noise_cases.values():
    all_observables.update(case.keys())

unknown = all_observables - set(observable_block)
if unknown:
    raise KeyError(
        f"noise_cases refers to observables not in {datafilename}: {sorted(unknown)}\n"
        f"available: {sorted(observable_block)}"
    )

x_raw_dict = {key: observable_block[key].numpy() for key in all_observables}

# %%
_sorted_obs = sorted(all_observables)
if len(_sorted_obs) < 2:
    raise ValueError(f"need at least two observables in noise_cases; got {_sorted_obs}")
observable_1, observable_2 = _sorted_obs[0], _sorted_obs[1]

for name in (observable_1, observable_2):
    if name not in all_observables:
        raise ValueError(f"{name} not used by any noise_case: {sorted(all_observables)}")

print(f"Observable 1: {observable_1}")
print(f"Observable 2: {observable_2}")

# %%
x_normalized_dict = {key: pipeline.normalize(observable_block[key].numpy()) for key in all_observables}

# %%
x_clean_concat = np.concatenate([pipeline.normalize(x_raw_dict[k]) for k in sorted(all_observables)], axis=1)
x_clean_tensor = torch.from_numpy(x_clean_concat).float()
y = torch.from_numpy(Parameters).float()

# %%
# Hyperparameters
output_dim   = y.shape[1]
hidden_dims  = [128, 64]
lr           = 1e-4
wd           = 1e-3     # was 1e-5 — stronger weight decay against overfitting
dropout_rate = 0.4      # was 0.2 — more dropout against overfitting
epochs       = 2000
val_fraction = 0.1        # for training-time model selection (best-weights)
test_fraction = 0.1       # held out for final unbiased R² / pull calibration
batch_size   = 64

# Best-weights restoration: at the end of fit(), load the weights from the epoch
# whose EMA-smoothed val_loss was lowest, not the last-epoch weights.
RESTORE_BEST_WEIGHTS = True
BEST_WEIGHTS_WINDOW  = 50

# %%
n_val = int(len(x_clean_tensor) * val_fraction)
n_test = int(len(x_clean_tensor) * test_fraction)
torch.manual_seed(0); np.random.seed(0)     # deterministic split — matches sweep/_pair_pipeline
split_perm = torch.randperm(len(x_clean_tensor))
idx_test  = split_perm[:n_test]                                     # held out for final reporting
idx_val   = split_perm[n_test:n_test + n_val]                        # model selection (best-weights)
idx_train = split_perm[n_test + n_val:]                              # gradient updates

x_val, y_val = x_clean_tensor[idx_val], y[idx_val]
val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=64, shuffle=False)

# %%
perm = np.random.permutation(len(idx_test))

# %%
N_SHUFFLE_PERMS = 10
_shuffle_rng = np.random.default_rng(0)
shuffle_perms = [_shuffle_rng.permutation(len(idx_test)) for _ in range(N_SHUFFLE_PERMS)]

# %%
importlib.reload(train)
importlib.reload(models)

# %%
criterion = nn.MSELoss()

# %% [markdown]
# ## Configure pipeline and plots modules

# %%
pipeline.configure(
    observable_1=observable_1,
    observable_2=observable_2,
    x_normalized_dict=x_normalized_dict,
    y=y,
    idx_val=idx_val,
    idx_test=idx_test,
    batch_size=batch_size,
    device=device,
    logflag=logflag,
    means=means,
    stds=stds,
    output_dim=output_dim,
    perm=perm,
)

# %% [markdown]
# ## Training loop

# %%
all_results = []

for case_name, selected_observables in noise_cases.items():
    print(f"\nTraining case: {case_name}")

    input_dim = sum(x_raw_dict[k].shape[1] for k in selected_observables)

    model = models.SimpleMLP(input_dim, hidden_dims, output_dim, dropout_rate).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.MSELoss()

    train_loader_fn = pipeline.make_train_loader_fn(selected_observables, x_normalized_dict, y, idx_train, batch_size)

    val_loader_fn = pipeline.make_val_loader_fn(selected_observables, x_normalized_dict, y, idx_val, batch_size)
    val_loader = val_loader_fn()

    train_losses, val_losses = train.fit_with_epoch_noise(
        model=model,
        train_loader=None,
        train_loader_fn=train_loader_fn,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=epochs,
        restore_best_weights=RESTORE_BEST_WEIGHTS,
        best_weights_smoothing_window=BEST_WEIGHTS_WINDOW,
    )

    all_results.append({
        "case_name": case_name,
        "selected_observables": selected_observables,
        "model": model,
        "train_losses": train_losses,
        "val_losses": val_losses
    })

# %%
pipeline.configure(all_results=all_results)

# %% [markdown]
# ## R² computation — aligned

# %%
output_dir = os.path.abspath(os.path.join(os.getcwd(), "../../noise_results"))
os.makedirs(output_dir, exist_ok=True)

r2_matrix = np.zeros((len(all_results), output_dim))

for result in all_results:
    model = result["model"]
    train_losses = result["train_losses"]
    val_losses = result["val_losses"]
    case_name = result["case_name"]

    print(f" Case: {case_name}")

    # Plot losses
    plt.figure(figsize=(10, 6))
    epochs_range = range(1, len(train_losses) + 1)
    plt.plot(epochs_range, train_losses, label='Training Loss')
    plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.title(f'Training and Validation Loss — {case_name}')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(output_dir, f"{case_name}_loss.png")
    plt.savefig(path, dpi=200)
    plt.show()

    all_predictions, all_true_values = pipeline.get_case_predictions(result, mode="aligned")

    n_cols = int(np.floor(np.sqrt(output_dim)))
    n_rows = int(np.ceil(output_dim / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = np.atleast_1d(axes).reshape(n_rows, n_cols)

    for i in range(output_dim):
        ax = axes.flat[i]
        preds_i = all_predictions[:, i]
        trues_i = all_true_values[:, i]
        ax.scatter(trues_i, preds_i, alpha=0.6)
        r2 = r2_score(trues_i, preds_i)
        rmse = np.sqrt(mean_squared_error(trues_i, preds_i))

        r2_matrix[all_results.index(result), i] = r2

        min_val = min(trues_i.min(), preds_i.min())
        max_val = max(trues_i.max(), preds_i.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title(f'R² = {r2:.3f}, RMSE = {rmse:.3f}')
        ax.grid(True)
        span = max_val - min_val
        ax.set_xlim(min_val - 0.1 * span, max_val + 0.1 * span)
        ax.set_ylim(min_val - 0.1 * span, max_val + 0.1 * span)

    for j in range(output_dim, n_rows * n_cols):
        fig.delaxes(axes.flat[j])

    fig.suptitle(f"Prediction Results — {case_name}", fontsize=16)
    fig.tight_layout()

# %%
r2_df = pd.DataFrame(
    r2_matrix,
    index=[f"{i}" for i in noise_cases.keys()],
    columns=[f"Param {j}" for j in range(r2_matrix.shape[1])]
)

plt.figure(figsize=(16, 8))
sns.heatmap(
    r2_df,
    vmin = -1.0,
    vmax = 1.0,
    cmap = 'Spectral',
    annot=True,
    fmt=".2f",
    linewidths = 0.2,
    cbar_kws={'label': 'Validation $R^{2}$'}
)
sns.set_style("white")

plt.title(f"Constraining Power Heatmap (Validation $R^{2}$)")
plt.ylabel("Observable")
plt.xlabel("Parameter")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Shuffle test — S₁: obs1_vs_truth (shuffle observable_2)

# %%
r2_matrix_shifted_observable_only = np.full((len(all_results), output_dim), np.nan)

for result_idx, result in enumerate(all_results):
    case_name = result["case_name"]
    print(f"S₁ — Case: {case_name}")

    all_predictions, all_true_values = pipeline.get_case_predictions(result, mode="obs1_vs_truth", perm=perm)

    for i in range(output_dim):
        r2_matrix_shifted_observable_only[result_idx, i] = r2_score(
            all_true_values[:, i], all_predictions[:, i]
        )

# %%
r2_matrix_shifted_observable_only, r2_std_matrix_observable_only = pipeline.average_r2_over_perms(
    "obs1_vs_truth", shuffle_perms
)

# %%
r2_shuffled = pd.DataFrame(
    r2_matrix_shifted_observable_only,
    index=[f"{i}" for i in noise_cases.keys()],
    columns=[f"Param {j}" for j in range(r2_matrix.shape[1])]
)

plt.figure(figsize=(16,8))
sns.heatmap(r2_shuffled, vmin=-1, vmax=1, annot=True, fmt=".1f",
            cmap="Spectral", linewidths=0.2, cbar_kws={"label": "Validation R²"})
plt.xlabel("Predicted Parameter")
plt.ylabel("Noise Case")
plt.title("R² Heatmap — S₁: shuffle obs2, keep obs1 truths")
plt.tight_layout()
plt.show()

# %%
delta_r2_observable_only = r2_matrix_shifted_observable_only - r2_matrix

delta_r2_df = pd.DataFrame(
    delta_r2_observable_only,
    index=[f"{i}" for i in noise_cases.keys()],
    columns=[f"Param {j}" for j in range(r2_matrix.shape[1])]
)

plt.figure(figsize=(16, 8))
sns.heatmap(delta_r2_df, annot=True, fmt=".2f", cmap="Spectral", center=0.0,
            linewidths=0.3, vmin=-0.5, cbar_kws={"label": "ΔR² (Shifted − Original)"})
plt.title("ΔR² — S₁: shuffle obs2")
plt.xlabel("Parameter")
plt.ylabel("Noise Case")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Shuffle test — S₂: obs2_vs_truth (shuffle observable_1)

# %%
r2_matrix_shifted_both = np.full((len(all_results), output_dim), np.nan)

for result_idx, result in enumerate(all_results):
    case_name = result["case_name"]
    print(f"S₂ — Case: {case_name}")

    all_predictions, all_true_values = pipeline.get_case_predictions(result, mode="obs2_vs_truth", perm=perm)

    for i in range(output_dim):
        r2_matrix_shifted_both[result_idx, i] = r2_score(
            all_true_values[:, i], all_predictions[:, i]
        )

# %%
r2_matrix_shifted_both, r2_std_matrix_both = pipeline.average_r2_over_perms("obs2_vs_truth", shuffle_perms)

# %%
r2_shuffled = pd.DataFrame(
    r2_matrix_shifted_both,
    index=[f"{i}" for i in noise_cases.keys()],
    columns=[f"Param {j}" for j in range(r2_matrix.shape[1])]
)

plt.figure(figsize=(16,8))
sns.heatmap(r2_shuffled, vmin=-1, vmax=1, annot=True, fmt=".1f",
            cmap="Spectral", linewidths=0.2, cbar_kws={"label": "Validation R²"})
plt.xlabel("Predicted Parameter")
plt.ylabel("Noise Case")
plt.title("R² Heatmap — S₂: shuffle obs1, keep obs2 truths")
plt.tight_layout()
plt.show()

# %%
delta_r2_both = r2_matrix_shifted_both - r2_matrix

delta_r2_df = pd.DataFrame(
    delta_r2_both,
    index=[f"{i}" for i in noise_cases.keys()],
    columns=[f"Param {j}" for j in range(r2_matrix.shape[1])]
)

plt.figure(figsize=(16, 8))
sns.heatmap(delta_r2_df, annot=True, fmt=".2f", cmap="Spectral", center=0.0,
            linewidths=0.3, vmin=-0.5, cbar_kws={"label": "ΔR² (Shifted − Original)"})
plt.title("ΔR² — S₂: shuffle obs1")
plt.xlabel("Parameter")
plt.ylabel("Noise Case")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Combined R² DataFrame

# %%
cases = [r["case_name"] for r in all_results]
P = r2_matrix.shape[1]
param_names = [f"θ{j}" for j in range(P)]

rows = []
for i, case in enumerate(cases):
    for j, p in enumerate(param_names):
        rows.append({
            "case": case,
            "param": p,
            "r2_aligned": float(r2_matrix[i, j]),
            "r2_shuf_obs1": float(r2_matrix_shifted_observable_only[i, j]),
            "r2_shuf_obs2": float(r2_matrix_shifted_both[i, j]),
            "r2_shuf_obs1_std": float(r2_std_matrix_observable_only[i, j]),
            "r2_shuf_obs2_std": float(r2_std_matrix_both[i, j]),
        })

dual_r2_df = pd.DataFrame(rows)
dual_r2_df["delta_obs1"] = dual_r2_df["r2_shuf_obs1"] - dual_r2_df["r2_aligned"]
dual_r2_df["delta_obs2"] = dual_r2_df["r2_shuf_obs2"] - dual_r2_df["r2_aligned"]

# %% [markdown]
# ## Configure plots module with trained results

# %%
plots.configure(
    all_results=all_results,
    output_dim=output_dim,
    observable_1=observable_1,
    observable_2=observable_2,
    logflag=logflag,
    means=means,
    stds=stds,
    x_normalized_dict=x_normalized_dict,
    y=y,
    idx_val=idx_val,
    param_names=param_names,
    noise_cases=noise_cases,
    batch_size=batch_size,
    device=device,
    perm=perm,
    r2_matrix=r2_matrix,
    r2_matrix_shifted_observable_only=r2_matrix_shifted_observable_only,
    r2_matrix_shifted_both=r2_matrix_shifted_both,
)

# %% [markdown]
# ## Plots

# %%
focus_params = ["θ0","θ1","θ2","θ4","θ7","θ11"]
for p in focus_params:
    plots.plot_param_curve_dual(dual_r2_df, p, figsize=(8,5))

# %%
param_np = [0,1,2,4,6,7,8,11,16]

# %%
fig = plots.plot_param_lines_min(0, mode="aligned", residual=True, n_sims=30, seed=42)

# %%
for p in param_np:
    fig = plots.plot_param_all_val(f"θ{p}", mode="aligned")

# %%
plots.plot_param_pair_normalized_values(
    "θ4",
    space="log",
    mode="obs2_vs_truth",
    normalize_endpoints="obs1_to_obs2"
)
plt.show()

# %%
for p in list(param_np):
    fig, stats = plots.plot_param_unordered_pair_normalized_values(
        param=p,
        obs1_key=observable_2,
        obs2_key=observable_1,
        space="processed",
        min_abs_denom=1.5,
    )
    plt.show()

# %%
fig, stats = plots.plot_constraint_structure_2d(
    param=4,
    case_name=all_results[0]["case_name"],
    obs1_key=observable_2,
    obs2_key=observable_1,
    mode="aligned",
    target_space="physical",
    plot_kind="hexbin",
    gridsize=45,
)
plt.show()
print(stats)

# %%
fig, stats = plots.plot_unordered_pair_prediction_map(
    param=4,
    case_name=all_results[3]["case_name"],
    obs1_key=observable_2,
    obs2_key=observable_1,
    space="processed",
    value_kind="pred",
    plot_kind="hexbin",
    gridsize=45,
)
plt.show()
print(stats)

# %%
figs_by_param = plots.plot_predictions_vs_true_by_noise(focus_params)
for p_label, fig in figs_by_param.items():
    plt.show()

# %% [markdown]
# ## Loops over param_np for analysis plots

# %%
for p in list(param_np):
    fig, stats = plots.plot_bias_progression_overlay(param=p)
    plt.show()

# %%
for p in list(param_np):
    fig, stats = plots.plot_directional_pull_by_true(param=p)
    plt.show()

# %%
for p in list(param_np):
    fig, stats = plots.plot_prediction_attractor_map(param=p)
    plt.show()

# %%
for p in list(param_np):
    fig, stats = plots.plot_per_sim_accuracy_heatmap(param=p)
    plt.show()

# %%
p_plot = 4
fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
_ax_iter = iter(axes)
_orig_subplots = plt.subplots
plt.subplots = lambda *a, **k: (fig, next(_ax_iter))
try:
    _, stats_obs1 = plots.plot_pair_normalized_shuffle_scatter(param=p_plot, n_pairs="all", case=None, mode="obs1_vs_truth", color_by_theta1=True)
    _, stats_obs2 = plots.plot_pair_normalized_shuffle_scatter(param=p_plot, n_pairs="all", case=None, mode="obs2_vs_truth", color_by_theta1=True)
finally:
    plt.subplots = _orig_subplots
fig.tight_layout()
plt.show()

# %%
