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
base_path = "../src/"
sys.path.append(base_path)
import models
import train
from losses import *

# %%

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
#datafilename='../../DATA/data_L25LH_TNG.hdf5'
datafilename='../../DATA/data_L50_TNG_v3.hdf5'
with h5py.File(datafilename, 'r') as f:
    print("Datasets available:")
    for key in f.keys():
        print(key)


with h5py.File(datafilename, 'r') as f:
    #Parameters = f['Parameters'][0, :1024].T.reshape(-1, 1)
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

# Load every dataset with one row per simulation — these are the observables usable in
# noise_cases. Bin-center arrays (logMh_*, logMs_*, SFRH_z) hold a single row, so they
# drop out automatically. Adding an observable to the file makes it available here with
# no code change.
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
for case in noise_cases.values(): 
    print(case.keys())

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

# ============================ SET THIS PER ANALYSIS ============================
# The pair under comparison — the single source of truth. Every loop and plot below
# resolves through these two names, so switching the pair is a one-line edit here.
#
# observable_2 is THE ONE THAT GETS SHUFFLED in both comparisons:
#   obs1_vs_truth -> shuffle observable_2, truths stay with observable_1
#                    R2 survives only if the model reads observable_1
#   obs2_vs_truth -> shuffle observable_2 and truths together
#                    R2 survives only if the model reads observable_2
# So set observable_2 to the observable whose information you want to destroy.
#
# Don't derive these from all_observables: it's a set, so list(...)[0]/[1] swapped
# them between kernel restarts; sorted() is stable but alphabetical, not your intent.
observable_1 = "Ms_Mh_s61"
observable_2 = "SFR_Ms_s61"
# ===============================================================================

for name in (observable_1, observable_2):
    if name not in all_observables:
        raise ValueError(f"{name} not used by any noise_case: {sorted(all_observables)}")

print(f"Observable 1: {observable_1}")
print(f"Observable 2: {observable_2}")


# %%
def resolve_shuffle(selected_observables, mode):
    """Which observables to shuffle, and whether the truths follow them.

    Both comparisons shuffle observable_2 — only the truth alignment differs, which is
    what isolates who carries the information:
      "aligned"       nothing shuffled (reference)
      "obs1_vs_truth" shuffle observable_2, truths stay with observable_1
                      -> R2 survives only if the model reads observable_1
      "obs2_vs_truth" shuffle observable_2 and truths together, truths follow observable_2
                      -> R2 survives only if the model reads observable_2

    Returns (keys_to_shuffle, shuffle_y), or None if this case has no observable_2
    to shuffle (e.g. single-observable reference cases).
    """
    if mode == "aligned":
        return set(), False
    if observable_2 not in selected_observables:
        return None
    if mode == "obs1_vs_truth":
        return {observable_2}, False
    if mode == "obs2_vs_truth":
        return {observable_2}, True
    raise ValueError(f"Unknown mode: {mode!r}")


# %%

# %%
def shuffle_observable(obs_dict,keys_to_shift, perm):
    shifted_dict = obs_dict.copy()
    
    for keys in keys_to_shift:
        shifted_dict[keys] = obs_dict[keys][perm]

    return shifted_dict


# %%
def add_noise_and_normalize(array_np, noise_level=0.0):
    """Degrade an observable with Gaussian noise, in units of each bin's std.

    The renormalize at the end is what makes noise_level mean "information
    destroyed" and nothing else. Adding noise without it left the noisy training
    inputs at std sqrt(1 + noise_level**2) while validation stayed at std 1 --
    a 5x scale gap at noise_level=5 -- so the observable was suppressed at
    validation by attenuation on top of the information loss. Renormalizing puts
    train and val on the same scale for identical information loss.

    Not in-place: `array_np + noise` rebinds rather than mutating the caller's
    array, so this can never corrupt x_normalized_dict.
    """
    if noise_level > 0:
        std = np.std(array_np, axis=0)
        array_np = array_np + np.random.normal(0.0, noise_level * std, array_np.shape)
    mean = np.mean(array_np, axis=0)
    std = np.std(array_np, axis=0)
    return (array_np - mean) / std


# %%
def normalize(array_np):
    mean = np.mean(array_np, axis = 0)
    std = np.std(array_np, axis = 0)
    array_np = (array_np - mean) / std

    return array_np


# %%
x_normalized_dict = {key: normalize(observable_block[key].numpy()) for key in all_observables}

# %%
x_clean_concat = np.concatenate([normalize(x_raw_dict[k]) for k in sorted(all_observables)], axis=1)
x_clean_tensor = torch.from_numpy(x_clean_concat).float()
y = torch.from_numpy(Parameters).float()

# %%
x_clean_tensor.shape

# %%
# Hyperparameters
#input_dim    = x.shape[1]
output_dim   = y.shape[1]
hidden_dims  = [128, 64]
#hidden_dims  = [128, 64, 64]
#hidden_dims  = [64, 64]
lr           = 1e-4
wd           = 1e-5
dropout_rate = 0.2
epochs       = 2000
val_fraction = 0.1
batch_size   = 64
separate_models = False

# %%
n_val = int(len(x_clean_tensor) * val_fraction)
# named split_perm, not perm: the shuffle perm below is a *different* permutation and
# several plot helpers read `perm` off globals(). Sharing the name meant re-running this
# cell last left `perm` at length len(x_clean_tensor), and those helpers would quietly
# fall back to a fresh permutation instead of the one the R2 loops used.
split_perm = torch.randperm(len(x_clean_tensor))
idx_train = split_perm[:-n_val]
idx_val = split_perm[-n_val:]

x_val, y_val = x_clean_tensor[idx_val], y[idx_val]
val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=64, shuffle=False)

# %%
#we randomize the indices of the validation set
# keep this one named `perm`: the plot helpers look it up via globals().get("perm")
perm = np.random.permutation(len(idx_val))

# %%
importlib.reload(train)
importlib.reload(models)

# %%
criterion = nn.MSELoss()


# %% [raw]
# # model, optimizer, loss function
#
# criterion = nn.MSELoss()
#
# if separate_models and output_dim > 1:
#     models_list = [models.SimpleMLP(input_dim, hidden_dims, 1, dropout_rate).to(device) for _ in range(output_dim)]
#     optimizers = [optim.Adam(m.parameters(), lr=lr, weight_decay=wd) for m in models_list]
#     #schedulers = [ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5, verbose=True) for opt in optimizers]
# else:
#     model = models.SimpleMLP(input_dim, hidden_dims, output_dim, dropout_rate).to(device)
#     optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
#     #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
#

# %%
def make_train_loader_fn(selected_observables, x_dict, y_vector, idx_train, batch_size):
    def loader_fn():
        x_list = []
        for key in sorted(selected_observables.keys()):
            noise_level = selected_observables[key]
            arr = x_dict[key][idx_train]  # Subset training samples
            x_proc = add_noise_and_normalize(arr, noise_level)  # noise, then back to std 1
            x_list.append(torch.from_numpy(x_proc).float())

        x_epoch = torch.cat(x_list, dim=1)
        y_epoch = y_vector[idx_train]


        return DataLoader(TensorDataset(x_epoch, y_epoch), batch_size=batch_size, shuffle=True)

    return loader_fn


# %%
def make_val_loader_fn(selected_observables, x_dict, y_vector, idx, batch_size, key_to_shuffle=None, perm=None, shuffle_y=False):
    # normalize key_to_shuffle to a set
    if key_to_shuffle is None:
        shuffle_keys = set()
    elif isinstance(key_to_shuffle, str):
        shuffle_keys = {key_to_shuffle}
    else:
        shuffle_keys = set(key_to_shuffle)

    # A key that matches nothing below would leave x unshuffled while shuffle_y still
    # scrambles the truths — a silently meaningless result. Fail loudly instead.
    unknown = shuffle_keys - set(selected_observables)
    if unknown:
        raise KeyError(
            f"key_to_shuffle {sorted(unknown)} is not among this case's observables "
            f"{sorted(selected_observables)}; the shuffle would silently do nothing."
        )
    idx = np.asarray(idx)

    #takes the selected observables and returns a function that generates a DataLoader for the validation set, with optional shuffling of specified keys and the target variable.
    def loader_fn():
        x_list = []
        for key in sorted(selected_observables.keys()):
            arr = x_dict[key][idx]  # shape: [len(idx), features]

            # shuffle only specified keys
            if perm is not None and key in shuffle_keys:
                arr = arr[perm]
            x_list.append(torch.from_numpy(arr).float())
        x_data = torch.cat(x_list, dim=1)
        # process y
        y_slice = y_vector[idx]
        if isinstance(y_slice, np.ndarray):
            y_data = torch.from_numpy(y_slice).float()
        else:
            y_data = y_slice

        if shuffle_y and perm is not None:
            y_data = y_data[perm]

        return DataLoader(TensorDataset(x_data, y_data), batch_size=batch_size, shuffle=False)

    return loader_fn



# %%
all_results = []

for case_name, selected_observables in noise_cases.items():
    print(f"\Running test case: {case_name}")
    
    input_dim = sum(x_raw_dict[k].shape[1] for k in selected_observables)
    
    model = models.SimpleMLP(input_dim, hidden_dims, output_dim, dropout_rate).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.MSELoss()

    train_loader_fn = make_train_loader_fn(selected_observables, x_normalized_dict, y, idx_train, batch_size)

    val_loader_fn = make_val_loader_fn(selected_observables, x_normalized_dict, y, idx_val, batch_size)
    val_loader = val_loader_fn()

    train_losses, val_losses = train.fit_with_epoch_noise(
        model=model,
        train_loader=None,
        train_loader_fn=train_loader_fn,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=epochs
    )

    # Store results in memory
    all_results.append({
        "case_name": case_name,
        "selected_observables": selected_observables,
        "model": model,
        "train_losses": train_losses,
        "val_losses": val_losses
    })


# %% [raw]
# train_losses_list, val_losses_list = [], []
#
# if separate_models and output_dim > 1:
#     for i, (model_i, opt_i) in enumerate(zip(models_list, optimizers)):
#         print(f"model #{i:d}")
#         tl, vl = train.fit(
#             model_i,
#             DataLoader(TensorDataset(x_train, y_train[:, i:i+1]), batch_size=batch_size, shuffle=True),
#             DataLoader(TensorDataset(x_val, y_val[:, i:i+1]), batch_size=batch_size, shuffle=False),
#             opt_i, criterion, device, epochs,
#             #scheduler=sched_i  # training 
#         )
#         train_losses_list.append(tl)
#         val_losses_list.append(vl)
#     train_losses = np.mean(np.stack(train_losses_list), axis=0)
#     val_losses = np.mean(np.stack(val_losses_list), axis=0)
# else:
#     train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)
#     train_losses, val_losses = train.fit(
#         model, train_loader, val_loader, optimizer, criterion, device, epochs,
#         #scheduler=scheduler  # Had to update the training script
#     )
#

# %%
from sklearn.metrics import r2_score, mean_squared_error

r2_matrix = np.zeros((len(all_results),output_dim))

output_dir = os.path.abspath(os.path.join(os.getcwd(), "../../noise_results"))
os.makedirs(output_dir, exist_ok=True)


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

    model.eval()
    predictions, true_values = [], []

    val_loader_fn = make_val_loader_fn(
        selected_observables=result["selected_observables"],
        x_dict=x_normalized_dict,
        y_vector=y,
        idx=idx_val,
        batch_size=batch_size, 
    )
    val_loader = val_loader_fn()

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            predictions.append(preds.cpu())
            true_values.append(yb.cpu())

    all_predictions = torch.cat(predictions).numpy()
    all_true_values = torch.cat(true_values).numpy()

    # Undo normalization if needed
    all_predictions = all_predictions * stds + means
    all_true_values = all_true_values * stds + means
    all_predictions[:, logflag] = np.exp(all_predictions[:, logflag])
    all_true_values[:, logflag] = np.exp(all_true_values[:, logflag])

    # Generate prediction vs true scatter plots
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

        #save r2 for summary statistics
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

    # Remove empty subplots
    for j in range(output_dim, n_rows * n_cols):
        fig.delaxes(axes.flat[j])

    fig.suptitle(f"Prediction Results — {case_name}", fontsize=16)
    fig.tight_layout()
    save_path = os.path.join(output_dir, f"{case_name}_predictions.png")
    #plt.savefig(save_path, dpi=200)
    #plt.show()
    


# %% [raw]
# rmse_matrix = np.zeros(len(all_results),output_dim)
#
# for i in range(len(all)):
#     for param_idx in range(len(val_losses_by_observable[obs_idx])):
#         val_loss_history = all_results[i][obs_idx][param_idx]
#         trues = true_values_by_observable[obs_idx][param_idx]
#         
#         mse = min(val_loss_history)
#         variance = np.var(trues)  # defaults to ddof=0 (population variance)
#         
#         rmse_matrix[obs_idx, param_idx] = mse  # store MSE for now
#         r2_matrix[obs_idx, param_idx] = 1 - (mse / variance)
#
# # Take square root to convert MSE → RMSE
# rmse_matrix = np.sqrt(rmse_matrix)

# %%
print(noise_cases.keys())

# %%
# Create a DataFrame for nicer labels (optional)

import pandas as pd
import seaborn as sns


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
#plt.savefig("../../noise_results/zs_mbh_r2.png", dpi=200)
plt.show()

# %% [markdown]
# ## Mixing and Matching Simulations

# %%
# nan (not 0) so cases skipped below never read as a real R² of 0
r2_matrix_shifted_observable_only = np.full((len(all_results), output_dim), np.nan)




for result_idx, result in enumerate(all_results):
    model = result["model"]
    selected_observables = result["selected_observables"]
    case_name = result["case_name"]

    shuffle = resolve_shuffle(selected_observables, "obs1_vs_truth")
    if shuffle is None:
        print(f"Skipping {case_name}: no {observable_2} to shuffle in this case")
        continue
    keys_to_shuffle, shuffle_y = shuffle

    print(f"Evaluating with shifted observable — Case: {case_name}")

    # Step 3: Build the validation loader using the shifted inputs
    val_loader_fn = make_val_loader_fn(
        selected_observables=selected_observables,
        x_dict= x_normalized_dict,
        y_vector= y,
        idx=idx_val,
        batch_size=batch_size,
        key_to_shuffle = keys_to_shuffle,
        perm = perm,
        shuffle_y = shuffle_y,
    )
    val_loader = val_loader_fn()

    # Step 4: Predict using the trained model
    model.eval()
    predictions, true_values = [], []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            predictions.append(preds.cpu())
            true_values.append(yb.cpu())

    all_predictions = torch.cat(predictions).numpy()
    all_true_values = torch.cat(true_values).numpy()

    # Step 5: Undo normalization
    all_predictions = all_predictions * stds + means
    all_true_values = all_true_values * stds + means
    all_predictions[:, logflag] = np.exp(all_predictions[:, logflag])
    all_true_values[:, logflag] = np.exp(all_true_values[:, logflag])

        

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

        #save r2 for summary statistics
        r2_matrix_shifted_observable_only[result_idx, i] = r2
        
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

    # Remove empty subplots
    for j in range(output_dim, n_rows * n_cols):
        fig.delaxes(axes.flat[j])

    fig.suptitle(f"Prediction Results — {case_name}", fontsize=16)
    fig.tight_layout()
    plt.show()


# %%
delta_r2_observable_only = r2_matrix_shifted_observable_only - r2_matrix
# Optional: also compute the delta for the second shuffling scenario
try:
    delta_r2_both = r2_matrix_shifted_both - r2_matrix
except NameError:
    pass


# %%
#we want observable one from simulation i and observable 2 from simulation (i+1)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Extract case names and parameter names
case_names = [result["case_name"] for result in all_results]


r2_shuffled = pd.DataFrame(
    r2_matrix_shifted_observable_only,
    index=[f"{i}" for i in noise_cases.keys()],
    columns=[f"Param {j}" for j in range(r2_matrix.shape[1])]
)

plt.figure(figsize=(16,8))
ax = sns.heatmap(
    r2_shuffled,
    vmin = -1,
    vmax = 1 , 
    annot= True,
    fmt=".1f",
    cmap="Spectral",
    linewidths=0.2,
    cbar_kws={"label":  "Validation R²"}
)

plt.xlabel("Predicted Parameter")
plt.ylabel("Noise Case")
plt.title("R² Heatmap with Observable 1 'Truths' and Shuffled 2nd Observable")
plt.tight_layout()
plt.show()


# %%
delta_r2_observable_only = r2_matrix_shifted_observable_only - r2_matrix

# %%
import matplotlib.pyplot as plt
import seaborn as sns

delta_r2_df = pd.DataFrame(
    delta_r2_observable_only,
    index=[f"{i}" for i in noise_cases.keys()],
    columns=[f"Param {j}" for j in range(r2_matrix.shape[1])]
)


plt.figure(figsize=(16, 8))
ax = sns.heatmap(
    delta_r2_df,
    annot=True,
    fmt=".2f",
    cmap="Spectral",
    center=0.0,
    linewidths=0.3,
    vmin = -0.5,
    cbar_kws={"label": "ΔR² (Original - Shifted)"}
)

plt.title("Drop in R² from Shuffling 2nd Observables and using 1st Observable 'Truth' ")
plt.xlabel("Cosmological/Feedback Parameter")
plt.ylabel("Noise/Shift Case")
plt.tight_layout()
plt.show()


# %% [markdown]
# ## Using Randomized Truth Values to Match 2nd Observable

# %%
# nan (not 0) so cases skipped below never read as a real R² of 0
r2_matrix_shifted_both = np.full((len(all_results), output_dim), np.nan)


for result_idx, result in enumerate(all_results):
    model = result["model"]
    selected_observables = result["selected_observables"]
    case_name = result["case_name"]

    shuffle = resolve_shuffle(selected_observables, "obs2_vs_truth")
    if shuffle is None:
        print(f"Skipping {case_name}: no {observable_2} to shuffle in this case")
        continue
    keys_to_shuffle, shuffle_y = shuffle

    print(f"Evaluating with shifted observable — Case: {case_name}")

    # Step 3: Build the validation loader using the shifted inputs
    val_loader_fn = make_val_loader_fn(
        selected_observables=selected_observables,
        x_dict= x_normalized_dict,
        y_vector= y,
        idx=idx_val,
        batch_size=batch_size,
        key_to_shuffle = keys_to_shuffle,
        perm = perm,
        shuffle_y = shuffle_y,
    )
    val_loader = val_loader_fn()

    # Step 4: Predict using the trained model
    model.eval()
    predictions, true_values = [], []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            predictions.append(preds.cpu())
            true_values.append(yb.cpu())

    all_predictions = torch.cat(predictions).numpy()
    all_true_values = torch.cat(true_values).numpy()

    # Step 5: Undo normalization
    all_predictions = all_predictions * stds + means
    all_true_values = all_true_values * stds + means
    all_predictions[:, logflag] = np.exp(all_predictions[:, logflag])
    all_true_values[:, logflag] = np.exp(all_true_values[:, logflag])

        

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

        #save r2 for summary statistics
        r2_matrix_shifted_both[result_idx, i] = r2
        
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

    # Remove empty subplots
    for j in range(output_dim, n_rows * n_cols):
        fig.delaxes(axes.flat[j])

    fig.suptitle(f"Prediction Results — {case_name}", fontsize=16)
    fig.tight_layout()
    plt.show()


# %%
#we want observable one from simulation i and observable 2 from simulation (i+1)
# NOTE: this cell used to re-init r2_matrix_shifted_both to zeros here, which wiped the
# loop's results before the heatmap below and plotted an all-zero grid.

# Extract case names and parameter names
case_names = [result["case_name"] for result in all_results]


r2_shuffled = pd.DataFrame(
    r2_matrix_shifted_both,
    index=[f"{i}" for i in noise_cases.keys()],
    columns=[f"Param {j}" for j in range(r2_matrix.shape[1])]
)

plt.figure(figsize=(16,8))
ax = sns.heatmap(
    r2_shuffled,
    vmin = -1,
    vmax = 1 , 
    annot= True,
    fmt=".1f",
    cmap="Spectral",
    linewidths=0.2,
    cbar_kws={"label":  "Validation R²"}
)

plt.xlabel("Predicted Parameter")
plt.ylabel("Noise Case")
plt.title("R² Heatmap: Shuffling 2nd Observable and using 2nd Observable 'Truth' ")
plt.tight_layout()
plt.show()


# %%
delta_r2_both = r2_matrix_shifted_both - r2_matrix

# %%
import matplotlib.pyplot as plt
import seaborn as sns

delta_r2_df = pd.DataFrame(
    delta_r2_both,
    index=[f"{i}" for i in noise_cases.keys()],
    columns=[f"Param {j}" for j in range(r2_matrix.shape[1])]
)


plt.figure(figsize=(16, 8))
ax = sns.heatmap(
    delta_r2_df,
    annot=True,
    fmt=".2f",
    cmap="Spectral",
    center=0.0,
    linewidths=0.3,
    vmin = -0.5,
    cbar_kws={"label": "ΔR² (Original - Shifted)"}
)

plt.title("Drop in R² from Shuffling 2nd Observables and using 2nd Observable 'Truth' ")
plt.xlabel("Cosmological/Feedback Parameter")
plt.ylabel("Noise/Shift Case")
plt.tight_layout()
plt.show()


# %%
# --- Build combined R² DataFrame: aligned + 2 shuffled cases ---
import pandas as pd
import numpy as np

required = [
    'all_results', 'r2_matrix',
    'r2_matrix_shifted_observable_only', 'r2_matrix_shifted_both'
]
if not all(name in globals() for name in required):
    raise RuntimeError("Missing required matrices; compute both shuffled cases first.")

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
            "r2_shuf_obs1": float(r2_matrix_shifted_observable_only[i, j]),  # shuffle X, keep Y
            "r2_shuf_obs2": float(r2_matrix_shifted_both[i, j])              # shuffle X and Y
        })

dual_r2_df = pd.DataFrame(rows)
dual_r2_df["delta_obs1"] = dual_r2_df["r2_aligned"] - dual_r2_df["r2_shuf_obs1"]
dual_r2_df["delta_obs2"] = dual_r2_df["r2_aligned"] - dual_r2_df["r2_shuf_obs2"]

# %%
import re

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



# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# assumes parse_case_name(...) and dual_clean_asym_order(...) are defined above

def plot_param_curve_dual(df: pd.DataFrame, param: str, figsize=(8, 5)):
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

    ax.plot(x, d["r2_aligned"].to_numpy(), marker="o", linewidth=2, label="aligned")
    ax.plot(x, d["r2_shuf_obs1"].to_numpy(), marker="s", linestyle="--", linewidth=2,
            label="shuffled (obs1 truths)")
    ax.plot(x, d["r2_shuf_obs2"].to_numpy(), marker="^", linestyle="-.", linewidth=2,
            label="shuffled (obs2 truths)")
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



# %%
focus_params = ["θ0","θ1","θ2","θ4","θ7","θ11"]
for p in focus_params:
    plot_param_curve_dual(dual_r2_df, p, figsize=(8,5))

# %%
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

def plot_predictions_vs_true(parameters,
                             noise_cases,
                             *,
                             results=all_results,
                             x_dict=x_normalized_dict,
                             y_vector=y,
                             idx=idx_val,
                             batch_size=batch_size,
                             param_labels=None,
                             save_path=None,
                             marker_size=18,
                             device=device):
    """
    Plot predicted vs. true values for selected parameters (columns of y)
    across one or more trained noise cases (in all_results).

    parameters:
      - list of parameter identifiers (int index, "θk", or label in param_labels)
      - or a single identifier
      - or "all" to include all parameters

    noise_cases:
      - list of case_name strings as in your noise_cases dict (and all_results)
      - or a single case string
      - or "all" to include all cases in all_results

    returns:
      - matplotlib Figure
    """
    if not results:
        raise ValueError("`results` is empty; train models and populate all_results first.")

    def _to_list(v):
        if v is None:
            return None
        if isinstance(v, (list, tuple, set)):
            return list(v)
        return [v]

    # Resolve parameter selection
    if isinstance(parameters, str) and parameters.lower() == "all":
        param_list = list(range(output_dim))
    else:
        param_list = _to_list(parameters)
        if not param_list:
            raise ValueError("Provide at least one parameter (index or label).")

    # Resolve case selection
    if isinstance(noise_cases, str) and noise_cases.lower() == "all":
        case_list = [r["case_name"] for r in results]
    else:
        case_list = _to_list(noise_cases)
        if not case_list:
            raise ValueError("Provide at least one case_name or 'all'.")

    # Labels: prefer user-provided labels, then global param_names, else θi
    default_labels = param_labels or globals().get("param_names") or [f"θ{i}" for i in range(output_dim)]
    label_to_idx = {label: i for i, label in enumerate(default_labels)}

    def _resolve_param(p):
        if isinstance(p, int):
            if not 0 <= p < output_dim:
                raise ValueError(f"Parameter index {p} out of range (0..{output_dim-1}).")
            return p
        if isinstance(p, str):
            # exact label
            if p in label_to_idx:
                return label_to_idx[p]
            # "θk"
            if p.startswith("θ") and p[1:].isdigit():
                i = int(p[1:])
                if not 0 <= i < output_dim:
                    raise ValueError(f"Parameter index {i} from {p} out of range.")
                return i
            # plain numeric string
            if p.isdigit():
                i = int(p)
                if not 0 <= i < output_dim:
                    raise ValueError(f"Parameter index {i} from string out of range.")
                return i
        raise ValueError(f"Cannot interpret parameter identifier: {p}")

    param_indices = [_resolve_param(p) for p in param_list]
    param_labels_resolved = [default_labels[i] for i in param_indices]

    # Build case lookup and validate
    case_to_result = {r["case_name"]: r for r in results}
    missing = [c for c in case_list if c not in case_to_result]
    if missing:
        raise ValueError(f"Cases not found in all_results: {missing}")

    # Helper: gather/cached predictions+truth for a case
    def _collect_predictions(result):
        cache_key = "_val_predictions_cache"
        if cache_key not in result:
            model = result["model"].to(device)
            loader_fn = make_val_loader_fn(
                selected_observables=result["selected_observables"],
                x_dict=x_dict,
                y_vector=y_vector,
                idx=idx,
                batch_size=batch_size,
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

            # Undo normalization/log as your training did
            pred_np = pred_np * stds + means
            true_np = true_np * stds + means
            pred_np[:, logflag] = np.exp(pred_np[:, logflag])
            true_np[:, logflag] = np.exp(true_np[:, logflag])

            result[cache_key] = (pred_np, true_np)
        return result[cache_key]

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
            ax.set_title(f"{case_name} | {p_label}\nR²={r2:.3f}, RMSE={rmse:.3f}")
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



# %%
fig = plot_predictions_vs_true(parameters="θ6", noise_cases="all",)

# %%
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

def plot_predictions_vs_true(parameters,
                             noise_cases,
                             *,
                             results=all_results,
                             x_dict=x_normalized_dict,
                             y_vector=y,
                             idx=idx_val,
                             batch_size=batch_size,
                             param_labels=None,
                             mode="aligned",              # "aligned", "obs1_vs_truth", "obs2_vs_truth"
                             keys_to_shuffle=None,        # optional explicit observable(s) to shuffle (name or list of names)
                             perm=None,                   # optional permutation; defaults to np.random.permutation(len(idx))
                             marker_size=18,
                             save_path=None,
                             device=device):
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
    default_labels = param_labels or globals().get("param_names") or [f"θ{i}" for i in range(output_dim)]
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

    # Permutation
    if perm is None:
        perm = np.random.permutation(len(idx))
    perm_tuple = tuple(map(int, perm))  # for cache key

    # Helper to choose which observables to shuffle for this case
    def _resolve_keys_to_shuffle(result):
        # delegates to the shared resolver so observable_1/observable_2 stay the
        # single source of truth
        if keys_to_shuffle is not None:
            return set(_to_list(keys_to_shuffle)), (mode == "obs2_vs_truth")
        resolved = resolve_shuffle(result["selected_observables"], mode)
        if resolved is None:
            raise ValueError(
                f"{result['case_name']} has no {observable_2} to shuffle for mode {mode!r}."
            )
        return resolved

    # Collect predictions and truths (cached per case+mode+keys+perm)
    def _collect_predictions(result):
        keys, shuffle_y_flag = _resolve_keys_to_shuffle(result)
        cache_key = ("_val_predictions_cache",
                     mode,
                     tuple(sorted(keys)),
                     perm_tuple,
                     shuffle_y_flag)
        if cache_key not in result:
            model = result["model"].to(device)
            loader_fn = make_val_loader_fn(
                selected_observables=result["selected_observables"],
                x_dict=x_dict,
                y_vector=y_vector,
                idx=idx,
                batch_size=batch_size,
                key_to_shuffle=list(keys) if keys else None,
                perm=perm if keys or shuffle_y_flag else None,
                shuffle_y=bool(shuffle_y_flag)
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

            # Undo normalization/log (as in your dataset cell)
            pred_np = pred_np * stds + means
            true_np = true_np * stds + means
            pred_np[:, logflag] = np.exp(pred_np[:, logflag])
            true_np[:, logflag] = np.exp(true_np[:, logflag])

            result[cache_key] = (pred_np, true_np)
        return result[cache_key]

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



# %%
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

def plot_param_all_val_lines(param,
                             *,
                             mode="aligned",          # "aligned", "obs1_vs_truth", "obs2_vs_truth"
                             n_sims=30,
                             seed=None,               # for reproducible sampling of sims
                             sim_indices=None,        # optional explicit indices within validation set (0..len(idx_val)-1)
                             cases="auto",            # "auto" uses ordered subset; or list of case names
                             keys_to_shuffle=None,    # optional explicit observable(s) to shuffle for obs1/obs2
                             perm=None,               # optional permutation (defaults to global 'perm' or fresh)
                             line_alpha=0.35,
                             line_width=1.2,
                             marker_size=0,
                             show_median=True,
                             results=all_results,
                             x_dict=x_normalized_dict,
                             y_vector=y,
                             idx=idx_val,
                             batch_size=batch_size,
                             param_labels=None,
                             device=device,
                             save_path=None):
    """
    For a single parameter, plot lines for randomly selected validation sims.
    Each line connects that sim’s predicted value across noise cases (ordered core only).

    mode:
      - "aligned": no shuffle (X, Y aligned)
      - "obs1_vs_truth": shuffle the first selected observable only; truths aligned
      - "obs2_vs_truth": shuffle the second selected observable only; truths aligned
    """
    if not results:
        raise ValueError("`results` is empty; train models and populate all_results first.")

    # Resolve parameter index and label
    default_labels = param_labels or globals().get("param_names") or [f"θ{i}" for i in range(output_dim)]
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
        perm = globals().get("perm", None)
        if perm is None or len(perm) != len(idx):
            perm = np.random.permutation(len(idx))
    perm = np.asarray(perm)
    if len(perm) != len(idx):
        raise ValueError("perm length must match len(idx_val).")

    # Resolve which keys to shuffle for a given case under this mode
    def _resolve_shuffle_keys(result):
        # delegates to the shared resolver so observable_1/observable_2 stay the
        # single source of truth
        if keys_to_shuffle is not None:
            keys = [keys_to_shuffle] if isinstance(keys_to_shuffle, str) else list(keys_to_shuffle)
            return keys, (mode == "obs2_vs_truth")
        resolved = resolve_shuffle(result["selected_observables"], mode)
        if resolved is None:
            raise ValueError(
                f"{result['case_name']} has no {observable_2} to shuffle for mode {mode!r}."
            )
        keys, shuffle_y = resolved
        return (sorted(keys) or None), shuffle_y

    # Cache and collect predictions/truths for full val set for a case
    def _collect_predictions(result):
        shuffle_keys, shuffle_y_flag = _resolve_shuffle_keys(result)
        cache_key = ("_all_val_param_lines_cache", mode, tuple(shuffle_keys or []), tuple(map(int, perm)), shuffle_y_flag)
        if cache_key not in result:
            model = result["model"].to(device)
            loader_fn = make_val_loader_fn(
                selected_observables=result["selected_observables"],
                x_dict=x_dict,
                y_vector=y_vector,
                idx=idx,
                batch_size=batch_size,
                key_to_shuffle=shuffle_keys,
                perm=perm if (shuffle_keys is not None or shuffle_y_flag) else None,
                shuffle_y=shuffle_y_flag,
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
            # Undo normalization/log
            pred_np = pred_np * stds + means
            true_np = true_np * stds + means
            pred_np[:, logflag] = np.exp(pred_np[:, logflag])
            true_np[:, logflag] = np.exp(true_np[:, logflag])
            result[cache_key] = (pred_np, true_np)
        return result[cache_key]

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



# %%
import numpy as np
import torch
import matplotlib.pyplot as plt

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
    labels = globals().get("param_names") or [f"θ{i}" for i in range(output_dim)]
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
    _perm = globals().get("perm")
    if _perm is None or len(_perm) != len(idx_val):
        rng_perm = np.random.default_rng(seed)
        _perm = rng_perm.permutation(len(idx_val))
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
        r2_mat = globals().get(matrix_name)
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



# %%
param_np = [0,1,2,4,6,7,8,11,16]

# %%

# %%
fig = plot_param_lines_min(0, mode="aligned", residual=True, n_sims=30, seed=42)

# %%
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

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
                       results=all_results,
                       x_dict=x_normalized_dict,
                       y_vector=y,
                       idx=idx_val,
                       batch_size=batch_size,
                       param_labels=None,
                       device=device,
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
    if not results:
        raise ValueError("`results` is empty; train models and populate all_results first.")

    # Resolve parameter index and label
    default_labels = param_labels or globals().get("param_names") or [f"θ{i}" for i in range(output_dim)]
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
        perm = globals().get("perm", None)
        if perm is None or len(perm) != len(idx):
            perm = np.random.permutation(len(idx))
    perm = np.asarray(perm)
    if len(perm) != len(idx):
        raise ValueError("perm length must match len(idx_val).")

    # Resolve which keys to shuffle for a given case under this mode
    def _resolve_shuffle_keys(result):
        # delegates to the shared resolver so observable_1/observable_2 stay the
        # single source of truth
        if keys_to_shuffle is not None:
            keys = [keys_to_shuffle] if isinstance(keys_to_shuffle, str) else list(keys_to_shuffle)
            return keys, (mode == "obs2_vs_truth")
        resolved = resolve_shuffle(result["selected_observables"], mode)
        if resolved is None:
            raise ValueError(
                f"{result['case_name']} has no {observable_2} to shuffle for mode {mode!r}."
            )
        keys, shuffle_y = resolved
        return (sorted(keys) or None), shuffle_y

    # Cache and collect predictions/truths for full val set for a case
    def _collect_predictions(result):
        shuffle_keys, shuffle_y_flag = _resolve_shuffle_keys(result)
        cache_key = ("_all_val_param_cache", mode, tuple(shuffle_keys or []), tuple(map(int, perm)), shuffle_y_flag)
        if cache_key not in result:
            model = result["model"].to(device)
            loader_fn = make_val_loader_fn(
                selected_observables=result["selected_observables"],
                x_dict=x_dict,
                y_vector=y_vector,
                idx=idx,
                batch_size=batch_size,
                key_to_shuffle=shuffle_keys,
                perm=perm if (shuffle_keys is not None or shuffle_y_flag) else None,
                shuffle_y=shuffle_y_flag,
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
            # Undo normalization/log as in your preprocessing
            pred_np = pred_np * stds + means
            true_np = true_np * stds + means
            pred_np[:, logflag] = np.exp(pred_np[:, logflag])
            true_np[:, logflag] = np.exp(true_np[:, logflag])
            result[cache_key] = (pred_np, true_np)
        return result[cache_key]

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



# %%

# %%
for p in param_np:
    fig = plot_param_all_val(f"θ{p}", mode="aligned")

# %%
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

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
    results=all_results,
    x_dict=x_normalized_dict,
    y_vector=y,
    idx=idx_val,
    batch_size=batch_size,
    param_labels=None,
    device=device,

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
    if not results:
        raise ValueError("`results` is empty; train models and populate all_results first.")

    # --- Resolve parameter index/label ---
    default_labels = param_labels or globals().get("param_names") or [f"θ{i}" for i in range(output_dim)]
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
        perm = globals().get("perm", None)
        if perm is None or len(perm) != len(idx):
            perm = np.random.permutation(len(idx))
    perm = np.asarray(perm)
    if len(perm) != len(idx):
        raise ValueError("perm length must match len(idx_val).")

    # --- Resolve which keys are shuffled for a case under this mode ---
    def _resolve_shuffle_keys(result):
        # This function reconstructs both truth alignments itself (t_obs1/t_obs2 below)
        # from which keys moved, so it never needs shuffle_y — it only needs to know
        # which observable was shuffled, which is always observable_2.
        if keys_to_shuffle is not None:
            return [keys_to_shuffle] if isinstance(keys_to_shuffle, str) else list(keys_to_shuffle)
        resolved = resolve_shuffle(result["selected_observables"], mode)
        if resolved is None:
            raise ValueError(
                f"{result['case_name']} has no {observable_2} to shuffle for mode {mode!r}."
            )
        keys, _ = resolved
        return sorted(keys) or None

    # --- Collect predictions and truth (idx-ordered) for a case ---
    def _collect_predictions(result):
        shuffle_keys = _resolve_shuffle_keys(result)
        cache_key = ("_pair_norm_values_cache",
                     mode,
                     tuple(shuffle_keys or []),
                     tuple(map(int, perm)),
                     space)
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

            # Undo standardization
            pred_np = pred_np * stds + means
            true_np = true_np * stds + means

            # Choose space
            if space == "physical":
                pred_np[:, logflag] = np.exp(pred_np[:, logflag])
                true_np[:, logflag] = np.exp(true_np[:, logflag])
            elif space == "log":
                # keep log space (no exp)
                pass
            else:
                raise ValueError("space must be 'physical' or 'log'.")

            result[cache_key] = (pred_np, true_np)
        return result[cache_key]

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



# %% [markdown]
#

# %%
plot_param_pair_normalized_values(
    "θ4",
    space="log",
    # obs_key_order now defaults to (observable_1, observable_2); the manual swap that
    # compensated for the old alphabetical convention is no longer needed
    mode="obs2_vs_truth",
    normalize_endpoints="obs1_to_obs2"    # observable_1 -> 0, observable_2 -> 1
)
plt.show()


# %% [markdown]
# ## Applying unordered pairs and variance to each noise case

# %% [markdown]
#

# %%

def make_pair_val_loader_fn(*,selected_observables,x_dict,y_vector,idx, batch_size,obs1_key,obs2_key,pairs):
    """
    Each sample corresponds to an unordered sim pair (i_row, j_row), but *inputs are directional*:
      - obs1_key features come from sim i_row
      - obs2_key features come from sim j_row
    For any extra keys (if present), we default to sim i_row.

    Returns DataLoader over:
      x_data: [n_pairs, total_features]
      y_cat:  [n_pairs, 2*output_dim] = concat([y_i, y_j]) so you have both endpoint truths per pair.
    """
    idx = np.asarray(idx)
    pairs = np.asarray(pairs, dtype=int)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pairs must be shape [n_pairs, 2].")

    n_val = len(idx)
    if np.any(pairs < 0) or np.any(pairs >= n_val):
        raise ValueError("pairs contain indices outside [0, n_val).")

    # canonicalize unordered form (i<j) and drop i==j
    pairs = np.sort(pairs, axis=1)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    if pairs.shape[0] == 0:
        raise ValueError("No valid pairs after removing i==j.")

    i_rows = pairs[:, 0]
    j_rows = pairs[:, 1]
    i_idx = idx[i_rows]  # dataset indices
    j_idx = idx[j_rows]

    # Build X in the same concatenation order as training: sorted keys
    keys_sorted = sorted(selected_observables.keys())

    x_list = []
    for key in keys_sorted:
        if key == obs1_key:
            arr = x_dict[key][i_idx]
        elif key == obs2_key:
            arr = x_dict[key][j_idx]
        else:
            # if any extra key exists, default to sim i
            arr = x_dict[key][i_idx]
        x_list.append(torch.from_numpy(arr).float())

    x_data = torch.cat(x_list, dim=1)

    # Store both truths (for endpoints)
    y_i = y_vector[i_idx]
    y_j = y_vector[j_idx]
    if isinstance(y_i, np.ndarray):
        y_i = torch.from_numpy(y_i).float()
    if isinstance(y_j, np.ndarray):
        y_j = torch.from_numpy(y_j).float()

    y_cat = torch.cat([y_i, y_j], dim=1)  # [n_pairs, 2*output_dim]

    return DataLoader(TensorDataset(x_data, y_cat), batch_size=batch_size, shuffle=False)




# %%
# ============================================================
# 2) PAIR SAMPLER: unique unordered pairs (avoid (i,j)/(j,i) duplicates)
# ============================================================
def sample_unique_unordered_pairs(n_val, n_pairs, *, seed=0):
    """
    Returns pairs array shape [n_pairs, 2] with i<j, unique, sampled randomly.
    """
    max_pairs = n_val * (n_val - 1) // 2
    if n_pairs > max_pairs:
        raise ValueError(f"Requested n_pairs={n_pairs} exceeds C(n_val,2)={max_pairs}.")

    rng = np.random.default_rng(seed)
    seen = set()
    out = []

    while len(out) < n_pairs:
        m = max(2000, 3 * (n_pairs - len(out)))
        a = rng.integers(0, n_val, size=m)
        b = rng.integers(0, n_val, size=m)

        for i, j in zip(a, b):
            if i == j:
                continue
            if i > j:
                i, j = j, i
            key = (int(i), int(j))
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
            if len(out) >= n_pairs:
                break

    return np.array(out, dtype=int)


def inspect_unordered_pair_denominator_distribution(
    param,
    *,
    obs1_key=observable_2,
    obs2_key=observable_1,
    normalize_endpoints="obs1_to_obs2",
    n_pairs=5000,
    seed=0,
    pairs=None,
    cases="auto",
    space="processed",
    bins=60,
    min_abs_denom=None,
    results=all_results,
    x_dict=x_normalized_dict,
    y_vector=y,
    idx=idx_val,
    batch_size=batch_size,
    param_labels=None,
    device=device,
):
    if not results:
        raise ValueError("`results` is empty; train models and populate all_results first.")

    default_labels = param_labels or globals().get("param_names") or [f"θ{i}" for i in range(output_dim)]
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
    idx=idx_val,
    y_vector=y,
    param_labels=None,
):
    default_labels = param_labels or globals().get("param_names") or [f"θ{i}" for i in range(output_dim)]
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
    obs1_key=observable_2,
    obs2_key=observable_1,

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

    results=all_results,
    x_dict=x_normalized_dict,
    y_vector=y,
    idx=idx_val,
    batch_size=batch_size,
    param_labels=None,
    device=device,

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
    if not results:
        raise ValueError("`results` is empty; train models and populate all_results first.")

    # -----------------------------
    # Resolve parameter index/label
    # -----------------------------
    default_labels = param_labels or globals().get("param_names") or [f"θ{i}" for i in range(output_dim)]
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
        r2_mat = globals().get("r2_matrix")
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

# %%


for p in list(param_np):
    fig, stats = plot_param_unordered_pair_normalized_values(
    param=p,
    obs1_key=observable_2,
    obs2_key=observable_1,
    space="processed",
    min_abs_denom=1.5,
)
    plt.show()


# %%
def plot_param_unordered_pair_decomposed_values(
    param,
    *,
    obs1_key=observable_2,
    obs2_key=observable_1,
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

    results=all_results,
    x_dict=x_normalized_dict,
    y_vector=y,
    idx=idx_val,
    batch_size=batch_size,
    param_labels=None,
    device=device,

    save_path=None,
):
    if not results:
        raise ValueError("`results` is empty; train models and populate all_results first.")

    obs1_label = "observable 1"
    obs2_label = "observable 2"

    default_labels = param_labels or globals().get("param_names") or [f"θ{i}" for i in range(output_dim)]
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
        r2_mat = globals().get("r2_matrix")
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


# %%
for p in params_example:
    fig, stats = plot_param_unordered_pair_decomposed_values(
        param=p,
        obs1_key=observable_2,
        obs2_key=observable_1,
        space="processed",
        mode="absolute",
        min_abs_denom=None,
    )
    plt.show()

# %%
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

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
    results=all_results,
    x_dict=x_normalized_dict,
    y_vector=y,
    idx=idx_val,
    batch_size=batch_size,
    device=device,
):
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
        # both comparison modes shuffle obs2; this function reads the truth alignment
        # off the shuffled/unshuffled split itself, so shuffle_y stays False
        if mode == "aligned":
            return None
        if keys_to_shuffle is not None:
            return [keys_to_shuffle] if isinstance(keys_to_shuffle, str) else list(keys_to_shuffle)
        if mode in ("obs1_vs_truth", "obs2_vs_truth"):
            return [obs2_key]
        raise ValueError("mode must be 'aligned', 'obs1_vs_truth', or 'obs2_vs_truth'.")

    shuffle_keys = _resolve_shuffle_keys()

    if perm is None and shuffle_keys is not None:
        perm = globals().get("perm", None)
        if perm is None or len(perm) != len(idx):
            perm = np.random.permutation(len(idx))
    if perm is not None:
        perm = np.asarray(perm)
        if len(perm) != len(idx):
            raise ValueError("perm length must match len(idx).")

    cache_key = (
        "_constraint_2d_cache_v1",
        case_name,
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
        obs1_arr = np.asarray(x_dict[obs1_key][idx_arr], copy=True)
        obs2_arr = np.asarray(x_dict[obs2_key][idx_arr], copy=True)

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
    x_dict=x_normalized_dict,
    results=all_results,
    y_vector=y,
    idx=idx_val,
    batch_size=batch_size,
    param_labels=None,
    device=device,
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
    default_labels = param_labels or globals().get("param_names") or [f"θ{i}" for i in range(output_dim)]
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



# %%
fig, stats = plot_constraint_structure_2d(
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
def plot_unordered_pair_prediction_map(
    param,
    *,
    obs1_key=observable_2,
    obs2_key=observable_1,
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
    results=all_results,
    x_dict=x_normalized_dict,
    y_vector=y,
    idx=idx_val,
    batch_size=batch_size,
    param_labels=None,
    device=device,
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
    if not results:
        raise ValueError("`results` is empty; train models and populate all_results first.")

    default_labels = param_labels or globals().get("param_names") or [f"θ{i}" for i in range(output_dim)]
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



# %%
fig, stats = plot_unordered_pair_prediction_map(
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
