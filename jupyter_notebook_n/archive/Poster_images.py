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
#     display_name: Python 3 (ipykernel)
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
import seaborn as sns

# %%

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",

    # Bigger title & labels
    "axes.titlesize": 42,      # was 36
    "axes.labelsize": 36,      # was 30

    # Slightly smaller tick labels
    "xtick.labelsize": 20,     # was 24
    "ytick.labelsize": 20,     # was 24

    # Legend a touch smaller than labels
    "legend.fontsize": 22,

    # Spacing / aesthetics
    "axes.titlepad": 16,
    "axes.labelpad": 12,
    "lines.linewidth": 2.5,
})

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

with h5py.File(datafilename, 'r') as f:
    Ms_Mh_s90 = f['Ms_Mh_s90'][:].T
    Ms_Mh_s61 = f['Ms_Mh_s61'][:].T
    MBH_Mh_s90 = f['MBH_Mh_s90'][:].T
    MBH_Mh_s61 = f['MBH_Mh_s61'][:].T
    Mg_Mh_s90 = f['Mg_Mh_s90'][:].T
    Mg_Mh_s61 = f['Mg_Mh_s61'][:].T
    Rs_Ms_s90 = f['Rs_Ms_s90'][:].T
    Rs_Ms_s61 = f['Rs_Ms_s61'][:].T
    SFR_Ms_s90 = f['SFR_Ms_s90'][:].T
    SFR_Ms_s61 = f['SFR_Ms_s61'][:].T
    Zs_Ms_s90 = f['Zs_Ms_s90'][:].T
    Zs_Ms_s61 = f['Zs_Ms_s61'][:].T
    SFRH_100Myr = f['SFRH_100Myr'][:].T
    logMh_s61 = f['logMh_s61'][:].T  
    logMh_s90 = f['logMh_s90'][:].T
    logMs_s61 = f['logMs_s61'][:].T
    logMs_s90 = f['logMs_s90'][:].T

# %%
print(Parameters.shape)

# %%
mbh_mh_avgs = np.mean(MBH_Mh_s61, axis = 0)
mbh_mh_std = np.std(MBH_Mh_s61, axis = 0)

mg_mh_avgs = np.mean(Mg_Mh_s61, axis = 0)
mg_mh_std = np.std(Mg_Mh_s61, axis = 0)




# %%
print(len(mbh_mh_avgs))

# %%
print(len(mbh_mh_std))

# %%
import numpy as np
import matplotlib.pyplot as plt

def centers_if_edges(x, n_expected):
    x = np.asarray(x)
    if x.ndim != 1:
        # Flatten any 2D/column/row shapes
        x = x.ravel()
    if x.size == n_expected + 1:             # edges → centers
        x = 0.5 * (x[:-1] + x[1:])
    return x

def coerce_vec(a, name):
    a = np.asarray(a)
    if a.ndim > 1:
        a = a.ravel()                         # (N,1) → (N,)
    # ensure numeric
    return a.astype(float)

def coerce_yerr(y, yerr, name):
    if yerr is None:
        return None
    y = np.asarray(y)
    e = np.asarray(yerr, dtype=float)
    if e.ndim == 0:
        return np.full_like(y, float(e))
    if e.ndim == 1:
        if e.size != y.size:
            raise ValueError(f"{name} yerr length {e.size} != y length {y.size}")
        return e
    if e.ndim == 2 and e.shape == (2, y.size):  # asymmetric
        return e
    raise ValueError(f"{name} yerr shape {e.shape} not compatible with y length {y.size}")

# --- Prepare x against BOTH series (handles edges/centers and shape) ---
N_mbh = np.asarray(mbh_mh_avgs).size
N_mg  = np.asarray(mg_mh_avgs).size
if N_mbh != N_mg:
    raise ValueError(f"mbh_mh_avgs length {N_mbh} != mg_mh_avgs length {N_mg}")

x = centers_if_edges(logMh_s61, n_expected=N_mbh)
x = coerce_vec(x, "x")

# --- Clean y and yerr ---
y_mbh = coerce_vec(mbh_mh_avgs, "mbh y")
e_mbh = coerce_yerr(y_mbh, mbh_mh_std, "mbh")

y_mg  = coerce_vec(mg_mh_avgs, "mg y")
e_mg  = coerce_yerr(y_mg, mg_mh_std, "mg")

# Final sanity: exact 1-D shape matches
if x.shape != y_mbh.shape:
    raise ValueError(f"x shape {x.shape} != mbh y shape {y_mbh.shape}")
if x.shape != y_mg.shape:
    raise ValueError(f"x shape {x.shape} != mg y shape {y_mg.shape}")

# --- Plot (poster sizes) ---
fig, ax = plt.subplots(figsize=(13, 10), constrained_layout=True)
ax.set_title("Halo Mass Scaling Relations", fontsize=44, pad=16)
ax.set_xlabel(r"$\log(M_h/M_\odot)$", fontsize=38, labelpad=12)
ax.set_ylabel(r"Mean Mass per $M_h$ Bin [$M_\odot$]", fontsize=38, labelpad=12)

ax.errorbar(x, y_mbh, yerr=e_mbh, fmt='o-', capsize=5, elinewidth=2.0, markersize=9,
            label=r"$M_{\rm BH}/M_h$")
ax.errorbar(x, y_mg,  yerr=e_mg,  fmt='s--', capsize=5, elinewidth=2.0, markersize=9,
            label=r"$M_{\rm g}/M_h$")

ax.minorticks_on()
ax.tick_params(axis="both", which="major", labelsize=22, direction="in", length=7, width=1.4)
ax.tick_params(axis="both", which="minor", labelsize=18, direction="in", length=4, width=1.0)
ax.grid(True, which="both", ls="--", alpha=0.6, linewidth=1.2)

leg = ax.legend(frameon=True, markerscale=1.2)
for txt in leg.get_texts():
    txt.set_fontsize(26)

plt.show()


# %%
origParameters[1000]

# %%

# %%
PartiallyLoggedParameters[1000]

# %%
observable_block = {
    "Ms_Mh_s61" : torch.from_numpy(Ms_Mh_s61 ).float(),
    "Ms_Mh_s90" : torch.from_numpy(Ms_Mh_s90).float(),
    "MBH_Mh_s90": torch.from_numpy(MBH_Mh_s90).float(),
    "MBH_Mh_s61": torch.from_numpy(MBH_Mh_s61).float(),
    "Mg_Mh_s90": torch.from_numpy(Mg_Mh_s90 ).float(),
    "Mg_Mh_s61": torch.from_numpy(Mg_Mh_s61 ).float(),
    "Rs_Ms_s90": torch.from_numpy(Rs_Ms_s90 ).float(),
    "Rs_Ms_s61": torch.from_numpy(Rs_Ms_s61 ).float(),
    "SFR_Ms_s90": torch.from_numpy(SFR_Ms_s90).float(),
    "SFR_Ms_s61": torch.from_numpy(SFR_Ms_s61).float(),
    "Zs_Ms_s90": torch.from_numpy(Zs_Ms_s90 ).float(),
    "Zs_Ms_s61": torch.from_numpy(Zs_Ms_s61 ).float(),
    "SFRH_100Myr":torch.from_numpy(SFRH_100Myr).float(),
}

# %%

# %%
noise_cases = {
    # Mg noisy, MBH clean
    "mg_5.0_mbh_0.0": {
        "Mg_Mh_s61": 5.0,
        "MBH_Mh_s61": 0.0
    },

    "mg_1.0_mbh_0.0": {
        "Mg_Mh_s61": 1.0,
        "MBH_Mh_s61": 0.0
    },

    # Both clean
    "mg_0.0_mbh_0.0": {
        "Mg_Mh_s61": 0.0,
        "MBH_Mh_s61": 0.0
    },

    # Mg clean, MBH noisy
    "mg_0.0_mbh_1.0": {
        "Mg_Mh_s61": 0.0,
        "MBH_Mh_s61": 1.0
    },
    "mg_0.0_mbh_5.0": {
        "Mg_Mh_s61": 0.0,
        "MBH_Mh_s61": 5.0
    },

    # Both noisy — diagonal cases
    "mg_1.0_mbh_1.0": {
        "Mg_Mh_s61": 1.0,
        "MBH_Mh_s61": 1.0
    },
    
    "mg_2.5_mbh_2.5":{
        "Mg_Mh_s61":2.5,
        "MBH_Mh_s61":2.5
    },
    
    "mg_5.0_mbh_5.0": {
        "Mg_Mh_s61": 5.0,
        "MBH_Mh_s61": 5.0
    },

    # Mg only
    "mg_clean": {
        "Mg_Mh_s61": 0.0
    },

    # MBH only
    "mbh_clean": {
        "MBH_Mh_s61": 0.0
    }
}


# %%
heatmap_one = {
   # MBH only
    "MBH_Clean": {
        "MBH_Mh_s61": 0.0
    },
    
    # Mg noisy, MBH clean
    "mg_5.0_mbh_0.0": {
        "Mg_Mh_s61": 5.0,
        "MBH_Mh_s61": 0.0
    },

    "mg_1.0_mbh_0.0": {
        "Mg_Mh_s61": 1.0,
        "MBH_Mh_s61": 0.0
    },

    # Both clean
    "mg_0.0_mbh_0.0": {
        "Mg_Mh_s61": 0.0,
        "MBH_Mh_s61": 0.0
    },

    # Mg clean, MBH noisy
    "mg_0.0_mbh_1.0": {
        "Mg_Mh_s61": 0.0,
        "MBH_Mh_s61": 1.0
    },
    "mg_0.0_mbh_5.0": {
        "Mg_Mh_s61": 0.0,
        "MBH_Mh_s61": 5.0
    },
    # Mg only
    "mg_clean": {
        "Mg_Mh_s61": 0.0
    },

    
}


# %%
heatmap_two = {
     # Both clean
    "mg_0.0_mbh_0.0": {
        "Mg_Mh_s61": 0.0,
        "MBH_Mh_s61": 0.0
    },
    
    # Both noisy — diagonal cases
     "mg_1.0_mbh_1.0": {
        "Mg_Mh_s61": 1.0,
        "MBH_Mh_s61": 1.0
    },
    
    "mg_2.5_mbh_2.5":{
        "Mg_Mh_s61":2.5,
        "MBH_Mh_s61":2.5
    },
    
    "mg_5.0_mbh_5.0": {
        "Mg_Mh_s61": 5.0,
        "MBH_Mh_s61": 5.0
    },
}

# %%
for case in noise_cases.values(): 
    print(case.keys())

# %%
all_observables = set()
for case in noise_cases.values():
    all_observables.update(case.keys())

x_raw_dict = {key: observable_block[key].numpy() for key in all_observables}


# %%
def shuffle_observable(obs_dict,keys_to_shift, perm):
    shifted_dict = obs_dict.copy()
    
    for keys in keys_to_shift:
        shifted_dict[keys] = obs_dict[keys][perm]

    return shifted_dict


# %%
def add_noise(array_np, noise_level=0.0):
    noise = np.random.normal(loc=0.0, scale=noise_level, size=array_np.shape)
    array_np += noise
    return array_np  


# %%
def normalize(array_np):
    mean = np.mean(array_np, axis = 0)
    std = np.std(array_np, axis = 0)
    array_np = (array_np - mean) / std

    return array_np


# %%
mg_clean = np.concatenate(Mg_Mh_s61, axis=0)
mbh_clean = np.concatenate(MBH_Mh_s61, axis = 0)

# %%
np.max(Mg_Mh_s61)

# %%
import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)

# Same KDEs as before
for sigma in [0.0, 1.0, 2.5, 5.0]:
    mg_noisy = add_noise(mg_clean, sigma)
    sns.kdeplot(
        mg_noisy,
        label=rf"$\sigma = {sigma}$",
        ax=ax,
        linewidth=2.2,   # just a touch thicker for posters
    )

# Labels/titles (explicit sizes so nothing overrides them)
ax.set_xlabel("Mg/Mh", fontsize=34, labelpad=8)
ax.set_ylabel("Density", fontsize=34, labelpad=8)
ax.set_title("Effect of Noise on Mg/Mh Distribution", fontsize=40, pad=14)

# Ticks
ax.tick_params(axis="both", labelsize=22)

# Legend: outside the plot, opaque white box, readable size
leg = ax.legend(
    title=r"Noise level ($\sigma$)",
    bbox_to_anchor=(1.02, 1), loc="upper left",
    borderaxespad=0.0,
    frameon=True, framealpha=1.0, facecolor="white", edgecolor="black"
)
leg.get_title().set_fontsize(24)
for t in leg.get_texts():
    t.set_fontsize(22)

# Optional grid
ax.grid(True, ls="--", alpha=0.5)

plt.show()


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
perm = torch.randperm(len(x_clean_tensor))
idx_train = perm[:-n_val]
idx_val = perm[-n_val:]

x_val, y_val = x_clean_tensor[idx_val], y[idx_val]
val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=64, shuffle=False)

# %%
#we randomize the indices of the validation set 

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
            x_proc = add_noise(arr, noise_level)  # Add noise and normalize
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
    idx = np.asarray(idx)
    
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
    


# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# -----------------------------
# 1) Define param labels
# -----------------------------
param_labels = {
    0:  r"$\Omega_m$",         # matter density
    4:  r"$n_s$",              # scalar spectral index
    6:  r"$\beta_{H}$",        # UVB H amplitude
    7:  r"$\Delta z_{H}$",     # UVB H reionization shift
    8:  r"$\beta_{\mathrm{He}}$", # UVB He amplitude
    28: r"$A_{\mathrm{Edd}}$", # BH Eddington factor
}

keep_cols = [f"Param {i}" for i in param_labels.keys()]

# -----------------------------
# 2) Define noise case labels
#    (example mapping for Mg & MBH — expand as needed)
# -----------------------------
noise_case_labels = {
    "mg_5.0_mbh_0.0": "Mg (high noise)",
    "mg_1.0_mbh_0.0": "Mg (low noise)",
    "mg_0.0_mbh_0.0": "Both clean",
    "mg_0.0_mbh_1.0": "MBH (low noise)",
    "mg_0.0_mbh_5.0": "MBH (high noise)",
    "mg_1.0_mbh_1.0": "Both noisy (low)",
    "mg_5.0_mbh_5.0": "Both noisy (high)",    
} 


# %%
for result in all_results:
    print(all_results.index(result))

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

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------
# 1) Build the full R^2 DataFrame
# -------------------------
r2_df = pd.DataFrame(
    r2_matrix,
    index=[f"{i}" for i in noise_cases.keys()],
    columns=[f"Param {j}" for j in range(r2_matrix.shape[1])]
)

# -------------------------
# 2) Keep only your six params and relabel columns with LaTeX
# -------------------------
param_labels = {
    0:  r"$\Omega_m$",
    4:  r"$n_s$",
    6:  r"$\beta_{H}$",
    7:  r"$\Delta z_{H}$",
    8:  r"$\beta_{\mathrm{He}}$",
    28: r"$A_{\mathrm{Edd}}$",
}
keep_cols = [f"Param {i}" for i in param_labels.keys()]
r2_df = r2_df[keep_cols].rename(columns={f"Param {i}": lab for i, lab in param_labels.items()})

# -------------------------
# 3) Reader-friendly labels for cases + your desired order
#    (high → clean → low → both-noisy at end)
# -------------------------
noise_case_labels = {
    "mg_5.0_mbh_0.0": "Mg (high noise)",
    "mg_0.0_mbh_5.0": "MBH (high noise)",
    "mg_0.0_mbh_0.0": "Both clean",
    "mg_1.0_mbh_0.0": "Mg (low noise)",
    "mg_0.0_mbh_1.0": "MBH (low noise)",
    "mg_5.0_mbh_5.0": "Both noisy (high)",
    "mg_1.0_mbh_1.0": "Both noisy (low)",
    # Optional singles:
    "MBH_Clean": "MBH only (clean)",
    "mg_clean":  "Mg only (clean)",
}

noise_order = [
    "Mg (high noise)", "MBH (high noise)",
    "Both clean",
    "Mg (low noise)", "MBH (low noise)",
    "Both noisy (high)", "Both noisy (low)",
    "MBH only (clean)", "Mg only (clean)",  # keep singles at the very end if present
]

# Apply labels; keep originals if not found
r2_df.index = [noise_case_labels.get(k, k) for k in r2_df.index]
# Enforce order for any labels that exist
present_order = [lab for lab in noise_order if lab in r2_df.index]
r2_df = r2_df.reindex(present_order)

# -------------------------
# 4) Plot poster-friendly heatmap (all cases)
# -------------------------
plt.figure(figsize=(16, 8))
ax = sns.heatmap(
    r2_df,
    vmin=-1.0, vmax=1.0, cmap="Spectral",
    annot=True, fmt=".2f",
    linewidths=0.2,
    cbar_kws={'label': r'Validation $R^{2}$'}
)
sns.set_style("white")

# Font sizes for poster readability
ax.set_title(r"Constraining Power Heatmap (Validation $R^{2}$)", fontsize=24, pad=12)
ax.set_ylabel("Noise/Shift Case", fontsize=18)
ax.set_xlabel("Parameter", fontsize=18)
ax.tick_params(axis="x", labelrotation=25, labelsize=14)
ax.tick_params(axis="y", labelsize=14)
ax.figure.axes[-1].yaxis.label.set_size(16)  # colorbar label

plt.tight_layout()
plt.show()


# %%
Parameters 0,4,6,7,8,28

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ---- columns to show + nice labels ----
param_labels = {
    0:  r"$\Omega_m$",
    4:  r"$n_s$",
    6:  r"$\beta_{H}$",
    7:  r"$\Delta z_{H}$",
    8:  r"$\beta_{\mathrm{He}}$",
    28: r"$A_{\mathrm{Edd}}$",
}
keep_cols   = [f"Param {i}" for i in param_labels]
col_rename  = {f"Param {i}": lab for i, lab in param_labels.items()}

# ---- base R^2 dataframe from your data ----
r2_df_base = pd.DataFrame(
    r2_matrix,
    index=[f"{i}" for i in noise_cases.keys()],
    columns=[f"Param {j}" for j in range(r2_matrix.shape[1])]
)[keep_cols].rename(columns=col_rename)

# ---- exact orders ----
heatmap_one_order = [
    "mbh_clean",
    "mg_5.0_mbh_0.0",
    "mg_1.0_mbh_0.0",
    "mg_0.0_mbh_0.0",
    "mg_0.0_mbh_1.0",
    "mg_0.0_mbh_5.0",
    "mg_clean",
]
heatmap_two_order = [
    "mg_0.0_mbh_0.0",
    "mg_1.0_mbh_1.0",
    "mg_2.5_mbh_2.5",
    "mg_5.0_mbh_5.0",
]

# ---- readable labels ----
label_map = {
    "mbh_clean":        "MBH only (clean)",
    "mg_clean":         "Mg only (clean)",
    "mg_0.0_mbh_0.0":   "Both clean",
    "mg_1.0_mbh_0.0":   "Mg (low noise), MBH clean",
    "mg_5.0_mbh_0.0":   "Mg (high noise), MBH clean",
    "mg_0.0_mbh_1.0":   "MBH (low noise), Mg clean",
    "mg_0.0_mbh_5.0":   "MBH (high noise), Mg clean",
    "mg_1.0_mbh_1.0":   "Both noisy (σ=1.0)",
    "mg_2.5_mbh_2.5":   "Both noisy (σ=2.5)",
    "mg_5.0_mbh_5.0":   "Both noisy (σ=5.0)",
}

def build_subset(df_base, ordered_keys):
    sub = df_base.loc[ordered_keys].copy()
    sub.index = [label_map.get(k, k) for k in sub.index]
    return sub

def plot_heatmap(
    df, title, vmin=-1.0, vmax=1.0,
    title_fs=56, label_fs=44, tick_fs=32, annot_fs=24,
    cbar_label_fs=40, cbar_tick_fs=30
):
    plt.figure(figsize=(22, 13))
    ax = sns.heatmap(
        df, vmin=vmin, vmax=vmax, cmap='Spectral',
        annot=True, fmt=".2f", linewidths=0.3,
        annot_kws={"size": annot_fs},
        cbar_kws={'label': r'Validation $R^{2}$'}
    )
    sns.set_style("white")

    ax.set_title(title, fontsize=title_fs, pad=20)
    ax.set_ylabel("Noise/Shift Case", fontsize=label_fs, labelpad=12)
    ax.set_xlabel("Parameter", fontsize=label_fs, labelpad=12)

    ax.tick_params(axis="x", labelrotation=25, labelsize=tick_fs)
    ax.tick_params(axis="y", labelrotation=0,  labelsize=tick_fs)

    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_size(cbar_label_fs)
    cbar.ax.tick_params(labelsize=cbar_tick_fs)

    plt.tight_layout()
    plt.show()

# ---- build and plot ----
r2_one = build_subset(r2_df_base, heatmap_one_order)
plot_heatmap(r2_one, r"Validation $R^{2}$ — Single-Observable Noise & Baselines")

r2_two = build_subset(r2_df_base, heatmap_two_order)
plot_heatmap(r2_two, r"Validation $R^{2}$ — Diagonal Noise (Both Observables)")


# %% [markdown]
# ## Mixing and Matching Simulations

# %%
print(x_normalized_dict)

# %%
r2_matrix_shifted = np.zeros((len(all_results), output_dim))




for result_idx, result in enumerate(all_results):
    model = result["model"]
    selected_observables = result["selected_observables"]
    case_name = result["case_name"]

    print(f"Evaluating with shifted observable — Case: {case_name}")
    
    # Step 3: Build the validation loader using the shifted inputs
    val_loader_fn = make_val_loader_fn(
        selected_observables=selected_observables,
        x_dict= x_normalized_dict,
        y_vector= y,
        idx=idx_val,
        batch_size=batch_size,
        key_to_shuffle = "MBH_Mh_s61", 
        perm = perm,
        shuffle_y = False
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
        r2_matrix_shifted[result_idx, i] = r2
        
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

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Extract case names and parameter names
case_names = [result["case_name"] for result in all_results]


r2_shuffled = pd.DataFrame(
    r2_matrix_shifted,
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
delta_r2 =r2_matrix_shifted - r2_matrix

# %%
import matplotlib.pyplot as plt
import seaborn as sns

delta_r2_df = pd.DataFrame(
    delta_r2,
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


# %%
# Plot (poster fonts — much larger)
fig, ax = plt.subplots(figsize=(26, 15), constrained_layout=True)
sns.barplot(
    data=delta_r2_long,
    x="Noise/Shift Case",
    y="ΔR²",
    hue="Parameter",
    palette="Spectral",
    order=order,
    ax=ax
)

ax.axhline(0, color="black", linewidth=1.2, linestyle="--")

ax.set_title(r"Drop in R² from mixing observables and using the parameters from $M_g$ as truth",
             fontsize=64, pad=22)
ax.set_xlabel("Noise Case for Observable Combination", fontsize=52, labelpad=16)
ax.set_ylabel(r"$\Delta R^2$ (Original - Shifted)", fontsize=52, labelpad=16)

ax.tick_params(axis="x", labelrotation=25, labelsize=36)
ax.tick_params(axis="y", labelsize=36)

leg = ax.legend(
    title="Cosmological/Feedback Parameter",
    bbox_to_anchor=(1.02, 1), loc="upper left",
    frameon=True
)
leg.get_title().set_fontsize(44)
for t in leg.get_texts():
    t.set_fontsize(36)

plt.show()


# %% [markdown]
# ## Using Randomized Truth Values to Match 2nd Observable

# %%
r2_matrix_shifted = np.zeros((len(all_results), output_dim))


for result_idx, result in enumerate(all_results):
    model = result["model"]
    selected_observables = result["selected_observables"]
    case_name = result["case_name"]

    print(f"Evaluating with shifted observable — Case: {case_name}")
    
    # Step 3: Build the validation loader using the shifted inputs
    val_loader_fn = make_val_loader_fn(
        selected_observables=selected_observables,
        x_dict= x_normalized_dict,
        y_vector= y,
        idx=idx_val,
        batch_size=batch_size,
        key_to_shuffle = "MBH_Mh_s61", 
        perm = perm,
        shuffle_y = True
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
        r2_matrix_shifted[result_idx, i] = r2
        
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

# %%
#we want observable one from simulation i and observable 2 from simulation (i+1)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Extract case names and parameter names
case_names = [result["case_name"] for result in all_results]


r2_shuffled = pd.DataFrame(
    r2_matrix_shifted,
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
delta_r2 =r2_matrix_shifted - r2_matrix

# %%
import matplotlib.pyplot as plt
import seaborn as sns

delta_r2_df = pd.DataFrame(
    delta_r2,
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

plt.title(r"Drop in R² from mixing observables and using the parameters from $MBH$ as truth")
plt.xlabel("Cosmological/Feedback Parameter")
plt.ylabel("Noise/Shift Case")
plt.tight_layout()
plt.show()


# %%

# %%
# Plot (poster fonts — much larger)
fig, ax = plt.subplots(figsize=(26, 15), constrained_layout=True)
sns.barplot(
    data=delta_r2_long,
    x="Noise/Shift Case",
    y="ΔR²",
    hue="Parameter",
    palette="Spectral",
    order=order,
    ax=ax
)

ax.axhline(0, color="black", linewidth=1.2, linestyle="--")

ax.set_title(r"Drop in R² from mixing observables and using the parameters from $M_g$ as truth",
             fontsize=64, pad=22)
ax.set_xlabel("Noise Case for Observable Combination", fontsize=52, labelpad=16)
ax.set_ylabel(r"$\Delta R^2$ (Original - Shifted)", fontsize=52, labelpad=16)

ax.tick_params(axis="x", labelrotation=25, labelsize=36)
ax.tick_params(axis="y", labelsize=36)

leg = ax.legend(
    title="Cosmological/Feedback Parameter",
    bbox_to_anchor=(1.02, 1), loc="upper left",
    frameon=True
)
leg.get_title().set_fontsize(44)
for t in leg.get_texts():
    t.set_fontsize(36)

plt.show()


# %%

# %%
