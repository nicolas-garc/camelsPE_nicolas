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
#     display_name: Python (py311-main)
#     language: python
#     name: py311-main
# ---

# %%
import sys
import importlib
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import random
import optuna
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
base_path = "../src/"
sys.path.append(base_path)
import models
import train
from losses import *

# %%
conda install optuna

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
datafilename='../../DATA/data_L25LH_TNG.hdf5'
#datafilename='../../DATA/data_L50_TNG_v3.hdf5'
with h5py.File(datafilename, 'r') as f:
    print("Datasets available:")
    for key in f.keys():
        print(key)

with h5py.File(datafilename, 'r') as f:
    #Parameters = f['Parameters'][0, :1024].T.reshape(-1, 1)
    Parameters = f['Parameters'][:, :1024].T
#logflag = np.array([False])
#logflag = np.array([False, False, True, True, True, True])
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
#print(Parameters.shape)
#print(Parameters.dtype)
#print(Parameters.min(axis=0))
#print(Parameters.max(axis=0))

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

# %%
#x = torch.from_numpy(Ms_Mh_s90).float()
x = torch.cat((torch.from_numpy(Ms_Mh_s90), torch.from_numpy(Ms_Mh_s61)), dim=1).float()
#x = torch.cat((torch.from_numpy(Ms_Mh_s90), torch.from_numpy(Ms_Mh_s61), torch.from_numpy(MBH_Mh_s90), torch.from_numpy(MBH_Mh_s61), torch.from_numpy(Mg_Mh_s90), torch.from_numpy(Mg_Mh_s61)), dim=1).float()
#x = torch.cat((torch.from_numpy(Ms_Mh_s90), torch.from_numpy(Ms_Mh_s61), torch.from_numpy(MBH_Mh_s90), torch.from_numpy(MBH_Mh_s61), torch.from_numpy(Mg_Mh_s90), torch.from_numpy(Mg_Mh_s61), torch.from_numpy(Rs_Ms_s90), torch.from_numpy(Rs_Ms_s61), torch.from_numpy(SFR_Ms_s90), torch.from_numpy(SFR_Ms_s61), torch.from_numpy(Zs_Ms_s90), torch.from_numpy(Zs_Ms_s61), torch.from_numpy(SFRH_100Myr)), dim=1).float()

normalize_inputs = True

if normalize_inputs:
    x_np = x.numpy()
    x_means = x_np.mean(axis=0)
    x_stds = x_np.std(axis=0)
    x_np_norm = (x_np - x_means) / x_stds
    x = torch.from_numpy(x_np_norm).float()


print("Total NaNs:", torch.isnan(x).sum().item())
print("Total Infs:", torch.isinf(x).sum().item())
x[torch.isnan(x)] = 11
x[torch.isinf(x)] = 11
print("Total NaNs:", torch.isnan(x).sum().item())
print("Total Infs:", torch.isinf(x).sum().item())
y = torch.from_numpy(Parameters).float()
print(x.shape)
print(y.shape)
#print(x.dtype)
#print(y.dtype)
#print(x[1:3,:])
#print(y[1:3,:])

# %% [raw]
# # Hyperparameters
# input_dim    = x.shape[1]
# output_dim   = y.shape[1]
# hidden_dims  = [128, 64]
# #hidden_dims  = [128, 64, 64]
# #hidden_dims  = [64, 64]
# lr           = 1e-4
# wd           = 1e-5
# dropout_rate = 0.2
# epochs       = 2000
# val_fraction = 0.1
# batch_size   = 64

# %% [raw]
# #set a specific seed to make results reproducible between normalized and unnormalized inputs
# seed = 10
#
# random.seed(seed)
# np.random.seed(seed)
#
# # PyTorch
# torch.manual_seed(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# %% [raw]
# full_dataset = TensorDataset(x, y)
#
# # -- split into train / val --
# n_val   = int(len(full_dataset) * val_fraction)
# n_train = len(full_dataset) - n_val
# train_ds, val_ds = random_split(full_dataset, [n_train, n_val])
#
# train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
# val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

# %%
importlib.reload(train)
importlib.reload(models)


# %% [raw]
# # model, optimizer, loss function
# model     = models.SimpleMLP(input_dim, hidden_dims, output_dim, dropout_rate).to(device)
# optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
# criterion = MSELoss()

# %% [raw]
# train_losses, val_losses = train.fit(model, train_loader, val_loader, optimizer, criterion, device, epochs)

# %%
def objective(trial):
    # --- Hyperparameter ranges ---
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    wd = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [16, 32])
    n_layers = trial.suggest_int("n_layers", 1,3)
    hidden_dims = [trial.suggest_int(f"n_units_l{i+1}", 32, 256) for i in range(n_layers)]
    
    # Reproducibility
    seed = 10
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Dataset (x and y must be defined in global scope)
    full_dataset = TensorDataset(x, y)
    n_val = int(len(full_dataset) * 0.1)
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Safe input/output dim inference
    input_dim = x.shape[1] if x.ndim > 1 else 1
    output_dim = y.shape[1] if y.ndim > 1 else 1

    # Model
    model = models.SimpleMLP(input_dim, hidden_dims, output_dim, dropout_rate).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100)

    try:
        train_losses, val_losses = train.fit(
            model, train_loader, val_loader,
            optimizer, criterion, device,
            epochs=1000, scheduler=scheduler
        )
    except Exception as e:
        print(f"Trial failed with error: {e}")
        return float("inf")

    return val_losses[-1]



# %% jupyter={"outputs_hidden": true}
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=1000)

# %%
print("Best trial:")
trial = study.best_trial

print("  Validation loss:", trial.value)
print("  Hyperparameters:")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# %%
#optuna.visualization.plot_optimization_history(study)
optuna.visualization.plot_param_importances(study)


# %%

# %%
optuna.visualization.plot_parallel_coordinate(study)

# %%
plt.figure(figsize=(10, 6))
epochs_range = range(1, epochs + 1)
plt.plot(epochs_range, train_losses, label='Training Loss')
plt.plot(epochs_range, val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../../training_plot_normalized.png")
plt.show()

# %%

# %%
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

all_predictions = all_predictions * stds + means
all_true_values = all_true_values * stds + means
all_predictions[:, logflag] = np.exp(all_predictions[:, logflag])
all_true_values[:, logflag] = np.exp(all_true_values[:, logflag])

# %%
# Create figure with subplots
#Npanels=[5,7]
Npanels=[3,2]
fig, axes = plt.subplots(nrows=Npanels[0], ncols=Npanels[1], figsize=(25,25))
for i in range(Npanels[0]):
    for j in range(Npanels[1]):
        ax = axes[i, j]

        predictions = all_predictions[:,i*Npanels[1]+j]
        true_values = all_true_values[:,i*Npanels[1]+j]
        
        ax.scatter(true_values, predictions, alpha=0.6)
        r2 = r2_score(true_values, predictions)
        rmse = np.sqrt(mean_squared_error(true_values, predictions))

        # Plotting ideal prediction line
        min_val = min(true_values.min(), predictions.min())
        max_val = max(true_values.max(), predictions.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

        ax.set_xlabel('True Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title(f'R² = {r2:.3f}, RMSE = {rmse:.3f}')
        ax.grid(True)
        ax.set_xlim(origParameters.min(axis=0)[i*Npanels[1]+j]-(origParameters.max(axis=0)[i*Npanels[1]+j]-origParameters.min(axis=0)[i*Npanels[1]+j])*0.1, origParameters.max(axis=0)[i*Npanels[1]+j]+(origParameters.max(axis=0)[i*Npanels[1]+j]-origParameters.min(axis=0)[i*Npanels[1]+j])*0.1)
        ax.set_ylim(origParameters.min(axis=0)[i*Npanels[1]+j]-(origParameters.max(axis=0)[i*Npanels[1]+j]-origParameters.min(axis=0)[i*Npanels[1]+j])*0.1, origParameters.max(axis=0)[i*Npanels[1]+j]+(origParameters.max(axis=0)[i*Npanels[1]+j]-origParameters.min(axis=0)[i*Npanels[1]+j])*0.1)

plt.tight_layout()
plt.savefig("../../results_plot_normalized.png")
plt.show()

# %%

# %%
# for the case of just one output

predictions = all_predictions
true_values = all_true_values

r2 = r2_score(true_values, predictions)
rmse = np.sqrt(mean_squared_error(true_values, predictions))

plt.figure(figsize=(7, 7))
plt.scatter(true_values, predictions, alpha=0.6)

# Plotting ideal prediction line
min_val = min(true_values.min(), predictions.min())
max_val = max(true_values.max(), predictions.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title(f'R² = {r2:.3f}, RMSE = {rmse:.3f}')
plt.grid(True)
plt.xlim(origParameters.min()-(origParameters.max()-origParameters.min())*0.1, origParameters.max()+(origParameters.max()-origParameters.min())*0.1)
plt.ylim(origParameters.min()-(origParameters.max()-origParameters.min())*0.1, origParameters.max()+(origParameters.max()-origParameters.min())*0.1)
plt.tight_layout()
plt.show()

# %%

# %%
