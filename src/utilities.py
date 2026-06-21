import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset, random_split

def normalize(
    array_np: np.ndarray
):
    """
    Used to normalize the data, takes a numpy array as input and returns numpy array
    """
    mean = np.mean(array_np, axis = 0)
    std = np.std(array_np, axis = 0)
    array_np = (array_np - mean) / std
    return array_np
def add_noise(array_np, noise_level=0.0):
    noise = np.random.normal(loc=0.0, scale=noise_level, size=array_np.shape)
    array_np += noise
    return array_np

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



def predict_case(model, val_loader_fn, device, *, means=None, stds=None, logflag=None):
    """
    Runs model on val_loader_fn(), returns (preds, trues) as numpy arrays [N, D].
    If means/stds are given (scalar or [D]), de-normalizes. If logflag (bool mask [D]) is given, exp() those cols.
    """
    val_loader = val_loader_fn()

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            preds.append(out.detach().cpu())
            trues.append(yb.detach().cpu())

    preds = torch.cat(preds).numpy()
    trues = torch.cat(trues).numpy()

    # Ensure 2D
    if preds.ndim == 1: preds = preds[:, None]
    if trues.ndim == 1: trues = trues[:, None]

    # De-normalize
    if (means is not None) and (stds is not None):
        means_arr = np.asarray(means).reshape(1, -1) if np.ndim(means) else float(means)
        stds_arr  = np.asarray(stds).reshape(1, -1)  if np.ndim(stds)  else float(stds)
        preds = preds * stds_arr + means_arr
        trues = trues * stds_arr + means_arr

    # Undo log
    if logflag is not None:
        mask = np.asarray(logflag, dtype=bool)
        preds[:, mask] = np.exp(preds[:, mask])
        trues[:, mask] = np.exp(trues[:, mask])

    return preds, trues


def plot_pred_true_grid(preds, trues, *, title="Prediction Results", output_dim=None, figsize_per_panel=(5,5), show=True):
    """
    Plots True vs Pred per output dim. Returns dict with r2, rmse arrays (length D).
    """
    preds = np.asarray(preds); trues = np.asarray(trues)
    if preds.ndim == 1: preds = preds[:, None]
    if trues.ndim == 1: trues = trues[:, None]
    D = output_dim or preds.shape[1]

    ncols = int(np.floor(np.sqrt(D))) or 1
    nrows = int(np.ceil(D / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per_panel[0]*ncols, figsize_per_panel[1]*nrows))
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    r2 = np.zeros(D); rmse = np.zeros(D)
    for i in range(D):
        ax = axes.flat[i]
        ti, pi = trues[:, i], preds[:, i]
        r2[i] = r2_score(ti, pi)
        rmse[i] = np.sqrt(mean_squared_error(ti, pi))

        ax.scatter(ti, pi, alpha=0.6)
        vmin, vmax = min(ti.min(), pi.min()), max(ti.max(), pi.max())
        span = vmax - vmin if vmax > vmin else 1.0
        a, b = vmin - 0.1*span, vmax + 0.1*span
        ax.plot([a, b], [a, b], 'r--', linewidth=2)
        ax.set_xlim(a, b); ax.set_ylim(a, b)
        ax.set_xlabel("True"); ax.set_ylabel("Predicted")
        ax.set_title(f"Out {i} • R²={r2[i]:.3f}, RMSE={rmse[i]:.3f}")
        ax.grid(True)

    for j in range(D, nrows*ncols):
        fig.delaxes(axes.flat[j])

    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    if show:
        plt.show()

    return {"r2": r2, "rmse": rmse}
