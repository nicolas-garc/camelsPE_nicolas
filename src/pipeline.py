"""
Pipeline utility functions for the CAMELS SBI noise-mixing experiment.

Call configure() once after loading data and setting up the train/val split
to register shared state (observable names, normalized data, device, etc.).
Functions that depend on this state use module-level variables set by configure().
"""
import sys as _sys
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score

# ---------------------------------------------------------------------------
# Module-level state — set via configure() before calling stateful functions.
# ---------------------------------------------------------------------------
observable_1 = None
observable_2 = None
x_normalized_dict = None
y = None
idx_val = None
idx_test = None
idx_train = None
x_raw_dict = None
batch_size = None
device = None
logflag = None
means = None
stds = None
output_dim = None
all_results = None
perm = None
hidden_dims = None
dropout_rate = None
epochs = None


def configure(**kwargs):
    """Set shared state for pipeline functions. Call once after data loading."""
    mod = _sys.modules[__name__]
    for k, v in kwargs.items():
        setattr(mod, k, v)


def _eval_indices():
    """Return the honest evaluation indices — prefers idx_test if configured, else
    falls back to idx_val (backward compat before the 3-way split was introduced)."""
    return idx_test if idx_test is not None else idx_val


# ---------------------------------------------------------------------------
# Pure utility functions (no module state needed).
# ---------------------------------------------------------------------------

def normalize(array_np):
    mean = np.mean(array_np, axis=0)
    std = np.std(array_np, axis=0)
    return (array_np - mean) / std


def shuffle_observable(obs_dict, keys_to_shift, perm):
    shifted_dict = obs_dict.copy()
    for keys in keys_to_shift:
        shifted_dict[keys] = obs_dict[keys][perm]
    return shifted_dict


# ---------------------------------------------------------------------------
# DataLoader factories.
# ---------------------------------------------------------------------------

def make_val_loader_fn(selected_observables, x_dict, y_vector, idx, batch_size,
                       key_to_shuffle=None, perm=None, shuffle_y=False):
    if key_to_shuffle is None:
        shuffle_keys = set()
    elif isinstance(key_to_shuffle, str):
        shuffle_keys = {key_to_shuffle}
    else:
        shuffle_keys = set(key_to_shuffle)

    unknown = shuffle_keys - set(selected_observables)
    if unknown:
        raise KeyError(
            f"key_to_shuffle {sorted(unknown)} is not among this case's observables "
            f"{sorted(selected_observables)}; the shuffle would silently do nothing."
        )
    idx = np.asarray(idx)

    def loader_fn():
        x_list = []
        for key in sorted(selected_observables.keys()):
            arr = x_dict[key][idx]
            if perm is not None and key in shuffle_keys:
                arr = arr[perm]
            x_list.append(torch.from_numpy(arr).float())
        x_data = torch.cat(x_list, dim=1)
        y_slice = y_vector[idx]
        if isinstance(y_slice, np.ndarray):
            y_data = torch.from_numpy(y_slice).float()
        else:
            y_data = y_slice
        if shuffle_y and perm is not None:
            y_data = y_data[perm]
        return DataLoader(TensorDataset(x_data, y_data), batch_size=batch_size, shuffle=False)
    return loader_fn


def make_pair_val_loader_fn(*, selected_observables, x_dict, y_vector, idx,
                            batch_size, obs1_key, obs2_key, pairs):
    """Build DataLoader for unordered sim pairs (i, j) with directional inputs."""
    idx = np.asarray(idx)
    pairs = np.asarray(pairs, dtype=int)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pairs must be shape [n_pairs, 2].")
    if np.any(pairs < 0) or np.any(pairs >= len(idx)):
        raise ValueError("pairs contain indices outside [0, n_val).")
    pairs = np.sort(pairs, axis=1)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    if pairs.shape[0] == 0:
        raise ValueError("No valid pairs after removing i==j.")

    i_rows, j_rows = pairs[:, 0], pairs[:, 1]
    i_idx, j_idx = idx[i_rows], idx[j_rows]
    keys_sorted = sorted(selected_observables.keys())

    x_list = []
    for key in keys_sorted:
        if key == obs1_key:
            arr = x_dict[key][i_idx]
        elif key == obs2_key:
            arr = x_dict[key][j_idx]
        else:
            arr = x_dict[key][i_idx]
        x_list.append(torch.from_numpy(arr).float())

    x_data = torch.cat(x_list, dim=1)
    y_i = y_vector[i_idx]
    y_j = y_vector[j_idx]
    if isinstance(y_i, np.ndarray):
        y_i = torch.from_numpy(y_i).float()
    if isinstance(y_j, np.ndarray):
        y_j = torch.from_numpy(y_j).float()
    y_cat = torch.cat([y_i, y_j], dim=1)
    return DataLoader(TensorDataset(x_data, y_cat), batch_size=batch_size, shuffle=False)


def sample_unique_unordered_pairs(n_val, n_pairs, *, seed=0):
    """Returns pairs array shape [n_pairs, 2] with i<j, unique, sampled randomly."""
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


# ---------------------------------------------------------------------------
# Stateful functions (use module-level variables set via configure).
# ---------------------------------------------------------------------------

def resolve_shuffle(selected_observables, mode):
    """Which observables to shuffle for each test.

    Returns (keys_to_shuffle, shuffle_y). shuffle_y is always False.
    Keys are intersected with selected_observables so single-observable
    reference cases fall out naturally as no-ops.
    """
    if mode == "aligned":
        return set(), False
    sel = set(selected_observables)
    if mode == "obs1_vs_truth":
        return {observable_2} & sel, False
    if mode == "obs2_vs_truth":
        return {observable_1} & sel, False
    raise ValueError(f"Unknown mode: {mode!r}")


def get_case_predictions(result, mode="aligned", perm=None, keys_to_shuffle=None, space="physical"):
    """Validation-set predictions + truths for one trained case, cached on result.

    This is the single place that runs a case's model forward on the validation
    set. Every plot that wants predictions should call this instead of
    re-running the model.

    space: "processed" (raw model I/O, mean 0 / std 1 -- what the loss trains
    on), "log" (processed un-standardized, logflag columns still logged), or
    "physical" (log undone too, via exp() on the logflag columns).
    """
    eval_idx = _eval_indices()
    if perm is None:
        # parameter shadows the module-level perm set by configure(); look it up explicitly
        perm = _sys.modules[__name__].perm
        if perm is None or len(perm) != len(eval_idx):
            perm = np.random.permutation(len(eval_idx))
    perm = np.asarray(perm)

    if keys_to_shuffle is not None:
        keys = set([keys_to_shuffle] if isinstance(keys_to_shuffle, str) else keys_to_shuffle)
        shuffle_y = (mode == "obs2_vs_truth")
    else:
        keys, shuffle_y = resolve_shuffle(result["selected_observables"], mode)

    perm_component = tuple(map(int, perm)) if (keys or shuffle_y) else None
    cache_key = (mode, tuple(sorted(keys)), perm_component, shuffle_y, space)

    cache = result.setdefault("_pred_cache", {})
    if cache_key in cache:
        return cache[cache_key]

    model = result["model"].to(device)
    loader_fn = make_val_loader_fn(
        selected_observables=result["selected_observables"],
        x_dict=x_normalized_dict,
        y_vector=y,
        idx=eval_idx,
        batch_size=batch_size,
        key_to_shuffle=list(keys) if keys else None,
        perm=perm if (keys or shuffle_y) else None,
        shuffle_y=bool(shuffle_y),
    )
    loader = loader_fn()

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds.append(model(xb).cpu())
            trues.append(yb.cpu())

    pred_np = torch.cat(preds).numpy()
    true_np = torch.cat(trues).numpy()
    if space != "processed":
        pred_np = pred_np * stds + means
        true_np = true_np * stds + means
        if space == "physical":
            pred_np[:, logflag] = np.exp(pred_np[:, logflag])
            true_np[:, logflag] = np.exp(true_np[:, logflag])
        elif space != "log":
            raise ValueError("space must be 'processed', 'log', or 'physical'.")

    cache[cache_key] = (pred_np, true_np)
    return cache[cache_key]


def average_r2_over_perms(mode, perms, results=None):
    """Mean and std R2 per case per parameter, averaged over several permutations."""
    if results is None:
        results = all_results
    r2_draws = np.full((len(perms), len(results), output_dim), np.nan)
    for p_idx, p in enumerate(perms):
        for r_idx, result in enumerate(results):
            preds, trues = get_case_predictions(result, mode=mode, perm=p)
            r2_draws[p_idx, r_idx, :] = r2_score(trues, preds, multioutput="raw_values")
    return np.nanmean(r2_draws, axis=0), np.nanstd(r2_draws, axis=0)
