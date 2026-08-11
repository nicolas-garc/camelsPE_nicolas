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

def add_noise(array_np, noise_level=0.0):
    """Add Gaussian noise to an already-normalized observable.

    DELIBERATELY does not renormalize afterwards — see CLAUDE.md for rationale.
    Non-mutating: returns array_np + noise (rebinds, does not modify in place).
    """
    noise = np.random.normal(loc=0.0, scale=noise_level, size=array_np.shape)
    return array_np + noise


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

def make_train_loader_fn(selected_observables, x_dict, y_vector, idx_train, batch_size):
    def loader_fn():
        x_list = []
        for key in sorted(selected_observables.keys()):
            noise_level = selected_observables[key]
            arr = x_dict[key][idx_train]
            x_proc = add_noise(arr, noise_level)
            x_list.append(torch.from_numpy(x_proc).float())
        x_epoch = torch.cat(x_list, dim=1)
        y_epoch = y_vector[idx_train]
        return DataLoader(TensorDataset(x_epoch, y_epoch), batch_size=batch_size, shuffle=True)
    return loader_fn


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
    pred_np = pred_np * stds + means
    true_np = true_np * stds + means
    if space == "physical":
        pred_np[:, logflag] = np.exp(pred_np[:, logflag])
        true_np[:, logflag] = np.exp(true_np[:, logflag])
    elif space != "log":
        raise ValueError("space must be 'physical' or 'log'.")

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


# ---------------------------------------------------------------------------
# Moment network — G(x) predicts per-parameter variance of the posterior.
# Follows Jeffrey & Wandelt 2020: given F trained on E[||theta - F(x)||^2],
# a second net G minimizing E[||(theta - F_fixed(x))^2 - G(x)||^2] predicts
# the posterior variance. We add k-fold out-of-fold residuals so G isn't
# biased by F having memorized its own training set.
# ---------------------------------------------------------------------------


def _train_one_mean_net(selected_observables, idx_train_fold, epochs_,
                        models_module, train_module, seed=None,
                        wd=1e-3, restore_best_weights=True,
                        best_weights_smoothing_window=50):
    """Train a fresh SimpleMLP on the given train indices with epoch-noise resampling.

    Defaults tuned for the overfitting-heavy small-data regime:
      wd=1e-3 (was 1e-5), restore_best_weights=True (was False).
    """
    import torch.nn as nn
    import torch.optim as optim
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    input_dim = sum(x_raw_dict[k].shape[1] for k in selected_observables)
    model = models_module.SimpleMLP(input_dim, hidden_dims, output_dim, dropout_rate).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=wd)
    criterion = nn.MSELoss()
    tl_fn = make_train_loader_fn(selected_observables, x_normalized_dict, y,
                                 idx_train_fold, batch_size)
    val_fn = make_val_loader_fn(selected_observables, x_normalized_dict, y,
                                idx_val, batch_size)
    train_module.fit_with_epoch_noise(
        model=model, train_loader=None, train_loader_fn=tl_fn,
        val_loader=val_fn(), optimizer=optimizer, criterion=criterion,
        device=device, epochs=epochs_,
        restore_best_weights=restore_best_weights,
        best_weights_smoothing_window=best_weights_smoothing_window,
    )
    return model


def _predict_clean(model, selected_observables, indices):
    """Forward the model on clean inputs at the given indices. Returns [N, output_dim] in normalized space."""
    loader = make_val_loader_fn(selected_observables, x_normalized_dict, y,
                                indices, batch_size)()
    model = model.to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            preds.append(model(xb.to(device)).cpu())
    return torch.cat(preds).numpy()



# ---------------------------------------------------------------------------
# Moment head M(x) — ONE network predicting variance + covariance jointly.
# Follows Jeffrey & Wandelt 2020 reference implementation
# (github.com/NiallJeffrey/MomentNetworks): a single network with k(k+1)/2
# outputs — the upper triangle of the k×k moment matrix for the k focus params:
#     targets(x, θ) = flatten_upper_tri[ (θ_i - F_i)(θ_j - F_j) ]  for i <= j
# where diagonal entries (i == j) are the variances, off-diagonal are covariances.
# Trained on OOF residuals from a K-fold pass through F, MSE loss.
# ---------------------------------------------------------------------------


def _resolve_focus_indices(focus_params, all_param_names=None):
    """Resolve mixed inputs (ints, 'θk', or explicit names) to (indices, labels)."""
    if all_param_names is None:
        all_param_names = [f"θ{j}" for j in range(output_dim)]
    idxs = []
    for p in focus_params:
        if isinstance(p, (int, np.integer)):
            idxs.append(int(p)); continue
        if isinstance(p, str):
            if p in all_param_names:
                idxs.append(all_param_names.index(p)); continue
            if p.startswith(("θ", "theta")):
                stripped = p[1:] if p.startswith("θ") else p[len("theta"):]
                if stripped.isdigit():
                    idxs.append(int(stripped)); continue
            if p.isdigit():
                idxs.append(int(p)); continue
        raise ValueError(f"can't resolve focus param {p!r}")
    idxs = np.asarray(idxs, dtype=int)
    labels = [all_param_names[i] for i in idxs]
    return idxs, labels


def _upper_tri_indices(k):
    """Row/col indices for the k(k+1)/2 upper-triangle entries (i <= j)."""
    rows, cols = np.triu_indices(k, k=0)
    return rows, cols


def _flat_to_symmetric(flat, k):
    """Turn a [N, k(k+1)/2] flat array into a [N, k, k] symmetric matrix.
    Row/col order follows np.triu_indices(k, 0).
    """
    N = flat.shape[0]
    rows, cols = _upper_tri_indices(k)
    M = np.zeros((N, k, k), dtype=flat.dtype)
    M[:, rows, cols] = flat
    M[:, cols, rows] = flat   # symmetrize
    return M


def _enforce_psd(cov_batch, eps=1e-8, quiet=True):
    """Project each [k,k] slice of [N,k,k] to nearest PSD. Returns (projected, n_fixed)."""
    out = np.array(cov_batch, dtype=float)
    fixed = 0
    for n in range(out.shape[0]):
        C = 0.5 * (out[n] + out[n].T)
        w, V = np.linalg.eigh(C)
        if w.min() < eps:
            w_clip = np.maximum(w, eps)
            out[n] = V @ np.diag(w_clip) @ V.T
            fixed += 1
        else:
            out[n] = C
    if fixed and not quiet:
        print(f"  [psd] {fixed}/{out.shape[0]} matrices had negative eigenvalue → clipped to {eps}")
    return out, fixed


def cov_to_corr(cov):
    """Convert a [..., k, k] covariance array to correlation. Guards against tiny/negative diagonals."""
    d = np.sqrt(np.clip(np.diagonal(cov, axis1=-2, axis2=-1), 1e-30, None))
    outer = d[..., :, None] * d[..., None, :]
    return cov / outer


def fit_moment_head(result, focus_params, *, models_module, train_module,
                     K=5, fold_epochs=None, moment_epochs=None,
                     moment_hidden_dims=None, moment_dropout=0.4,
                     wd=1e-3, moment_wd=None,
                     restore_best_weights=True,
                     best_weights_smoothing_window=50,
                     all_param_names=None, seed=0):
    """Train ONE network M(x) that predicts variance + covariance for `focus_params`.

    Architecture: SimpleMLP with output_dim = k(k+1)/2 where k = len(focus_params).
    Output layout: upper triangle of the k×k moment matrix, in the row-major order
    given by np.triu_indices(k, 0) — diagonal first row of the triangle, then row 0
    entries, etc. See np.triu_indices docs.

    Training targets (per sim n, per output m):
        target[n, m] = (θ_n[i_m] − F(x_n)[i_m]) * (θ_n[j_m] − F(x_n)[j_m])
    where (i_m, j_m) are the m-th (i, j) with i <= j from _upper_tri_indices(k).
    Diagonal entries (i == j) are squared residuals; off-diagonal are cross-products.

    Uses K-fold OOF residuals (train subset F's) so targets reflect unseen-data
    residual magnitudes, not train-set memorization. Then M is trained on
    (x_train, z-scored targets) with MSE. Val loader uses F*'s residuals on
    idx_val for model selection (restore_best_weights).

    Attaches to `result`:
      moment_model               — trained M (SimpleMLP), k(k+1)/2 outputs
      moment_focus_indices       — np.array of the k parameter indices (into output_dim)
      moment_focus_params        — list of the k focus label strings
      moment_target_mean, _std   — z-score parameters to invert M's output
      oof_residuals              — [n_train, output_dim] residuals in normalized space
      moment_train_losses, moment_val_losses
    """
    from sklearn.model_selection import KFold
    import torch.nn as nn
    import torch.optim as optim

    if fold_epochs is None:
        fold_epochs = epochs
    if moment_epochs is None:
        moment_epochs = epochs
    if moment_hidden_dims is None:
        moment_hidden_dims = hidden_dims
    if moment_wd is None:
        moment_wd = wd

    focus_indices, focus_labels = _resolve_focus_indices(focus_params, all_param_names)
    k = len(focus_indices)
    rows, cols = _upper_tri_indices(k)
    n_out = len(rows)  # k(k+1)/2
    print(f"  [{result['case_name']}] moment head: k={k} focus params → {n_out} outputs "
          f"(variance + covariance combined)")

    selected_observables = result["selected_observables"]
    idx_train_np = np.asarray(idx_train)

    # -------------- K-fold OOF residuals for the FULL output_dim --------------
    kf = KFold(n_splits=K, shuffle=True, random_state=seed)
    oof_pred = np.full((len(idx_train_np), output_dim), np.nan)
    for k_fold, (fit_idx, hold_idx) in enumerate(kf.split(idx_train_np)):
        print(f"  [{result['case_name']}] fold {k_fold + 1}/{K} — training partial F")
        fold_train_indices = idx_train_np[fit_idx]
        model_k = _train_one_mean_net(selected_observables, fold_train_indices,
                                       fold_epochs, models_module, train_module,
                                       seed=seed + k_fold, wd=wd,
                                       restore_best_weights=restore_best_weights,
                                       best_weights_smoothing_window=best_weights_smoothing_window)
        holdout_indices = idx_train_np[hold_idx]
        oof_pred[hold_idx] = _predict_clean(model_k, selected_observables, holdout_indices)
        del model_k
        if device.type == "cuda":
            torch.cuda.empty_cache()

    y_train_norm = y[idx_train_np].numpy()
    residuals_full = y_train_norm - oof_pred                              # [n_train, output_dim]
    focus_res = residuals_full[:, focus_indices]                          # [n_train, k]

    # -------------- Assemble targets: flat upper triangle of r_i * r_j ---------
    targets = focus_res[:, rows] * focus_res[:, cols]                      # [n_train, n_out]
    t_mean = targets.mean(axis=0)
    t_std = targets.std(axis=0)
    t_std = np.where(t_std > 0, t_std, 1.0)
    t_z = (targets - t_mean) / t_std

    # -------------- Val targets from F*'s residuals on idx_val ----------------
    F_star = result["model"]
    val_preds = _predict_clean(F_star, selected_observables, np.asarray(idx_val))
    val_res_full = y[np.asarray(idx_val)].numpy() - val_preds
    val_focus_res = val_res_full[:, focus_indices]
    val_targets = val_focus_res[:, rows] * val_focus_res[:, cols]
    val_t_z = (val_targets - t_mean) / t_std

    # -------------- Train M --------------
    input_dim = sum(x_raw_dict[k_].shape[1] for k_ in selected_observables)
    torch.manual_seed(seed); np.random.seed(seed)
    M = models_module.SimpleMLP(input_dim, moment_hidden_dims, n_out, moment_dropout).to(device)
    M_opt = optim.Adam(M.parameters(), lr=1e-4, weight_decay=moment_wd)
    M_loss = nn.MSELoss()

    keys_sorted = sorted(selected_observables.keys())
    x_train_list = [torch.from_numpy(x_normalized_dict[k_][idx_train_np]).float()
                    for k_ in keys_sorted]
    x_val_list = [torch.from_numpy(x_normalized_dict[k_][np.asarray(idx_val)]).float()
                  for k_ in keys_sorted]
    x_train_tensor = torch.cat(x_train_list, dim=1)
    x_val_tensor = torch.cat(x_val_list, dim=1)
    t_train_tensor = torch.from_numpy(t_z).float()
    t_val_tensor = torch.from_numpy(val_t_z).float()

    M_train_loader = DataLoader(TensorDataset(x_train_tensor, t_train_tensor),
                                 batch_size=batch_size, shuffle=True)
    M_val_loader = DataLoader(TensorDataset(x_val_tensor, t_val_tensor),
                               batch_size=batch_size, shuffle=False)

    print(f"  [{result['case_name']}] training M on OOF moment targets")
    tl, vl = train_module.fit(
        model=M, train_loader=M_train_loader, val_loader=M_val_loader,
        optimizer=M_opt, criterion=M_loss, device=device, epochs=moment_epochs,
        restore_best_weights=restore_best_weights,
        best_weights_smoothing_window=best_weights_smoothing_window,
    )

    result["moment_model"] = M
    result["moment_focus_indices"] = focus_indices
    result["moment_focus_params"] = list(focus_labels)
    result["moment_target_mean"] = t_mean
    result["moment_target_std"] = t_std
    result["oof_residuals"] = residuals_full
    result["moment_train_losses"] = tl
    result["moment_val_losses"] = vl
    return result


def _predict_moment_matrix(result, indices):
    """Internal: run M on the given indices and return [N, k, k] cov in NORMALIZED space."""
    M = result["moment_model"].to(device)
    keys_sorted = sorted(result["selected_observables"].keys())
    x_list = [torch.from_numpy(x_normalized_dict[k_][indices]).float() for k_ in keys_sorted]
    x_data = torch.cat(x_list, dim=1)
    M.eval()
    with torch.no_grad():
        t_z = M(x_data.to(device)).cpu().numpy()                          # [N, k(k+1)/2]
    flat = t_z * result["moment_target_std"] + result["moment_target_mean"]
    k = len(result["moment_focus_indices"])
    return _flat_to_symmetric(flat, k)


def predict_covariance_matrix(result, *, indices=None, space="normalized",
                                enforce_psd=True):
    """Return (mu, cov, truth) all restricted to the k focus_params.

    Shapes: mu:[N,k], cov:[N,k,k], truth:[N,k].

    Both variance (diagonal) AND covariance (off-diagonal) come from a single
    trained M network — no separate G/H.

    space:
      "normalized"  — z-scored log-target space (M's native)
      "log_partial" — undo z-score only; covariance transforms as D · Cov · D^T
                      with D = diag(stds[focus_indices]). mu, truth same transform.
      "physical" is NOT supported — mixing log/linear units per param makes the
      covariance ill-defined. Use "log_partial" for interpretable Gaussians.
    """
    if "moment_model" not in result:
        raise ValueError(f"result {result['case_name']!r} has no moment_model — "
                         "call fit_moment_head first.")
    if indices is None:
        indices = np.asarray(_eval_indices())
    indices = np.asarray(indices)

    focus_indices = result["moment_focus_indices"]
    k = len(focus_indices)

    # μ from F* (full [N, output_dim]), then restrict
    mu_all_norm = _predict_clean(result["model"], result["selected_observables"], indices)
    mu_norm = mu_all_norm[:, focus_indices]
    truth_all_norm = y[indices].numpy()
    truth_norm = truth_all_norm[:, focus_indices]

    cov_norm = _predict_moment_matrix(result, indices)

    if space == "normalized":
        cov_out = cov_norm
        mu_out = mu_norm
        truth_out = truth_norm
    elif space == "log_partial":
        D = stds[focus_indices]
        cov_out = cov_norm * (D[None, :, None] * D[None, None, :])
        mu_out = mu_norm * D + means[focus_indices]
        truth_out = truth_norm * D + means[focus_indices]
    else:
        raise ValueError(f"space must be 'normalized' or 'log_partial'; got {space!r}")

    if enforce_psd:
        cov_out, _ = _enforce_psd(cov_out)
    return mu_out, cov_out, truth_out


def predict_with_uncertainty(result, *, indices=None, space="normalized"):
    """Backward-compat wrapper. Returns (mu, sigma, truth) shapes [N, output_dim].
    Sigma is filled from the diagonal of the moment matrix for focus params; NaN
    for non-focus params (sigma undefined). mu and truth are full-dim from F*/y.

    space is applied to sigma only in "log_partial" (multiply by stds).
    For per-param Gaussian marginal plots, focus-only entries are what matters.
    """
    if "moment_model" not in result:
        raise ValueError(f"case {result['case_name']} has no moment_model; call fit_moment_head first.")
    if indices is None:
        indices = np.asarray(_eval_indices())
    indices = np.asarray(indices)

    mu_focus, cov_focus, truth_focus = predict_covariance_matrix(
        result, indices=indices, space=space, enforce_psd=True)
    focus_indices = result["moment_focus_indices"]

    N = len(indices)
    mu_all = _predict_clean(result["model"], result["selected_observables"], indices)
    truth_all = y[indices].numpy()

    if space == "log_partial":
        mu_all = mu_all * stds + means
        truth_all = truth_all * stds + means
    elif space != "normalized":
        raise ValueError(f"space must be 'normalized' or 'log_partial'; got {space!r}")

    sigma_all = np.full_like(mu_all, np.nan)
    diag = np.diagonal(cov_focus, axis1=1, axis2=2)
    sigma_all[:, focus_indices] = np.sqrt(np.clip(diag, 0.0, None))
    return mu_all, sigma_all, truth_all


# ---------------------------------------------------------------------------
# Heteroscedastic-Gaussian alternative to the moment network.
#
# One network jointly predicts (μ, log σ²) per parameter. Trained with Gaussian
# negative-log-likelihood (Kendall & Gal 2017). MSE warmup phase (first N epochs)
# stabilizes μ before the variance head sees any gradient — otherwise σ can
# collapse toward large values and starve μ of gradient at initialization.
#
# Predicts per-sim variance directly (heteroscedastic) — no K-fold, no separate
# variance network. Diagonal only (no covariance); extend to Cholesky
# parametrization later if needed.
# ---------------------------------------------------------------------------

import math as _math


def gaussian_nll_diag(mu_pred, log_var_pred, target):
    """Gaussian NLL with diagonal covariance, per-parameter independent.

    Assumes p(target | x) = ∏_i N(target_i; μ_pred_i(x), exp(log_var_pred_i(x))).
    Returns a scalar averaged over batch × params.

        NLL = 0.5 * ( (y - μ)² / σ² + log σ² + log(2π) )
    """
    diff2 = (target - mu_pred) ** 2
    # exp(-log_var) is numerically stable up to |log_var| ≈ 700
    return 0.5 * (diff2 * torch.exp(-log_var_pred) + log_var_pred).mean()


def _train_one_hetero_net(selected_observables, idx_train_fold, epochs_,
                           models_module, train_module, seed=None,
                           mse_warmup_epochs=200,
                           hetero_hidden_dims=None, hetero_dropout=None,
                           hetero_wd=None,
                           restore_best_weights=True,
                           best_weights_smoothing_window=50):
    """Train a HeteroscedasticMLP with epoch-noise resampling.

    Two-phase training:
      epochs [0, mse_warmup_epochs) — pure MSE loss on μ (σ head predicts but
        receives no gradient — its outputs won't matter yet).
      epochs [mse_warmup_epochs, epochs_) — pure Gaussian NLL loss.

    restore_best_weights operates on the val NLL loss curve (via the same
    fit_with_epoch_noise machinery, but with a custom NLL criterion).
    """
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    if seed is not None:
        torch.manual_seed(seed); np.random.seed(seed)

    _h = hetero_hidden_dims if hetero_hidden_dims is not None else hidden_dims
    _d = hetero_dropout if hetero_dropout is not None else dropout_rate
    _wd = hetero_wd if hetero_wd is not None else 1e-3

    input_dim = sum(x_raw_dict[k].shape[1] for k in selected_observables)
    model = models_module.HeteroscedasticMLP(input_dim, list(_h),
                                              output_dim, _d).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=_wd)

    def _hetero_criterion(pred, target):
        # pred is a tuple (mu, log_var) — pack it into a single tensor here so
        # fit_with_epoch_noise's shape checks pass
        raise NotImplementedError

    # Roll our own training loop (fit_with_epoch_noise assumes scalar loss)
    tl_fn = make_train_loader_fn(selected_observables, x_normalized_dict, y,
                                  idx_train_fold, batch_size)
    val_fn = make_val_loader_fn(selected_observables, x_normalized_dict, y,
                                 idx_val, batch_size)
    from tqdm.auto import tqdm

    train_losses, val_losses = [], []
    alpha = 2.0 / (best_weights_smoothing_window + 1)
    smoothed = None
    best_smoothed = float("inf")
    best_state = None
    best_epoch = -1
    pbar = tqdm(total=epochs_, desc="Hetero", unit="iter")

    for epoch in range(1, epochs_ + 1):
        model.train()
        loader = tl_fn()
        total = 0.0
        n_batches = 0
        for xb, yb in loader:
            if xb.ndim > 2:
                xb = xb.view(xb.size(0), -1)
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            mu, log_var = model(xb)
            if epoch <= mse_warmup_epochs:
                loss = ((yb - mu) ** 2).mean()          # MSE, log_var untouched
            else:
                loss = gaussian_nll_diag(mu, log_var, yb)
            loss.backward()
            optimizer.step()
            total += loss.item(); n_batches += 1
        train_losses.append(total / max(n_batches, 1))

        # Val loss — always NLL (that's what we care about at deployment)
        model.eval()
        vtotal = 0.0; vn = 0
        with torch.no_grad():
            for xb, yb in val_fn():
                if xb.ndim > 2:
                    xb = xb.view(xb.size(0), -1)
                xb, yb = xb.to(device), yb.to(device)
                mu, log_var = model(xb)
                vloss = gaussian_nll_diag(mu, log_var, yb).item()
                vtotal += vloss; vn += 1
        vloss_epoch = vtotal / max(vn, 1)
        val_losses.append(vloss_epoch)

        if restore_best_weights and epoch > mse_warmup_epochs:
            smoothed = vloss_epoch if smoothed is None else \
                       alpha * vloss_epoch + (1 - alpha) * smoothed
            if smoothed < best_smoothed:
                best_smoothed = smoothed
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch

        pbar.update(1)
        if epoch % 200 == 0:
            phase = "MSE" if epoch <= mse_warmup_epochs else "NLL"
            pbar.write(f"[Iter {epoch:4d}][{phase}] train={train_losses[-1]:.4f} val_nll={vloss_epoch:.4f}")

    pbar.close()

    if restore_best_weights and best_state is not None:
        model.load_state_dict(best_state)
        print(f"[hetero best-weights] restored epoch {best_epoch}  "
              f"val_nll ema={best_smoothed:.4f}")

    return model, train_losses, val_losses


def fit_hetero_case(result, focus_params, *, models_module, train_module,
                     epochs=None, mse_warmup_epochs=200,
                     hetero_hidden_dims=None, hetero_dropout=None,
                     hetero_wd=None,
                     restore_best_weights=True,
                     best_weights_smoothing_window=50,
                     all_param_names=None, seed=0):
    """Train a single HeteroscedasticMLP per case; attach to result.

    Attaches:
      hetero_model             — trained HeteroscedasticMLP (2 heads: μ, log σ²)
      hetero_focus_indices     — indices of focus_params in output_dim
      hetero_focus_params      — label strings
      hetero_train_losses, hetero_val_losses
    """
    if epochs is None:
        epochs = globals().get("epochs", 1000) or 1000
    focus_indices, focus_labels = _resolve_focus_indices(focus_params, all_param_names)

    print(f"  [{result['case_name']}] fitting hetero net: {epochs} epochs "
          f"({mse_warmup_epochs} MSE warmup + {epochs - mse_warmup_epochs} NLL)")
    model, tl, vl = _train_one_hetero_net(
        result["selected_observables"], np.asarray(idx_train), epochs,
        models_module, train_module, seed=seed,
        mse_warmup_epochs=mse_warmup_epochs,
        hetero_hidden_dims=hetero_hidden_dims,
        hetero_dropout=hetero_dropout,
        hetero_wd=hetero_wd,
        restore_best_weights=restore_best_weights,
        best_weights_smoothing_window=best_weights_smoothing_window,
    )
    result["hetero_model"] = model
    result["hetero_focus_indices"] = focus_indices
    result["hetero_focus_params"] = list(focus_labels)
    result["hetero_train_losses"] = tl
    result["hetero_val_losses"] = vl
    return result


def predict_hetero(result, *, indices=None, space="normalized"):
    """Return (mu, sigma, truth) — shape [N, output_dim] (all params, not just focus).

    Uses the hetero net's own μ and √exp(log σ²). No PSD enforcement needed
    (diagonal is always non-negative by construction of exp).
    """
    if "hetero_model" not in result:
        raise ValueError(f"case {result['case_name']} has no hetero_model; call fit_hetero_case first.")
    if indices is None:
        indices = np.asarray(_eval_indices())
    indices = np.asarray(indices)

    keys_sorted = sorted(result["selected_observables"].keys())
    x_list = [torch.from_numpy(x_normalized_dict[k_][indices]).float() for k_ in keys_sorted]
    x_data = torch.cat(x_list, dim=1)
    model = result["hetero_model"].to(device)
    model.eval()
    with torch.no_grad():
        mu, log_var = model(x_data.to(device))
        mu_norm = mu.cpu().numpy()
        sigma_norm = np.sqrt(np.exp(log_var.cpu().numpy()))

    truth_norm = y[indices].numpy()

    if space == "normalized":
        return mu_norm, sigma_norm, truth_norm
    if space == "log_partial":
        mu_pl = mu_norm * stds + means
        sigma_pl = sigma_norm * stds
        truth_pl = truth_norm * stds + means
        return mu_pl, sigma_pl, truth_pl
    raise ValueError(f"space must be 'normalized' or 'log_partial'; got {space!r}")


def predict_hetero_covariance(result, *, indices=None, space="normalized"):
    """Return (mu, cov, truth) restricted to focus_params, cov = diag(σ²).

    Compatibility wrapper matching the moment-network's predict_covariance_matrix
    API so existing plot functions (plot_moment_corner, plot_pull_calibration)
    work unchanged on hetero results.
    """
    mu_all, sigma_all, truth_all = predict_hetero(result, indices=indices, space=space)
    focus_indices = result["hetero_focus_indices"]
    k = len(focus_indices)
    mu = mu_all[:, focus_indices]
    sigma = sigma_all[:, focus_indices]
    truth = truth_all[:, focus_indices]
    # Build [N, k, k] diagonal covariance
    N = mu.shape[0]
    cov = np.zeros((N, k, k))
    for i in range(k):
        cov[:, i, i] = sigma[:, i] ** 2
    return mu, cov, truth
