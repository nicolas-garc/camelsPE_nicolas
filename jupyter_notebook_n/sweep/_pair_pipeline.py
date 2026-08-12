"""Shared pipeline for the sweep notebooks.

Each pair-specific notebook (pair_0N_*/run.py) just calls run_pair(obs_a, obs_b, ...).
This module contains the actual training + R2 + plotting logic — extracted from
test-noise-Copy1.py — so the per-pair notebooks stay tiny and easy to inspect.

Runs headless (matplotlib Agg); saves all plots + trained models to out_dir.
"""
import os
import sys
import time
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import seaborn as sns

_SRC = "/Users/nicolasgarcia/Downloads/GAL_SBI/camelsPE/src"
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
import models
import train
import pipeline
import plots

DATAFILE = "/Users/nicolasgarcia/Downloads/GAL_SBI/DATA/data_L50_TNG_v3.hdf5"
LOGFLAG_MASK = np.array([False, False, True, True, True, True, False, False, False, True, True,
                         False, False, True, False, True, False, True, True, False, False, True,
                         True, True, True, True, True, False, True, False, True, False, False,
                         False, True])


def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _HeteroMuOnly(nn.Module):
    """Adapter that exposes only the μ head of a HeteroscedasticMLP, so it can be
    used anywhere a mean-only nn.Module is expected (e.g. pipeline.get_case_predictions).
    Defined at module scope so torch.save can pickle it."""
    def __init__(self, hetero_model):
        super().__init__()
        self.hetero = hetero_model
    def forward(self, x):
        mu, _log_var = self.hetero(x)
        return mu


def _load_data():
    """Return (y_tensor, x_normalized_dict_all_14, logflag, means, stds, observable_block)."""
    with h5py.File(DATAFILE, "r") as f:
        Parameters = f["Parameters"][:, :1024].T
    logflag = LOGFLAG_MASK[:Parameters.shape[1]]
    if not np.all(Parameters[:, logflag] > 0):
        raise ValueError("Non-positive value in logflag columns")
    Pll = Parameters.copy()
    Pll[:, logflag] = np.log(Pll[:, logflag])
    means = Pll.mean(axis=0)
    stds = Pll.std(axis=0)
    Pnorm = (Pll - means) / stds
    n_sims = Pnorm.shape[0]
    with h5py.File(DATAFILE, "r") as f:
        observable_block = {
            key: torch.from_numpy(f[key][:].T).float()
            for key in sorted(f.keys())
            if key != "Parameters" and f[key].shape[-1] == n_sims
        }
    y = torch.from_numpy(Pnorm).float()
    return y, logflag, means, stds, observable_block


def _default_noise_cases(obs_a, obs_b, obs1, obs2):
    """Build 7 noise cases per pair: 5 mixed + 2 single-observable references.

    Case-name convention follows test-noise-Copy1 (obs2 listed first, obs1 second):
    'obs2_val_obs1_val'. Plots' parse_case_name splits on '_', so we use short
    tokens 'A' (= observable_1, alphabetically first) and 'B' (= observable_2).
    """
    del obs_a, obs_b  # obs1/obs2 already resolve via sorted order
    return {
        # 5 mixed cases: symmetric noise sweep across the (A,B) plane through both-clean
        "B_5.0_A_0.0": {obs2: 5.0, obs1: 0.0},
        "B_2.5_A_0.0": {obs2: 2.5, obs1: 0.0},
        "B_0.0_A_0.0": {obs2: 0.0, obs1: 0.0},
        "B_0.0_A_2.5": {obs2: 0.0, obs1: 2.5},
        "B_0.0_A_5.0": {obs2: 0.0, obs1: 5.0},
        # 2 single-observable references
        "B_clean": {obs2: 0.0},
        "A_clean": {obs1: 0.0},
    }


def run_pair(obs_a, obs_b, out_dir, *, epochs=1500, batch_size=64,
             hidden_dims=(128, 64), dropout_rate=0.3, wd=1e-3,
             restore_best_weights=True, best_weights_smoothing_window=50,
             val_fraction=0.1, test_fraction=0.1, seed=0):
    """Run the full test-noise pipeline for one observable pair, all 35 params.

    Trains 7 mean-net cases (5 mixed + 2 single-observable references),
    computes aligned + 2 shuffle-mode R2 matrices covering all 35 parameters,
    saves heatmaps + pickled results to out_dir. No per-parameter plots are
    made here — the R2 matrices/CSVs/results.pt carry all 35 params so plots
    can be picked by hand afterward (see analyze.py / analyze_all.py). Blocks
    until done.

    Overfitting-targeted defaults (changed from previous sweep):
      dropout_rate: 0.2 -> 0.4     (stronger regularization)
      wd:           1e-5 -> 1e-3   (stronger weight decay)
      restore_best_weights: False -> True  (keep the EMA-best-val-loss checkpoint)
      best_weights_smoothing_window: 50    (EMA window for val loss)
    """
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    device = _get_device()
    print(f"[pair] {obs_a} × {obs_b}  device={device}  epochs={epochs}")

    y, logflag, means, stds, observable_block = _load_data()
    if obs_a not in observable_block or obs_b not in observable_block:
        raise KeyError(f"observable pair {(obs_a, obs_b)} not in HDF5; available: {sorted(observable_block)}")

    # Which is obs1 (alphabetically first) and obs2 depends on sort order
    obs1, obs2 = sorted([obs_a, obs_b])
    all_observables = {obs1, obs2}
    print(f"       observable_1 = {obs1}   observable_2 = {obs2}")

    x_raw_dict = {k: observable_block[k].numpy() for k in all_observables}
    x_normalized_dict = {k: pipeline.normalize(observable_block[k].numpy()) for k in all_observables}
    noise_cases = _default_noise_cases(obs_a, obs_b, obs1, obs2)

    output_dim = y.shape[1]
    n_val = int(len(y) * val_fraction)
    n_test = int(len(y) * test_fraction)
    torch.manual_seed(seed); np.random.seed(seed)
    split_perm = torch.randperm(len(y))
    idx_test = split_perm[:n_test]                                     # held out for final reporting
    idx_val = split_perm[n_test:n_test + n_val]                         # model selection (best-weights)
    idx_train = split_perm[n_test + n_val:]                             # gradient updates
    perm = np.random.permutation(len(idx_test))                         # eval-time perm on test

    N_SHUFFLE_PERMS = 10
    _rng = np.random.default_rng(0)
    shuffle_perms = [_rng.permutation(len(idx_test)) for _ in range(N_SHUFFLE_PERMS)]

    pipeline.configure(
        observable_1=obs1, observable_2=obs2,
        x_normalized_dict=x_normalized_dict, x_raw_dict=x_raw_dict,
        y=y, idx_val=idx_val, idx_test=idx_test, idx_train=idx_train,
        batch_size=batch_size, device=device,
        logflag=logflag, means=means, stds=stds, output_dim=output_dim,
        hidden_dims=list(hidden_dims), dropout_rate=dropout_rate, epochs=epochs,
        perm=perm,
    )

    # ---------------- train F* per case ----------------
    all_results = []
    for i, (case_name, selected_observables) in enumerate(noise_cases.items()):
        t_case = time.time()
        print(f"       [{i+1}/{len(noise_cases)}] training {case_name}")
        input_dim = sum(x_raw_dict[k].shape[1] for k in selected_observables)
        model = models.SimpleMLP(input_dim, list(hidden_dims), output_dim, dropout_rate).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=wd)
        criterion = nn.MSELoss()
        tlfn = pipeline.make_train_loader_fn(selected_observables, x_normalized_dict, y, idx_train, batch_size)
        vloader = pipeline.make_val_loader_fn(selected_observables, x_normalized_dict, y, idx_val, batch_size)()
        tl, vl = train.fit_with_epoch_noise(
            model=model, train_loader=None, train_loader_fn=tlfn,
            val_loader=vloader, optimizer=optimizer, criterion=criterion,
            device=device, epochs=epochs,
            restore_best_weights=restore_best_weights,
            best_weights_smoothing_window=best_weights_smoothing_window,
        )
        all_results.append({
            "case_name": case_name, "selected_observables": selected_observables,
            "model": model, "train_losses": tl, "val_losses": vl,
        })
        print(f"          done in {(time.time()-t_case)/60:.1f} min  final val loss={vl[-1]:.4f}  min val loss={min(vl):.4f}")
    pipeline.configure(all_results=all_results)

    # ---------------- R2 matrices ----------------
    r2_matrix = np.zeros((len(all_results), output_dim))
    for ri, r in enumerate(all_results):
        p, t = pipeline.get_case_predictions(r, mode="aligned")
        r2_matrix[ri] = r2_score(t, p, multioutput="raw_values")

    r2_shifted_obs, r2_std_obs = pipeline.average_r2_over_perms("obs1_vs_truth", shuffle_perms)
    r2_shifted_both, r2_std_both = pipeline.average_r2_over_perms("obs2_vs_truth", shuffle_perms)

    param_names = [f"θ{j}" for j in range(output_dim)]
    r2_df = pd.DataFrame(r2_matrix, index=list(noise_cases.keys()), columns=param_names)
    r2_df.to_csv(os.path.join(out_dir, "r2_aligned.csv"))
    pd.DataFrame(r2_shifted_obs, index=list(noise_cases.keys()), columns=param_names).to_csv(
        os.path.join(out_dir, "r2_shifted_obs1_vs_truth.csv"))
    pd.DataFrame(r2_shifted_both, index=list(noise_cases.keys()), columns=param_names).to_csv(
        os.path.join(out_dir, "r2_shifted_obs2_vs_truth.csv"))

    # Combined dual df for plot_param_curve_dual
    rows = []
    for i, r in enumerate(all_results):
        for j, p in enumerate(param_names):
            rows.append({
                "case": r["case_name"], "param": p,
                "r2_aligned": float(r2_matrix[i, j]),
                "r2_shuf_obs1": float(r2_shifted_obs[i, j]),
                "r2_shuf_obs2": float(r2_shifted_both[i, j]),
                "r2_shuf_obs1_std": float(r2_std_obs[i, j]),
                "r2_shuf_obs2_std": float(r2_std_both[i, j]),
            })
    dual_r2_df = pd.DataFrame(rows)
    dual_r2_df.to_csv(os.path.join(out_dir, "dual_r2_df.csv"), index=False)

    plots.configure(
        all_results=all_results, output_dim=output_dim,
        observable_1=obs1, observable_2=obs2,
        logflag=logflag, means=means, stds=stds,
        x_normalized_dict=x_normalized_dict, y=y, idx_val=idx_val,
        param_names=param_names, noise_cases=noise_cases,
        batch_size=batch_size, device=device, perm=perm,
        r2_matrix=r2_matrix,
        r2_matrix_shifted_observable_only=r2_shifted_obs,
        r2_matrix_shifted_both=r2_shifted_both,
    )

    # ---------------- save plots ----------------
    def _save(fig, name):
        p = os.path.join(plots_dir, name + ".png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"       saved {p}")

    # Aligned R^2 heatmap
    fig = plt.figure(figsize=(16, 6))
    sns.heatmap(r2_df, vmin=-1, vmax=1, annot=True, fmt=".2f", cmap="Spectral",
                cbar_kws={"label": "aligned R²"}, linewidths=0.2)
    plt.title(f"aligned R²  |  {obs1} × {obs2}")
    plt.tight_layout()
    _save(fig, "01_r2_aligned")

    # Shifted heatmaps
    for name, arr in [("02_r2_shuffled_obs2", r2_shifted_obs), ("03_r2_shuffled_obs1", r2_shifted_both)]:
        df = pd.DataFrame(arr, index=list(noise_cases.keys()), columns=param_names)
        fig = plt.figure(figsize=(16, 6))
        sns.heatmap(df, vmin=-1, vmax=1, annot=True, fmt=".2f", cmap="Spectral", linewidths=0.2)
        plt.title(f"{name}  |  {obs1} × {obs2}")
        plt.tight_layout()
        _save(fig, name)

    # ΔR² heatmaps
    for name, delta in [("04_delta_r2_shuffled_obs2", r2_shifted_obs - r2_matrix),
                        ("05_delta_r2_shuffled_obs1", r2_shifted_both - r2_matrix)]:
        df = pd.DataFrame(delta, index=list(noise_cases.keys()), columns=param_names)
        fig = plt.figure(figsize=(16, 6))
        sns.heatmap(df, vmin=-0.5, center=0.0, annot=True, fmt=".2f", cmap="Spectral", linewidths=0.3,
                    cbar_kws={"label": "ΔR² (shifted − aligned)"})
        plt.title(f"{name}  |  {obs1} × {obs2}")
        plt.tight_layout()
        _save(fig, name)

    # No per-parameter plots here (no focus params) — dual_r2_df.csv has all
    # 35 params x 7 cases x {aligned, shuf_obs1, shuf_obs2}; pick plots by hand.

    # Save trained models + metadata for later analysis
    torch.save({
        "obs_a": obs_a, "obs_b": obs_b, "obs1": obs1, "obs2": obs2,
        "noise_cases": noise_cases,
        "all_results": all_results,
        "r2_matrix": r2_matrix,
        "r2_matrix_shifted_observable_only": r2_shifted_obs,
        "r2_matrix_shifted_both": r2_shifted_both,
        "r2_std_matrix_observable_only": r2_std_obs,
        "r2_std_matrix_both": r2_std_both,
        # split reproducibility metadata
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
        "seed": seed,
    }, os.path.join(out_dir, "results.pt"))

    dt = (time.time() - t0) / 60
    print(f"[pair] DONE {obs_a} × {obs_b}  in {dt:.1f} min  → {out_dir}")


def run_moment_example(obs_a, obs_b, out_dir, *, epochs=500,
                       fold_epochs=300, var_epochs=300, K=3,
                       batch_size=64, hidden_dims=(128, 64), dropout_rate=0.4,
                       wd=1e-3, restore_best_weights=True,
                       best_weights_smoothing_window=50,
                       moment_dropout=None, moment_wd=None,
                       moment_hidden_dims=None,
                       val_fraction=0.1, test_fraction=0.1, seed=0,
                       focus_params=("θ0", "θ1", "θ2", "θ4", "θ7", "θ11")):
    """Run one MomentNetwork example on the given pair. Lightweight settings."""
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    device = _get_device()
    print(f"[moment] {obs_a} × {obs_b}  device={device}  epochs={epochs} K={K}")

    y, logflag, means, stds, observable_block = _load_data()
    obs1, obs2 = sorted([obs_a, obs_b])
    all_observables = {obs1, obs2}
    x_raw_dict = {k: observable_block[k].numpy() for k in all_observables}
    x_normalized_dict = {k: pipeline.normalize(observable_block[k].numpy()) for k in all_observables}

    # 4-case set: obs1-alone, obs2-alone, both-clean, both-moderate-noise
    noise_cases = pipeline.default_sbi_reference_cases(obs1, obs2)

    output_dim = y.shape[1]
    n_val = int(len(y) * val_fraction)
    n_test = int(len(y) * test_fraction)
    torch.manual_seed(seed); np.random.seed(seed)
    split_perm = torch.randperm(len(y))
    idx_test = split_perm[:n_test]                                     # held out for final reporting
    idx_val = split_perm[n_test:n_test + n_val]                         # model selection during training
    idx_train = split_perm[n_test + n_val:]                             # gradient updates
    perm = np.random.permutation(len(idx_test))

    pipeline.configure(
        observable_1=obs1, observable_2=obs2,
        x_normalized_dict=x_normalized_dict, x_raw_dict=x_raw_dict,
        y=y, idx_val=idx_val, idx_test=idx_test, idx_train=idx_train,
        batch_size=batch_size, device=device,
        logflag=logflag, means=means, stds=stds, output_dim=output_dim,
        hidden_dims=list(hidden_dims), dropout_rate=dropout_rate, epochs=epochs,
        perm=perm,
    )

    all_results = []
    for i, (case_name, selected_observables) in enumerate(noise_cases.items()):
        t_case = time.time()
        print(f"       [{i+1}/{len(noise_cases)}] training F* {case_name}")
        input_dim = sum(x_raw_dict[k].shape[1] for k in selected_observables)
        model = models.SimpleMLP(input_dim, list(hidden_dims), output_dim, dropout_rate).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=wd)
        criterion = nn.MSELoss()
        tlfn = pipeline.make_train_loader_fn(selected_observables, x_normalized_dict, y, idx_train, batch_size)
        vloader = pipeline.make_val_loader_fn(selected_observables, x_normalized_dict, y, idx_val, batch_size)()
        tl, vl = train.fit_with_epoch_noise(
            model=model, train_loader=None, train_loader_fn=tlfn,
            val_loader=vloader, optimizer=optimizer, criterion=criterion,
            device=device, epochs=epochs,
            restore_best_weights=restore_best_weights,
            best_weights_smoothing_window=best_weights_smoothing_window,
        )
        all_results.append({
            "case_name": case_name, "selected_observables": selected_observables,
            "model": model, "train_losses": tl, "val_losses": vl,
        })
        print(f"          done in {(time.time()-t_case)/60:.1f} min")
    pipeline.configure(all_results=all_results)

    r2_matrix = np.zeros((len(all_results), output_dim))
    for ri, r in enumerate(all_results):
        p, t = pipeline.get_case_predictions(r, mode="aligned")
        r2_matrix[ri] = r2_score(t, p, multioutput="raw_values")

    # Moment head defaults: match F unless caller loosened them
    _m_dropout = dropout_rate if moment_dropout is None else moment_dropout
    _m_wd = wd if moment_wd is None else moment_wd
    for r in all_results:
        print(f"       fitting moment head M for {r['case_name']}  "
              f"(dropout={_m_dropout}, wd={_m_wd})")
        pipeline.fit_moment_head(r, focus_params=list(focus_params),
                                  models_module=models, train_module=train,
                                  K=K, fold_epochs=fold_epochs,
                                  moment_epochs=var_epochs, wd=wd,
                                  moment_wd=_m_wd, moment_dropout=_m_dropout,
                                  moment_hidden_dims=moment_hidden_dims,
                                  restore_best_weights=restore_best_weights,
                                  best_weights_smoothing_window=best_weights_smoothing_window)

    param_names = [f"θ{j}" for j in range(output_dim)]
    # No plotting here — analyze_all.py generates the full summary plot suite from
    # results.pt after training. Keeping the training path plot-free avoids stale
    # per-sim PNGs in the results dir.

    torch.save({
        "obs_a": obs_a, "obs_b": obs_b, "obs1": obs1, "obs2": obs2,
        "noise_cases": noise_cases,
        "all_results": all_results,
        "r2_matrix": r2_matrix,
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
        "seed": seed,
    }, os.path.join(out_dir, "results.pt"))

    dt = (time.time() - t0) / 60
    print(f"[moment] DONE {obs_a} × {obs_b}  in {dt:.1f} min  → {out_dir}")


def run_hetero_example(obs_a, obs_b, out_dir, *, epochs=1000,
                        mse_warmup_epochs=200,
                        batch_size=64,
                        hetero_hidden_dims=(128, 64),
                        hetero_dropout=0.4, hetero_wd=1e-3,
                        restore_best_weights=True,
                        best_weights_smoothing_window=50,
                        val_fraction=0.1, test_fraction=0.1, seed=0,
                        focus_params=("θ0", "θ1", "θ2", "θ4", "θ7", "θ8", "θ11", "θ16")):
    """Train a heteroscedastic Gaussian MLP per case (no K-fold, no separate M).

    ONE network per case predicts (μ, log σ²) jointly per parameter, trained
    with MSE for the first mse_warmup_epochs then Gaussian NLL.
    """
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    device = _get_device()
    print(f"[hetero] {obs_a} × {obs_b}  device={device}  epochs={epochs} "
          f"(MSE warmup {mse_warmup_epochs}, then NLL)")

    y, logflag, means, stds, observable_block = _load_data()
    obs1, obs2 = sorted([obs_a, obs_b])
    all_observables = {obs1, obs2}
    x_raw_dict = {k: observable_block[k].numpy() for k in all_observables}
    x_normalized_dict = {k: pipeline.normalize(observable_block[k].numpy()) for k in all_observables}

    noise_cases = pipeline.default_sbi_reference_cases(obs1, obs2)

    output_dim = y.shape[1]
    n_val = int(len(y) * val_fraction)
    n_test = int(len(y) * test_fraction)
    torch.manual_seed(seed); np.random.seed(seed)
    split_perm = torch.randperm(len(y))
    idx_test = split_perm[:n_test]
    idx_val = split_perm[n_test:n_test + n_val]
    idx_train = split_perm[n_test + n_val:]
    perm = np.random.permutation(len(idx_test))

    pipeline.configure(
        observable_1=obs1, observable_2=obs2,
        x_normalized_dict=x_normalized_dict, x_raw_dict=x_raw_dict,
        y=y, idx_val=idx_val, idx_test=idx_test, idx_train=idx_train,
        batch_size=batch_size, device=device,
        logflag=logflag, means=means, stds=stds, output_dim=output_dim,
        hidden_dims=list(hetero_hidden_dims), dropout_rate=hetero_dropout, epochs=epochs,
        perm=perm,
    )

    all_results = []
    for i, (case_name, selected_observables) in enumerate(noise_cases.items()):
        t_case = time.time()
        print(f"\n       [{i+1}/{len(noise_cases)}] hetero training {case_name}")
        input_dim = sum(x_raw_dict[k].shape[1] for k in selected_observables)
        result = {
            "case_name": case_name,
            "selected_observables": selected_observables,
        }
        # Also need to train a plain F* for R² comparison (matches moment-example structure)
        # But: hetero_model's μ head IS a mean net → we can compute R² from it directly.
        pipeline.fit_hetero_case(
            result, focus_params=list(focus_params),
            models_module=models, train_module=train,
            epochs=epochs, mse_warmup_epochs=mse_warmup_epochs,
            hetero_hidden_dims=list(hetero_hidden_dims),
            hetero_dropout=hetero_dropout,
            hetero_wd=hetero_wd,
            restore_best_weights=restore_best_weights,
            best_weights_smoothing_window=best_weights_smoothing_window,
        )
        # Adapter so get_case_predictions (expects a mean-only Module) can use the μ head
        result["model"] = _HeteroMuOnly(result["hetero_model"]).to(device)
        result["train_losses"] = result["hetero_train_losses"]
        result["val_losses"] = result["hetero_val_losses"]
        all_results.append(result)
        print(f"          case done in {(time.time()-t_case)/60:.1f} min")

    pipeline.configure(all_results=all_results)

    r2_matrix = np.zeros((len(all_results), output_dim))
    for ri, r in enumerate(all_results):
        p, t = pipeline.get_case_predictions(r, mode="aligned")
        r2_matrix[ri] = r2_score(t, p, multioutput="raw_values")

    torch.save({
        "obs_a": obs_a, "obs_b": obs_b, "obs1": obs1, "obs2": obs2,
        "noise_cases": noise_cases,
        "all_results": all_results,
        "r2_matrix": r2_matrix,
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
        "seed": seed,
        "training_type": "hetero_nll",
    }, os.path.join(out_dir, "results.pt"))

    dt = (time.time() - t0) / 60
    print(f"\n[hetero] DONE {obs_a} × {obs_b}  in {dt:.1f} min  → {out_dir}")


def run_sbi_example(obs_a, obs_b, out_dir, *,
                     density_estimator="maf",
                     hidden_features=50, num_transforms=5,
                     training_batch_size=64, learning_rate=5e-4,
                     validation_fraction=0.1,
                     stop_after_epochs=50, max_epochs=1000,
                     prior_bound=6.0,
                     val_fraction=0.1, test_fraction=0.1, seed=0,
                     focus_params=("θ0", "θ1", "θ2", "θ4", "θ7", "θ8", "θ11", "θ16")):
    """Train NPE-C (SBI) with a MAF density estimator on one pair.

    One posterior per case, saved into results.pt. Amortized: at inference,
    sample from posterior at any x. Focus params only (k=8 is sensible for a MAF).
    """
    import sbi_pipeline as sbi_pl
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    device = _get_device()   # sbi uses CPU internally per the pipeline setting
    print(f"[sbi] {obs_a} × {obs_b}  device=cpu (SBI)  epochs<={max_epochs}")

    y, logflag, means, stds, observable_block = _load_data()
    obs1, obs2 = sorted([obs_a, obs_b])
    all_observables = {obs1, obs2}
    x_raw_dict = {k: observable_block[k].numpy() for k in all_observables}
    x_normalized_dict = {k: pipeline.normalize(observable_block[k].numpy()) for k in all_observables}

    # Also used by _analyze_helpers.train_sbi_for_pair — the in-process,
    # Agg-free SBI trainer for interactive notebooks (this module forces the
    # Agg matplotlib backend at import time, which breaks inline Jupyter
    # plotting, so that trainer deliberately avoids importing it and keeps
    # its own copy of the split/case-setup logic).
    noise_cases = pipeline.default_sbi_reference_cases(obs1, obs2)

    output_dim = y.shape[1]
    n_val = int(len(y) * val_fraction)
    n_test = int(len(y) * test_fraction)
    torch.manual_seed(seed); np.random.seed(seed)
    split_perm = torch.randperm(len(y))
    idx_test = split_perm[:n_test]
    idx_val = split_perm[n_test:n_test + n_val]
    idx_train = split_perm[n_test + n_val:]
    perm = np.random.permutation(len(idx_test))

    pipeline.configure(
        observable_1=obs1, observable_2=obs2,
        x_normalized_dict=x_normalized_dict, x_raw_dict=x_raw_dict,
        y=y, idx_val=idx_val, idx_test=idx_test, idx_train=idx_train,
        batch_size=training_batch_size, device=device,
        logflag=logflag, means=means, stds=stds, output_dim=output_dim,
        hidden_dims=[128, 64], dropout_rate=0.4, epochs=max_epochs,
        perm=perm,
    )

    all_results = []
    for i, (case_name, selected_observables) in enumerate(noise_cases.items()):
        t_case = time.time()
        print(f"\n       [{i+1}/{len(noise_cases)}] SBI training {case_name}")
        result = {
            "case_name": case_name,
            "selected_observables": selected_observables,
        }
        sbi_pl.fit_sbi_posterior(
            result, focus_params=list(focus_params),
            density_estimator=density_estimator,
            hidden_features=hidden_features, num_transforms=num_transforms,
            training_batch_size=training_batch_size, learning_rate=learning_rate,
            validation_fraction=validation_fraction,
            stop_after_epochs=stop_after_epochs, max_epochs=max_epochs,
            prior_bound=prior_bound, seed=seed,
        )
        all_results.append(result)
        print(f"          case done in {(time.time()-t_case)/60:.1f} min")

    pipeline.configure(all_results=all_results)

    torch.save({
        "obs_a": obs_a, "obs_b": obs_b, "obs1": obs1, "obs2": obs2,
        "noise_cases": noise_cases,
        "all_results": all_results,
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
        "seed": seed,
        "training_type": "sbi_npe_maf",
    }, os.path.join(out_dir, "results.pt"))

    dt = (time.time() - t0) / 60
    print(f"\n[sbi] DONE {obs_a} × {obs_b}  in {dt:.1f} min  → {out_dir}")
