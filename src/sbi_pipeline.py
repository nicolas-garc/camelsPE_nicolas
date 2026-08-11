"""Neural Posterior Estimation (SBI) via the sbi package.

Trains a MAF normalizing flow per case to approximate p(θ_focus | x). Amortized:
evaluate the learned posterior at any test x cheaply. Unlike moment/hetero,
this can capture non-Gaussian posterior structure (multimodality, skew).

Uses the shared state from src/pipeline.py (idx_train, idx_test, x_normalized_dict,
y, focus indices, etc.) so it plugs into the same 3-way split as the other pipelines.
"""
import os
import sys
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import pipeline as _pipeline   # for module state + _resolve_focus_indices


def prepare_sbi_data(selected_observables, indices, focus_indices, *,
                      x_normalized_dict, y, seed=0):
    """Build (θ, x) tensors for SBI training on one case.

    Applies the case's per-observable noise levels (single realization per sim
    — SBI trains on a fixed dataset, no per-epoch noise resampling).

    Returns:
      theta_tensor: [N, k] — focus-param slice of the parameter vector
      x_tensor:     [N, input_dim] — concatenated observables, with noise added
    """
    rng = np.random.default_rng(seed)
    indices = np.asarray(indices)
    x_list = []
    for key in sorted(selected_observables.keys()):
        noise_level = selected_observables[key]
        arr = x_normalized_dict[key][indices].copy()
        if noise_level > 0:
            arr = arr + rng.normal(0.0, noise_level, size=arr.shape)
        x_list.append(torch.from_numpy(arr).float())
    x_tensor = torch.cat(x_list, dim=1)
    theta_tensor = y[indices][:, focus_indices].float().clone()
    return theta_tensor, x_tensor


def fit_sbi_posterior(result, focus_params, *,
                       density_estimator="maf",
                       hidden_features=50, num_transforms=5,
                       training_batch_size=64, learning_rate=5e-4,
                       validation_fraction=0.1,
                       stop_after_epochs=50, max_epochs=1000,
                       prior_bound=6.0, seed=0,
                       all_param_names=None):
    """Train an amortized NPE-C posterior on this case's data.

    Uses sbi's NPE_C (formerly SNPE_C — DIRECT posterior estimation via a
    normalizing flow). Attaches to `result`:
      sbi_posterior          — the trained sbi posterior object; call .sample(...)
      sbi_density_estimator  — the raw density estimator (nflows module)
      sbi_focus_indices      — parameter indices this posterior covers
      sbi_focus_params       — labels
      sbi_train_losses, sbi_val_losses  — loss curves (from sbi's internal train)
    """
    # Import late — sbi is a heavy dependency
    from sbi.inference import NPE_C
    from sbi.utils import BoxUniform
    from sbi.neural_nets import posterior_nn

    focus_indices, focus_labels = _pipeline._resolve_focus_indices(
        focus_params, all_param_names)
    k = len(focus_indices)

    # Prior: wide box uniform over normalized-log space (parameters are z-scored,
    # so ±6 covers effectively the full support with room to spare). NPE-C
    # doesn't sample from the prior at training time — it's used at inference
    # to reject samples outside the support.
    prior = BoxUniform(
        low=torch.full((k,), -prior_bound),
        high=torch.full((k,),  prior_bound),
    )

    # Build (θ, x) training tensors from idx_train, with case noise applied
    theta_train, x_train = prepare_sbi_data(
        result["selected_observables"], _pipeline.idx_train, focus_indices,
        x_normalized_dict=_pipeline.x_normalized_dict, y=_pipeline.y, seed=seed,
    )
    print(f"  [{result['case_name']}] SBI training set: θ shape {tuple(theta_train.shape)}, "
          f"x shape {tuple(x_train.shape)}")

    # NPE_C with MAF density estimator (Jo 2023 default)
    density_estimator_builder = posterior_nn(
        model=density_estimator,
        hidden_features=hidden_features,
        num_transforms=num_transforms,
    )
    inference = NPE_C(prior=prior, density_estimator=density_estimator_builder,
                      device="cpu")  # MPS support for sbi is patchy; CPU is fine at k=8
    inference.append_simulations(theta_train, x_train)

    density_estimator_trained = inference.train(
        training_batch_size=training_batch_size,
        learning_rate=learning_rate,
        validation_fraction=validation_fraction,
        stop_after_epochs=stop_after_epochs,
        max_num_epochs=max_epochs,
        show_train_summary=True,
    )
    posterior = inference.build_posterior(density_estimator_trained)

    # Save loss curves — sbi stores them on inference._summary
    summary = inference.summary
    tl = summary.get("training_loss", [])
    vl = summary.get("validation_loss", [])

    result["sbi_posterior"] = posterior
    result["sbi_density_estimator"] = density_estimator_trained
    result["sbi_focus_indices"] = focus_indices
    result["sbi_focus_params"] = list(focus_labels)
    result["sbi_train_losses"] = list(tl)
    result["sbi_val_losses"] = list(vl)
    return result


def sample_posterior(result, x_obs, n_samples=5000, show_progress=False):
    """Sample from the trained posterior at one observation.

    Args:
      x_obs: [input_dim] tensor — one observed x

    Returns:
      samples: [n_samples, k] numpy array
    """
    if "sbi_posterior" not in result:
        raise ValueError(f"case {result['case_name']!r} has no sbi_posterior — "
                         "call fit_sbi_posterior first.")
    if isinstance(x_obs, np.ndarray):
        x_obs = torch.from_numpy(x_obs).float()
    if x_obs.ndim == 2 and x_obs.shape[0] == 1:
        x_obs = x_obs.squeeze(0)
    posterior = result["sbi_posterior"]
    samples = posterior.sample((n_samples,), x=x_obs,
                                show_progress_bars=show_progress)
    return samples.detach().cpu().numpy()


def predict_moments_from_samples(result, *, indices=None, n_samples=2000,
                                   space="normalized"):
    """For each test sim, sample the posterior and compute empirical (μ, Σ).

    Returns:
      mu:    [N, k]     empirical mean of posterior samples per sim
      cov:   [N, k, k]  empirical covariance
      truth: [N, k]     ground truth for each sim
      samples_per_sim:  [N, n_samples, k]  raw samples (for non-Gaussian analysis)
    """
    if indices is None:
        indices = np.asarray(_pipeline._eval_indices())
    indices = np.asarray(indices)

    focus_indices = result["sbi_focus_indices"]
    k = len(focus_indices)
    N = len(indices)

    # Build clean x_test for the case's observables — evaluation is always on clean
    x_list = []
    keys_sorted = sorted(result["selected_observables"].keys())
    for key in keys_sorted:
        arr = _pipeline.x_normalized_dict[key][indices]
        x_list.append(torch.from_numpy(arr).float())
    x_test = torch.cat(x_list, dim=1)   # [N, input_dim]

    posterior = result["sbi_posterior"]
    mu = np.empty((N, k))
    cov = np.empty((N, k, k))
    samples_per_sim = np.empty((N, n_samples, k))

    for i in range(N):
        s = posterior.sample((n_samples,), x=x_test[i], show_progress_bars=False)
        s = s.detach().cpu().numpy()
        samples_per_sim[i] = s
        mu[i] = s.mean(axis=0)
        cov[i] = np.cov(s, rowvar=False)

    truth_norm = _pipeline.y[indices].numpy()[:, focus_indices]

    if space == "normalized":
        return mu, cov, truth_norm, samples_per_sim
    if space == "log_partial":
        D = _pipeline.stds[focus_indices]
        m = _pipeline.means[focus_indices]
        mu_pl = mu * D + m
        cov_pl = cov * (D[None, :, None] * D[None, None, :])
        truth_pl = truth_norm * D + m
        samples_pl = samples_per_sim * D[None, None, :] + m[None, None, :]
        return mu_pl, cov_pl, truth_pl, samples_pl
    raise ValueError(f"space must be 'normalized' or 'log_partial'; got {space!r}")


# ---------------------------------------------------------------------------
# Generalizable corner plots — beyond single-sim views.
# ---------------------------------------------------------------------------


def plot_sbi_pull_corner(result, *, indices=None, n_samples=1000,
                          space="log_partial", figsize_per_panel=1.7):
    """Joint pull calibration corner. For each test sim compute pull z_i =
    (θ_true - μ_sample) / σ_sample, then draw the k×k corner of z across all
    test sims. If perfectly calibrated (unbiased μ, correct Σ), every panel is
    a standard Gaussian: diagonals ≈ N(0, 1), off-diagonals are isotropic
    scatter with no correlation.

    Non-zero correlation in the off-diagonal panels means the covariance is
    mis-estimating cross-parameter uncertainty — even if per-param σ is right
    marginally, the joint uncertainty structure is off.

    One plot summarizes the whole test set. Population-level calibration.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    if indices is None:
        indices = np.asarray(_pipeline._eval_indices())
    indices = np.asarray(indices)

    mu, cov, truth, _ = predict_moments_from_samples(
        result, indices=indices, n_samples=n_samples, space=space)
    sigma = np.sqrt(np.clip(np.diagonal(cov, axis1=1, axis2=2), 1e-30, None))
    pull = (truth - mu) / sigma                # [N, k]

    labels = result.get("sbi_focus_params") or result.get("moment_focus_params") \
             or [f"θ{i}" for i in range(pull.shape[1])]
    k = pull.shape[1]

    fig, axes = plt.subplots(k, k, figsize=(figsize_per_panel * k, figsize_per_panel * k),
                              squeeze=False)
    xs = np.linspace(-4, 4, 200)
    ref_pdf = norm.pdf(xs)

    for i in range(k):
        for j in range(k):
            ax = axes[i, j]
            if j > i:
                ax.set_visible(False); continue
            if i == j:
                z = pull[:, i]; z = z[np.isfinite(z)]
                ax.hist(z, bins=20, density=True, color="steelblue", alpha=0.6,
                         edgecolor="none")
                ax.plot(xs, ref_pdf, "k--", linewidth=1.2, label="N(0,1)" if i == 0 else None)
                ax.axvline(0, color="red", linewidth=0.5, alpha=0.6)
                ax.set_xlim(-4, 4); ax.set_yticks([])
                if i == 0:
                    ax.text(0.03, 0.97, f"std={z.std():.2f}",
                             transform=ax.transAxes, va="top", ha="left",
                             fontsize=7, family="monospace",
                             bbox=dict(boxstyle="round", facecolor="white", alpha=0.75))
            else:
                zj = pull[:, j]; zi = pull[:, i]
                mask = np.isfinite(zi) & np.isfinite(zj)
                ax.scatter(zj[mask], zi[mask], s=6, alpha=0.5, color="steelblue")
                ax.axhline(0, color="grey", linewidth=0.4, alpha=0.5)
                ax.axvline(0, color="grey", linewidth=0.4, alpha=0.5)
                # 1σ box for reference
                ax.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], "k-", linewidth=0.5,
                         alpha=0.5)
                r_corr = np.corrcoef(zj[mask], zi[mask])[0, 1]
                ax.text(0.03, 0.97, f"r={r_corr:+.2f}",
                         transform=ax.transAxes, va="top", ha="left", fontsize=7,
                         family="monospace",
                         bbox=dict(boxstyle="round", facecolor="white", alpha=0.75))
                ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.grid(True, alpha=0.2)
            if i == k - 1: ax.set_xlabel(labels[j], fontsize=8)
            else: ax.set_xticklabels([])
            if j == 0 and i > 0: ax.set_ylabel(labels[i], fontsize=8)
            elif j > 0: ax.set_yticklabels([])

    fig.suptitle(f"Pull-corner — case {result['case_name']}  ({space})\n"
                  f"Diagonals: 1D pull. Off-diagonals: 2D pull scatter (well-calib → isotropic, r≈0)",
                  fontsize=10, y=0.995)
    fig.tight_layout()
    return fig


def plot_sbi_multi_sim_corner(result, sim_indices, *, n_samples=2000,
                                space="log_partial", figsize_per_panel=1.7,
                                show_true=True, cmap_name="tab10"):
    """Overlay corner-plot samples from multiple test sims in one figure.

    For each chosen sim, samples the posterior and draws its diagonal (1D histogram)
    and off-diagonal (2D scatter). Different sims in different colors. Shows how
    much the posterior *shape* varies across x (as opposed to per-sim, which shows
    the shape for ONE x).
    """
    import matplotlib.pyplot as plt

    sim_indices = list(sim_indices)
    all_indices = np.asarray(_pipeline._eval_indices())
    _, _, truth, samples_all = predict_moments_from_samples(
        result, indices=all_indices, n_samples=n_samples, space=space)
    labels = result.get("sbi_focus_params") or result.get("moment_focus_params") \
             or [f"θ{i}" for i in range(samples_all.shape[-1])]
    k = samples_all.shape[-1]

    fig, axes = plt.subplots(k, k, figsize=(figsize_per_panel * k, figsize_per_panel * k),
                              squeeze=False)
    cmap = plt.get_cmap(cmap_name)

    # Global axis limits from all shown sims
    S_shown = np.concatenate([samples_all[s] for s in sim_indices], axis=0)
    axis_ranges = [(S_shown[:, i].min(), S_shown[:, i].max()) for i in range(k)]
    if show_true:
        for i in range(k):
            lo, hi = axis_ranges[i]
            for s in sim_indices:
                lo = min(lo, truth[s, i]); hi = max(hi, truth[s, i])
            axis_ranges[i] = (lo, hi)

    for s_idx, s in enumerate(sim_indices):
        S = samples_all[s]  # [n_samples, k]
        color = cmap(s_idx)
        for i in range(k):
            for j in range(k):
                if j > i:
                    axes[i, j].set_visible(False); continue
                ax = axes[i, j]
                if i == j:
                    ax.hist(S[:, i], bins=30, density=True, histtype="step",
                             color=color, linewidth=1.3,
                             label=f"sim #{s}" if i == 0 else None)
                    if show_true:
                        ax.axvline(truth[s, i], color=color, linewidth=0.9,
                                    linestyle=":", alpha=0.7)
                else:
                    ax.scatter(S[:, j], S[:, i], s=1, alpha=0.05, color=color)
                    if show_true:
                        ax.plot(truth[s, j], truth[s, i], "x", color=color,
                                 markersize=6, markeredgewidth=1.2, alpha=0.9)

    for i in range(k):
        for j in range(k):
            if j > i: continue
            ax = axes[i, j]
            ax.set_xlim(axis_ranges[j])
            if i != j:
                ax.set_ylim(axis_ranges[i])
                ax.grid(True, alpha=0.2)
            else:
                ax.set_yticks([])
            if i == k - 1: ax.set_xlabel(labels[j], fontsize=8)
            else: ax.set_xticklabels([])
            if j == 0 and i > 0: ax.set_ylabel(labels[i], fontsize=8)
            elif j > 0: ax.set_yticklabels([])

    # Aggregate legend
    handles, lbls = [], []
    seen = set()
    for a in axes.flat:
        for h, l in zip(*a.get_legend_handles_labels()):
            if l and l not in seen:
                handles.append(h); lbls.append(l); seen.add(l)
    if handles:
        fig.legend(handles, lbls, loc="upper right", frameon=False, fontsize=9,
                   bbox_to_anchor=(0.98, 0.98))
    fig.suptitle(f"Multi-sim corner overlay — {len(sim_indices)} test sims, "
                  f"case {result['case_name']}", fontsize=10, y=0.995)
    fig.tight_layout()
    return fig
