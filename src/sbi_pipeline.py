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
                                   space="normalized", mode="aligned", perm=None):
    """For each test sim, sample the posterior and compute empirical (μ, Σ).

    mode: "aligned" | "obs1_vs_truth" | "obs2_vs_truth" — same shuffle-test
      semantics as pipeline.resolve_shuffle/get_case_predictions. "obs1_vs_truth"
      shuffles observable_2's rows (via `perm`) before feeding the posterior,
      truths stay unpermuted — R² (from the posterior mean) survives only if
      the flow actually reads observable_1. "obs2_vs_truth" is the dual (shuffles
      observable_1). Keys are intersected with this case's observables, so
      single-observable reference cases fall out as natural no-ops, exactly like
      the mean-net shuffle test.
    perm: permutation over local positions in `indices`. Required for a
      reproducible/averaged shuffle; if omitted and mode != "aligned", a fresh
      random permutation is drawn.

    Returns:
      mu:    [N, k]     empirical mean of posterior samples per sim
      cov:   [N, k, k]  empirical covariance
      truth: [N, k]     ground truth for each sim (never shuffled)
      samples_per_sim:  [N, n_samples, k]  raw samples (for non-Gaussian analysis)
    """
    if indices is None:
        indices = np.asarray(_pipeline._eval_indices())
    indices = np.asarray(indices)

    focus_indices = result["sbi_focus_indices"]
    k = len(focus_indices)
    N = len(indices)

    keys_to_shuffle, _ = _pipeline.resolve_shuffle(result["selected_observables"], mode)
    if keys_to_shuffle and perm is None:
        perm = np.random.permutation(N)

    # Build x_test for the case's observables — clean, except keys_to_shuffle
    # (mode != "aligned"), which get their rows permuted. Truths stay put.
    x_list = []
    keys_sorted = sorted(result["selected_observables"].keys())
    for key in keys_sorted:
        arr = _pipeline.x_normalized_dict[key][indices]
        if key in keys_to_shuffle:
            arr = arr[perm]
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


def sbi_shuffle_r2(result, mode, *, indices=None, n_samples=1000, space="log_partial"):
    """R² of the posterior mean vs truth under a shuffle mode, averaged over
    several independent shuffle draws.

    SBI analog of pipeline.average_r2_over_perms: the point estimate here is
    the posterior mean (empirical, from re-sampling the flow at the chimera x
    each draw), not a single deterministic forward pass, so each perm draw
    means both a fresh row-permutation AND fresh posterior samples.

    Returns:
      r2_mean, r2_std: [k] arrays — mean/std R² per focus param across draws.
      If mode == "aligned", there's one draw (no randomness to average over)
      and r2_std is all zeros.
    """
    from sklearn.metrics import r2_score
    if indices is None:
        indices = np.asarray(_pipeline._eval_indices())
    indices = np.asarray(indices)
    N = len(indices)

    n_draws = 1 if mode == "aligned" else 5
    draws = []
    for d in range(n_draws):
        perm = None if mode == "aligned" else np.random.default_rng(d).permutation(N)
        mu, _, truth, _ = predict_moments_from_samples(
            result, indices=indices, n_samples=n_samples, space=space,
            mode=mode, perm=perm)
        draws.append(r2_score(truth, mu, multioutput="raw_values"))
    draws = np.stack(draws, axis=0)
    return draws.mean(axis=0), draws.std(axis=0)


# ---------------------------------------------------------------------------
# Fit-quality diagnostics — is the trained flow any good, independent of
# whether it's reading the right observable (that's what sbi_shuffle_r2 is for).
# ---------------------------------------------------------------------------


def plot_sbi_loss_curves(result, *, figsize=(6, 4)):
    """Train/val loss (NPE-C's internal MLE loss) per epoch for one case.

    Cheapest possible sanity check — no resampling, just the numbers
    fit_sbi_posterior already stored. Read it as:
      - val loss still dropping at the last epoch -> stop_after_epochs/max_epochs
        cut training short, consider raising them.
      - val well above train for a long stretch -> overfitting.
      - NaN/inf anywhere -> unstable training, usually prior_bound too tight
        or learning_rate too high.
    """
    import matplotlib.pyplot as plt

    tl = result.get("sbi_train_losses") or []
    vl = result.get("sbi_val_losses") or []
    fig, ax = plt.subplots(figsize=figsize)
    if tl:
        ax.plot(range(1, len(tl) + 1), tl, label="train", color="steelblue")
    if vl:
        ax.plot(range(1, len(vl) + 1), vl, label="val", color="darkorange")
    ax.set_xlabel("epoch", fontsize=9)
    ax.set_ylabel("NPE-C loss", fontsize=9)
    ax.set_title(f"SBI loss curves — {result['case_name']}", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def compute_sbc_ranks(result, *, indices=None, n_samples=1000, space="log_partial"):
    """Simulation-based calibration ranks (Talts et al. 2018): for each
    held-out test sim, the rank of the true theta among n_samples posterior
    draws, per focus param. A well-calibrated posterior gives ranks uniformly
    distributed in [0, n_samples] across the test set — this works directly
    on the raw samples, so unlike a Gaussian pull test it doesn't require (or
    assume) the posterior is Gaussian-shaped.

    Returns:
      ranks: [N, k] int array, each entry in [0, n_samples]
      labels: focus param names
    """
    _, _, truth, samples = predict_moments_from_samples(
        result, indices=indices, n_samples=n_samples, space=space)
    ranks = (samples < truth[:, None, :]).sum(axis=1)   # [N, k]
    labels = result.get("sbi_focus_params") or result.get("moment_focus_params") \
             or [f"θ{i}" for i in range(ranks.shape[1])]
    return ranks, labels


def plot_sbc_rank_hist(result, *, indices=None, n_samples=1000, space="log_partial",
                        n_bins=None, figsize_per_panel=3.2):
    """SBC rank histogram, one panel per focus param. Ranks should be uniform
    if the posterior is well-calibrated:
      - U-shaped (piled up at both ends)  -> posterior too narrow (overconfident)
      - hump in the middle                -> posterior too wide (underconfident)
      - skewed to one side                -> biased mean

    The shaded band is the ~95% expected range under perfect calibration
    (normal approximation to Binomial(N, 1/n_bins)) — bars poking outside it
    are likely real miscalibration, not just sampling noise from a modest
    test-set size.
    """
    import matplotlib.pyplot as plt

    ranks, labels = compute_sbc_ranks(result, indices=indices, n_samples=n_samples, space=space)
    N, k = ranks.shape
    n_bins = n_bins or max(5, min(20, N // 5))

    fig, axes = plt.subplots(1, k, figsize=(figsize_per_panel * k, figsize_per_panel), squeeze=False)
    axes = axes[0]
    bin_edges = np.linspace(0, n_samples, n_bins + 1)
    expected = N / n_bins
    band = 1.96 * np.sqrt(N * (1 / n_bins) * (1 - 1 / n_bins))

    for i in range(k):
        ax = axes[i]
        ax.axhspan(expected - band, expected + band, color="grey", alpha=0.25,
                   label="95% band" if i == 0 else None)
        ax.axhline(expected, color="grey", linewidth=1.0, linestyle="--")
        ax.hist(ranks[:, i], bins=bin_edges, color="steelblue", alpha=0.8, edgecolor="white")
        ax.set_title(labels[i], fontsize=9)
        ax.set_xlabel("rank", fontsize=8)
        if i == 0:
            ax.set_ylabel("count", fontsize=8)
            ax.legend(fontsize=7, frameon=False)

    fig.suptitle(f"SBC rank histogram — case {result['case_name']}  ({space}, N={N} test sims, "
                 f"n_samples={n_samples})\nUniform = calibrated. U-shape = overconfident. Hump = underconfident.",
                 fontsize=10, y=1.05)
    fig.tight_layout()
    return fig


def plot_sbc_coverage(result, *, indices=None, n_samples=1000, space="log_partial",
                       figsize=(5, 5)):
    """Empirical vs. nominal coverage, aggregated across the test set — the
    same rank information as plot_sbc_rank_hist, condensed to one diagonal-line
    plot instead of a histogram grid. On the y=x line -> calibrated; below it
    -> overconfident (credible intervals too narrow, real value falls outside
    them more often than claimed); above it -> underconfident.
    """
    import matplotlib.pyplot as plt

    ranks, labels = compute_sbc_ranks(result, indices=indices, n_samples=n_samples, space=space)
    N, k = ranks.shape
    frac_rank = ranks / n_samples
    levels = np.linspace(0.01, 0.99, 50)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, label="ideal")
    cmap = plt.get_cmap("tab10")
    for i in range(k):
        empirical = [np.mean(np.abs(frac_rank[:, i] - 0.5) <= level / 2) for level in levels]
        ax.plot(levels, empirical, color=cmap(i), linewidth=1.6, label=labels[i])
    ax.set_xlabel("nominal credible level", fontsize=9)
    ax.set_ylabel("empirical coverage", fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title(f"SBC coverage — case {result['case_name']}  ({space}, N={N})", fontsize=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Generalizable corner plots — beyond single-sim views.
# ---------------------------------------------------------------------------


def _mode_tag(mode):
    return "" if mode == "aligned" else f"  [SHUFFLE: {mode}]"


def _corner_grid(k, figsize_per_panel):
    """Shared k×k corner-plot scaffold: axes grid with the upper triangle hidden."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(k, k, figsize=(figsize_per_panel * k, figsize_per_panel * k),
                              squeeze=False)
    for i in range(k):
        for j in range(k):
            if j > i:
                axes[i, j].set_visible(False)
    return fig, axes


def _corner_finish(fig, axes, labels):
    """Shared corner-plot finishing: border axis labels + aggregate legend."""
    k = len(labels)
    for i in range(k):
        for j in range(k):
            if j > i:
                continue
            ax = axes[i, j]
            if i == k - 1: ax.set_xlabel(labels[j], fontsize=8)
            else: ax.set_xticklabels([])
            if j == 0 and i > 0: ax.set_ylabel(labels[i], fontsize=8)
            elif j > 0: ax.set_yticklabels([])

    handles, lbls = [], []
    seen = set()
    for a in axes.flat:
        for h, l in zip(*a.get_legend_handles_labels()):
            if l and l not in seen:
                handles.append(h); lbls.append(l); seen.add(l)
    if handles:
        fig.legend(handles, lbls, loc="upper right", frameon=False, fontsize=9,
                   bbox_to_anchor=(0.98, 0.98))
    fig.tight_layout()


def plot_sbi_pull_corner(result, *, indices=None, n_samples=1000,
                          space="log_partial", figsize_per_panel=1.7,
                          mode="aligned", perm=None):
    """Joint pull calibration corner. For each test sim compute pull z_i =
    (θ_true - μ_sample) / σ_sample, then draw the k×k corner of z across all
    test sims. If perfectly calibrated (unbiased μ, correct Σ), every panel is
    a standard Gaussian: diagonals ≈ N(0, 1), off-diagonals are isotropic
    scatter with no correlation.

    Non-zero correlation in the off-diagonal panels means the covariance is
    mis-estimating cross-parameter uncertainty — even if per-param σ is right
    marginally, the joint uncertainty structure is off.

    One plot summarizes the whole test set. Population-level calibration.

    mode/perm: pass "obs1_vs_truth" or "obs2_vs_truth" to see calibration
    collapse under the shuffle test (see predict_moments_from_samples).
    """
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    if indices is None:
        indices = np.asarray(_pipeline._eval_indices())
    indices = np.asarray(indices)

    mu, cov, truth, _ = predict_moments_from_samples(
        result, indices=indices, n_samples=n_samples, space=space,
        mode=mode, perm=perm)
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

    fig.suptitle(f"Pull-corner — case {result['case_name']}  ({space}){_mode_tag(mode)}\n"
                  f"Diagonals: 1D pull. Off-diagonals: 2D pull scatter (well-calib → isotropic, r≈0)",
                  fontsize=10, y=0.995)
    fig.tight_layout()
    return fig


def plot_sbi_multi_sim_corner(result, sim_indices, *, n_samples=2000,
                                space="log_partial", figsize_per_panel=1.7,
                                show_true=True, cmap_name="tab10",
                                mode="aligned", perm=None):
    """Overlay corner-plot samples from multiple test sims in one figure.

    For each chosen sim, samples the posterior and draws its diagonal (1D histogram)
    and off-diagonal (2D scatter). Different sims in different colors. Shows how
    much the posterior *shape* varies across x (as opposed to per-sim, which shows
    the shape for ONE x).

    mode/perm: pass "obs1_vs_truth" or "obs2_vs_truth" to see how posterior
    shape/location degrades under the shuffle test (see predict_moments_from_samples).
    """
    import matplotlib.pyplot as plt

    sim_indices = list(sim_indices)
    all_indices = np.asarray(_pipeline._eval_indices())
    _, _, truth, samples_all = predict_moments_from_samples(
        result, indices=all_indices, n_samples=n_samples, space=space,
        mode=mode, perm=perm)
    labels = result.get("sbi_focus_params") or result.get("moment_focus_params") \
             or [f"θ{i}" for i in range(samples_all.shape[-1])]
    k = samples_all.shape[-1]

    fig, axes = _corner_grid(k, figsize_per_panel)
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
                    continue
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

    _corner_finish(fig, axes, labels)
    fig.suptitle(f"Multi-sim corner overlay — {len(sim_indices)} test sims, "
                  f"case {result['case_name']}{_mode_tag(mode)}", fontsize=10, y=0.995)
    return fig


def plot_sbi_case_overlay_corner(results, sim_idx, *, n_samples=5000, space="log_partial",
                                   figsize_per_panel=1.8, cmap_name="tab10", case_labels=None):
    """Overlay posterior samples from several trained CASES at one test sim,
    one color per case. Dual of plot_sbi_multi_sim_corner (which overlays
    several sims for one case) — this overlays several cases for one sim, e.g.
    to compare how the posterior shape changes as observables are added/noised.

    results: list of trained result dicts (each needs sbi_posterior).
    case_labels: optional display names, same order as results (defaults to
    each result's case_name).
    """
    import matplotlib.pyplot as plt

    per_case = []
    for r in results:
        _, _, truth, samples = predict_moments_from_samples(
            r, indices=None, n_samples=n_samples, space=space)
        per_case.append((samples[sim_idx], truth[sim_idx]))
    labels = results[0].get("sbi_focus_params") or results[0].get("moment_focus_params") \
             or [f"θ{i}" for i in range(per_case[0][0].shape[-1])]
    k = len(labels)
    truth_vec = per_case[0][1]   # same sim -> same truth regardless of case
    case_labels = case_labels or [r["case_name"] for r in results]

    fig, axes = _corner_grid(k, figsize_per_panel)
    cmap = plt.get_cmap(cmap_name)

    for c_idx, (S, _truth) in enumerate(per_case):
        color = cmap(c_idx)
        for i in range(k):
            for j in range(k):
                if j > i:
                    continue
                ax = axes[i, j]
                if i == j:
                    ax.hist(S[:, i], bins=40, density=True, histtype="step",
                             color=color, linewidth=1.4,
                             label=case_labels[c_idx] if i == 0 else None)
                    ax.axvline(truth_vec[i], color="red", linestyle="--", linewidth=1.0, alpha=0.7)
                else:
                    ax.scatter(S[:, j], S[:, i], s=1, alpha=0.05, color=color)
                    ax.plot(truth_vec[j], truth_vec[i], "x", color="red",
                             markersize=9, markeredgewidth=1.6)

    _corner_finish(fig, axes, labels)
    fig.suptitle(f"SBI posterior samples — val sim #{sim_idx}  (real distribution, not Gaussian)",
                  fontsize=10, y=0.995)
    return fig
