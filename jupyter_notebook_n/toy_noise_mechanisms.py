# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
# ---

# %% [markdown]
# # Toy sandbox: noise-injection mechanisms
#
# A miniature of the real pipeline where the ground truth is known, for testing
# degradation mechanisms **without touching the real analysis**. Everything here
# is a linear-Gaussian universe solved with least squares (what MSE training
# converges to for a linear problem), plus an optional MLP check at the end that
# uses the real `SimpleMLP`.
#
# Findings this notebook reproduces (session of 2026-07-16):
# - **current** (noise at train, no renorm, clean val): the signal component of
#   the training input is identical to the val input, so the noisy channel's val
#   contribution shrinks by the reliability `1/(1+σ²)` — the diminished
#   dependence the network could learn. Robust degradation. **This is what the
#   real pipeline does, on purpose.**
# - **renorm** (noise + renormalize at train, clean val): learns the *identical*
#   mapping, but validation feeds a signal `sqrt(1+σ²)`× stronger than trained
#   on. Suppression weakens to `1/sqrt(1+σ²)` and never reaches the floor.
#   Rejected.
# - **noise/mix at both** train and val: answers a different question ("how much
#   information does a genuinely noisier measurement carry"). Also reaches the
#   floor; useful as a robustness check.

# %%
import numpy as np
import matplotlib.pyplot as plt

def nrm(a):
    return (a - a.mean(0)) / a.std(0)

def r2(y, p):
    return 1 - ((y - p)**2).sum() / ((y - y.mean())**2).sum()

def fit_linear(X, y):
    A = np.hstack([X, np.ones((len(X), 1))])
    W, *_ = np.linalg.lstsq(A, y, rcond=None)
    return W

def predict(W, X):
    return np.hstack([X, np.ones((len(X), 1))]) @ W

# %% [markdown]
# ## The toy universe
#
# One parameter θ, two observables. O₁ is a noisier view of θ than O₂, so
# degrading O₂ should hand the model off to O₁ and R² should fall to the
# "O₁ alone" floor. Change the two view noises to explore other regimes.

# %%
N, NV = 40_000, 8_000            # total sims, validation size
VIEW_NOISE_O1 = 0.80             # O1 is the weaker observable
VIEW_NOISE_O2 = 0.20             # O2 is the stronger one (the one we degrade)

rng = np.random.default_rng(0)
theta = rng.normal(size=N)
x1 = nrm((theta + VIEW_NOISE_O1 * rng.normal(size=N)).reshape(-1, 1))
x2 = nrm((theta + VIEW_NOISE_O2 * rng.normal(size=N)).reshape(-1, 1))
tr, va = slice(0, N - NV), slice(N - NV, N)

# %% [markdown]
# ## The mechanisms
#
# Each mechanism maps the clean normalized observable to a (train version,
# val version) pair. To try a new one, add a function here and a line in
# `MECHANISMS` — nothing else changes.

# %%
def mech_current(x, nl, rng):
    """Real pipeline: noise at train, NO renormalize, clean validation."""
    return x + rng.normal(0, nl, x.shape), x

def mech_renorm(x, nl, rng):
    """Rejected: renormalize after the noise, clean validation."""
    return nrm(x + rng.normal(0, nl, x.shape)), x

def mech_noise_both(x, nl, rng):
    """Noise + renormalize at train AND val (fresh noise each)."""
    return nrm(x + rng.normal(0, nl, x.shape)), nrm(x + rng.normal(0, nl, x.shape))

def mech_mix_both(x, nl, rng):
    """Variance-preserving mix at both: z = sqrt(1-a)x + sqrt(a)e, Var(z)=1 exactly."""
    a = nl**2 / (1 + nl**2)
    mix = lambda: np.sqrt(1 - a) * x + np.sqrt(a) * rng.normal(size=x.shape)
    return mix(), mix()

MECHANISMS = {
    "current":    (mech_current,    "#D85A30", "-"),
    "renorm":     (mech_renorm,     "#185FA5", "-"),
    "noise_both": (mech_noise_both, "#1D9E75", "--"),
    "mix_both":   (mech_mix_both,   "#534AB7", "-"),
}

def run_mechanism(mech, nl, rng):
    """Train on [x1, degraded x2], evaluate on the val split. Returns R2, |w(O2)|."""
    z2_tr, z2_va = mech(x2, nl, rng)
    W = fit_linear(np.hstack([x1[tr], z2_tr[tr]]), theta[tr])
    p = predict(W, np.hstack([x1[va], z2_va[va]]))
    return r2(theta[va], p), abs(W[1])

# %% [markdown]
# ## Experiment 1 — which mechanisms reach the floor?
#
# The floor is the O₁-alone R²: where the model *should* land once O₂ carries
# nothing. A mechanism that plateaus above the floor is not fully degrading.

# %%
rng = np.random.default_rng(1)
W_floor = fit_linear(x1[tr], theta[tr])
floor = r2(theta[va], predict(W_floor, x1[va]))
W_ceil = fit_linear(np.hstack([x1[tr], x2[tr]]), theta[tr])
ceil = r2(theta[va], predict(W_ceil, np.hstack([x1[va], x2[va]])))
print(f"O1-alone floor R² = {floor:.3f}    both-clean ceiling R² = {ceil:.3f}")

NOISE_LEVELS = [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3.5, 5, 7, 10]
results = {name: [run_mechanism(m, nl, rng) for nl in NOISE_LEVELS]
           for name, (m, _, _) in MECHANISMS.items()}

# %%
fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.5))
for name, (m, color, ls) in MECHANISMS.items():
    a1.plot(NOISE_LEVELS, [v[0] for v in results[name]], ls, color=color, lw=2.2, marker="o", ms=4, label=name)
    a2.plot(NOISE_LEVELS, [v[1] for v in results[name]], ls, color=color, lw=2.2, marker="o", ms=4, label=name)
a1.axhline(floor, color="k", ls=":", lw=2)
a1.text(max(NOISE_LEVELS), floor + 0.005, "O1-alone floor", ha="right", va="bottom", fontsize=9)
a1.axhline(ceil, color="gray", ls=":", lw=1)
a1.set_xlabel("noise_level on O2"); a1.set_ylabel("aligned R²(θ)")
a1.set_title("Degradation curves"); a1.legend(); a1.grid(alpha=0.25)
a2.set_xlabel("noise_level on O2"); a2.set_ylabel("|learned weight on O2|")
a2.set_title("Learned reliance on O2"); a2.legend(); a2.grid(alpha=0.25)
plt.tight_layout()
plt.show()

for name in MECHANISMS:
    gap = results[name][-1][0] - floor
    print(f"{name:<12} R² at max noise: {results[name][-1][0]:.3f}   gap above floor: {gap:+.3f}")

# %% [markdown]
# ## Experiment 2 — the two clean-val schemes learn the *same* mapping
#
# `w_current × sqrt(1+σ²) = w_renorm` at every noise level: renormalizing only
# changes the units of the weight. What differs is that renorm's validation
# feeds that weight a signal `sqrt(1+σ²)`× stronger than it was calibrated on —
# which is exactly why its degradation is weak.

# %%
rng = np.random.default_rng(2)
print(f"{'noise':<8}{'w_current':>12}{'x sqrt(1+s^2)':>16}{'w_renorm':>12}")
for nl in [0.5, 1, 2.5, 5, 10]:
    _, w_cur = run_mechanism(mech_current, nl, np.random.default_rng(42))
    _, w_ren = run_mechanism(mech_renorm, nl, np.random.default_rng(42))
    print(f"{nl:<8}{w_cur:>12.4f}{w_cur*np.sqrt(1+nl**2):>16.4f}{w_ren:>12.4f}")

# %% [markdown]
# ## Experiment 3 — shuffle signatures of the six categories
#
# Construct each category from known ground truth (M1/M2 choose which parameter
# combinations each observable sees), then run the pipeline's exact shuffle
# procedure:
# - **S₁**: scramble O₂, truths stay with O₁ → R² survives iff the model reads O₁
# - **S₂**: scramble O₂ and truths together → R² survives iff the model reads O₂
#
# Off the S₁=S₂ diagonal = asymmetric reliance. Along it, position separates
# synergy (0,0) from redundancy (+,+). Cases 4 and 5 coincide at the origin —
# only the aligned R² tells them apart. Note R² has no floor at 0: an
# uncorrelated but variable prediction scores −1 or worse.

# %%
def shuffle_signature(M1, M2, view_noise=0.3, seed=0):
    """Aligned R², S1, S2 for theta_1, given each observable's view M of theta."""
    g = np.random.default_rng(seed)
    th = g.normal(size=(N, 2))
    o1 = th @ np.atleast_2d(M1).T + view_noise * g.normal(size=(N, np.atleast_2d(M1).shape[0]))
    o2 = th @ np.atleast_2d(M2).T + view_noise * g.normal(size=(N, np.atleast_2d(M2).shape[0]))
    X, n1 = np.hstack([o1, o2]), o1.shape[1]
    W = fit_linear(X[tr], th[tr])
    Xv, thv = X[va], th[va]
    perm = g.permutation(len(Xv))
    Xs = Xv.copy(); Xs[:, n1:] = Xv[perm, n1:]          # scramble O2 only
    A  = r2(thv[:, 0], predict(W, Xv)[:, 0])
    S1 = r2(thv[:, 0], predict(W, Xs)[:, 0])            # truths stay with O1
    S2 = r2(thv[perm, 0], predict(W, Xs)[:, 0])         # truths follow O2
    return A, S1, S2

CATEGORIES = {
    "Case 1 single, no gain":  ([[1, 0], [0, 1]], [[0, 0]]),
    "Case 2 single, improves": ([[1, 1]],         [[0, 1]]),
    "Case 3 both, better":     ([[1, 0.5]],       [[0.5, 1]]),
    "Case 4 only combined":    ([[1, 1]],         [[1, -1]]),
    "Case 5 no information":   ([[0, 0]],         [[0, 0]]),
    "Case 6 both, no gain":    ([[1, 0], [0, 1]], [[1, 0], [0, 1]]),
}

fig, ax = plt.subplots(figsize=(8.5, 7.5))
lim = (-2.7, 1.4)
ax.plot(lim, lim, color="gray", ls="--", lw=1.5)
ax.axhline(0, color="lightgray", lw=1); ax.axvline(0, color="lightgray", lw=1)
print(f"{'category':<26}{'aligned':>9}{'S1':>8}{'S2':>8}")
for name, (M1, M2) in CATEGORIES.items():
    A, S1, S2 = shuffle_signature(M1, M2)
    dead = abs(A) < 0.1
    ax.plot(S1, S2, "o", ms=12, color="gray" if dead else "#534AB7", mec="k")
    ax.annotate(f"{name}\naligned={A:.2f}", (S1, S2), xytext=(10, 10),
                textcoords="offset points", fontsize=8.5)
    print(f"{name:<26}{A:>9.2f}{S1:>8.2f}{S2:>8.2f}")
ax.set_xlabel("S1  (reliance on O1)"); ax.set_ylabel("S2  (reliance on O2)")
ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal"); ax.grid(alpha=0.2)
ax.set_title("Shuffle plane — ground truth known")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Experiment 4 (optional, needs torch) — same comparison with the real MLP
#
# The linear analysis is exact for a linear readout; this checks the conclusions
# hold for the actual ReLU `SimpleMLP`. Skipped automatically when torch is not
# installed (e.g. running locally) — run on the cluster.

# %%
try:
    import sys, torch
    sys.path.append("../src")
    from models import SimpleMLP
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False
    print("torch not available — skipping the MLP check (run on the cluster)")

# %%
if HAVE_TORCH:
    def fit_mlp(Xtr, ytr, epochs=300, lr=1e-3, seed=0):
        torch.manual_seed(seed)
        net = SimpleMLP(Xtr.shape[1], [64, 32], 1, dropout_rate=0.0)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        Xt = torch.from_numpy(Xtr).float()
        yt = torch.from_numpy(ytr.reshape(-1, 1)).float()
        for _ in range(epochs):
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(net(Xt), yt)
            loss.backward()
            opt.step()
        net.eval()
        return net

    def run_mechanism_mlp(mech, nl, seed=0):
        g = np.random.default_rng(seed)
        z2_tr, z2_va = mech(x2, nl, g)
        # smaller subset — the MLP is slower than lstsq
        sub = slice(0, 8_000)
        net = fit_mlp(np.hstack([x1[sub], z2_tr[sub]]), theta[sub], seed=seed)
        with torch.no_grad():
            p = net(torch.from_numpy(np.hstack([x1[va], z2_va[va]])).float()).numpy().ravel()
        return r2(theta[va], p)

    print(f"{'noise':<8}" + "".join(f"{n:>12}" for n in MECHANISMS))
    for nl in [0, 1, 2.5, 5]:
        row = f"{nl:<8}"
        for name, (m, _, _) in MECHANISMS.items():
            row += f"{run_mechanism_mlp(m, nl):>12.3f}"
        print(row)
    print(f"\nlinear floor for reference: {floor:.3f}")

# %% [markdown]
# ## Adding a new mechanism
#
# 1. Write a function `(x, noise_level, rng) -> (train_version, val_version)`.
# 2. Add it to `MECHANISMS` with a color.
# 3. Re-run Experiment 1. If it should destroy O₂'s information, it must reach
#    the O₁-alone floor; if it plateaus above, validation is still feeding the
#    model usable signal from O₂.
