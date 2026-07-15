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
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# %%
rng = np.random.default_rng(0)

n = 40_000
d = 15

X = rng.normal(size=(n, d))

# True non-linear function
w1 = rng.normal(size=d)
w2 = rng.normal(size=d)
quad = (X[:, :5] ** 2) @ rng.normal(size=5)
interaction = (X[:, 0] * X[:, 1]) * 0.8
nonlinear = np.tanh(X @ w1) + 0.5 * np.tanh(X @ w2)

y_clean = 2.0 + nonlinear + 0.3 * quad + interaction
y = y_clean + rng.normal(scale=1.0, size=n)  # observation noise
y = y.astype(np.float32)

# %%
#train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# %%
class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
    def forward(self, x):
        return self.net(x)


# %%
device = "cpu"
model = MLP(d).to(device)
X_train_t = torch.from_numpy(X_train).float().to(device)
y_train_t = torch.from_numpy(y_train).float().view(-1, 1).to(device)
X_test_t = torch.from_numpy(X_test).float().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# %%
batch_size = 256
num_epochs = 40

for epoch in range(num_epochs):
    model.train()
    perm = rng.permutation(len(X_train_t))
    X_train_epoch = X_train_t[perm]
    y_train_epoch = y_train_t[perm]

    for i in range(0, len(X_train_epoch), batch_size):
        xb = X_train_epoch[i:i+batch_size]
        yb = y_train_epoch[i:i+batch_size]

        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

# %%
model.eval()
with torch.no_grad():
    y_hat_test = model(X_test_t).cpu().numpy().ravel()

# Real R^2
r2_real = r2_score(y_test, y_hat_test)

# Shuffled "ground truth" R^2
y_test_shuffled = rng.permutation(y_test)
r2_shuffled = r2_score(y_test_shuffled, y_hat_test)


# %%
def var(a):
    return np.var(a, ddof=0)

def cov(a, b):
    return np.mean((a - a.mean()) * (b - b.mean()))

var_y = var(y_test)
cov_y_yhat = cov(y_test, y_hat_test)

lhs = r2_real - r2_shuffled
rhs = 2 * cov_y_yhat / var_y

print(f"R2_real      : {r2_real:.4f}")
print(f"R2_shuffled  : {r2_shuffled:.4f}")
print(f"Gap (real - shuffled): {lhs:.4f}")
print(f"2 * Cov(y, y_hat)/Var(y): {rhs:.4f}")

# Extra: see how negative the shuffled R^2 is
print(f"Var(y_hat)/Var(y): {var(y_hat_test)/var_y:.4f}")


print(r)

# %%


rng = np.random.default_rng(0)

# 1. Two-input nonlinear data ----------------------------------------
n = 40_000
X = rng.normal(size=(n, 2))
x1, x2 = X[:, 0], X[:, 1]

g = (
    1.5 * np.tanh(1.2 * x1 + 0.8 * x2)
    - 0.7 * np.tanh(-0.5 * x1 + 1.3 * x2)
    + 0.4 * x1 * x2
    + 0.2 * x1**2
    - 0.1 * x2**2
)

y = (g + rng.normal(scale=0.5, size=n)).astype(np.float32)

y_bar = y.mean()
f = 0.6
y_shrunk = y_bar + f * (y - y_bar)

# Train/test split: train on shrunk, evaluate vs true
X_train, X_test, y_train_shrunk, y_test_shrunk, y_train_true, y_test_true = train_test_split(
    X, y_shrunk, y, test_size=0.3, random_state=42
)

X_train_t = torch.from_numpy(X_train).float()
y_train_t = torch.from_numpy(y_train_shrunk).float().view(-1, 1)
X_test_t = torch.from_numpy(X_test).float()

# 2. Small NN --------------------------------------------------------
class MLP2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
    def forward(self, x):
        return self.net(x)

model = MLP2D()
opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
crit = nn.MSELoss()

# 3. Train on shrunk targets ----------------------------------------
batch_size = 256
epochs = 40

for epoch in range(epochs):
    model.train()
    idx = rng.permutation(len(X_train_t))
    Xb = X_train_t[idx]
    yb = y_train_t[idx]

    for i in range(0, len(Xb), batch_size):
        xb = Xb[i:i+batch_size]
        ybatch = yb[i:i+batch_size]

        opt.zero_grad()
        pred = model(xb)
        loss = crit(pred, ybatch)
        loss.backward()
        opt.step()

# 4. Baseline R^2 on clean test -------------------------------------
model.eval()
with torch.no_grad():
    y_hat_test = model(X_test_t).numpy().ravel()

r2_base = r2_score(y_test_true, y_hat_test)
print("Baseline R2 (no shuffle):", r2_base)


# %%
# Make a copy of X_test
X_test_x1_shuf = X_test.copy()
X_test_x1_shuf[:, 0] = rng.permutation(X_test_x1_shuf[:, 0])  # shuffle first feature

with torch.no_grad():
    y_hat_x1_shuf = model(torch.from_numpy(X_test_x1_shuf).float()).numpy().ravel()

r2_x1_shuf = r2_score(y_test_true, y_hat_x1_shuf)
print("R2 after shuffling x1 only:", r2_x1_shuf)
print("Drop from baseline (x1):   ", r2_base - r2_x1_shuf)


# %%
X_test_x2_shuf = X_test.copy()
X_test_x2_shuf[:, 1] = rng.permutation(X_test_x2_shuf[:, 1])  # shuffle second feature

with torch.no_grad():
    y_hat_x2_shuf = model(torch.from_numpy(X_test_x2_shuf).float()).numpy().ravel()

r2_x2_shuf = r2_score(y_test_true, y_hat_x2_shuf)
print("R2 after shuffling x2 only:", r2_x2_shuf)
print("Drop from baseline (x2):   ", r2_base - r2_x2_shuf)


# %%
