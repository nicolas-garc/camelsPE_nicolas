"""Training loop for the noise sweep — device-resident, no DataLoader.

Inputs live on the device for the whole run. Each epoch resamples the input
noise directly on the device and iterates minibatches by slicing a fresh
permutation. For MLPs this small, the old per-epoch DataLoader rebuild plus
per-batch host->device copy cost more than the forward/backward pass itself.

Noise is added to the already-normalized observable and NOT renormalized, and
validation always uses clean inputs — see CLAUDE.md. Train loss (noisy input)
is therefore not comparable to val loss (clean input).
"""
import torch


def fit_with_epoch_noise(model, x_train, y_train, x_val, y_val, noise_std,
                         optimizer, criterion, epochs, batch_size,
                         restore_best_weights=True,
                         best_weights_smoothing_window=50,
                         log_every=500):
    """Train one case. Returns (train_losses, val_losses), one entry per epoch.

    x_train/x_val: (n, n_features) on device, normalized, clean.
    noise_std:     (n_features,) on device — per-column noise level, so one
                   observable can be noisy while the other stays clean.
    restore_best_weights: keep the weights at the best EMA-smoothed val loss
                   (window = best_weights_smoothing_window) rather than the
                   last epoch's.
    """
    n = x_train.shape[0]
    noisy = bool(noise_std.any().item())
    alpha = 2.0 / (best_weights_smoothing_window + 1)
    smoothed, best_smoothed, best_state, best_epoch = None, float("inf"), None, -1
    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        # fresh noise draw + fresh minibatch order every epoch
        x_epoch = x_train + torch.randn_like(x_train) * noise_std if noisy else x_train
        order = torch.randperm(n, device=x_train.device)
        total = 0.0
        for i in range(0, n, batch_size):
            sel = order[i:i + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x_epoch[sel]), y_train[sel])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(sel)
        train_losses.append(total / n)

        model.eval()
        with torch.no_grad():
            val_losses.append(criterion(model(x_val), y_val).item())

        if restore_best_weights:
            v = val_losses[-1]
            smoothed = v if smoothed is None else alpha * v + (1 - alpha) * smoothed
            if smoothed < best_smoothed:
                best_smoothed, best_epoch = smoothed, epoch
                best_state = {k: v_.detach().clone() for k, v_ in model.state_dict().items()}

        if log_every and epoch % log_every == 0:
            print(f"        epoch {epoch:5d}  train={train_losses[-1]:.4f}  "
                  f"val={val_losses[-1]:.4f}", flush=True)

    if restore_best_weights and best_state is not None:
        model.load_state_dict(best_state)
        print(f"        best-weights: epoch {best_epoch}  raw val={val_losses[best_epoch-1]:.4f}  "
              f"ema={best_smoothed:.4f}  vs last={val_losses[-1]:.4f}", flush=True)

    return train_losses, val_losses
