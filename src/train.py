import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm.auto import tqdm
from typing import Optional, Union, Callable
from collections import deque


class _BestWeightsTracker:
    """Rolling-window best-weights tracker for jitter-resistant restoration.

    Runs alongside a training loop: call `update(epoch, val_loss, model)`
    every epoch, then `restore(model)` at the end.

    Chooses the "best" epoch as follows:
      - Maintain a rolling window of the last `window` (val_loss, state_dict).
      - Compute smoothed val = mean over that window (jitter-resistant).
      - When smoothed val hits a new minimum, snapshot the SINGLE epoch inside
        that window whose raw val_loss was lowest.

    So the smoothed-mean gates *which window* counts as "best," and inside
    the best window we still take the single-epoch minimum weights (not an
    average of weights, which would change model behaviour). window=1 collapses
    to plain raw-min single-epoch selection.
    """

    def __init__(self, window: int):
        self.W = max(1, int(window))
        self.val_window = deque(maxlen=self.W)
        self.state_window = deque(maxlen=self.W)
        self.best_smoothed = float("inf")
        self.best_epoch = -1
        self.best_raw_val = float("inf")
        self.best_state = None

    def update(self, epoch: int, val_loss: float, model: nn.Module):
        self.val_window.append(float(val_loss))
        # Detach + clone on-device; small model → negligible cost.
        self.state_window.append({k: v.detach().clone() for k, v in model.state_dict().items()})
        if len(self.val_window) == self.W:
            smoothed = sum(self.val_window) / self.W
            if smoothed < self.best_smoothed:
                self.best_smoothed = smoothed
                # argmin raw val_loss WITHIN the best-smoothed window
                argmin = min(range(self.W), key=lambda i: self.val_window[i])
                self.best_state = self.state_window[argmin]
                self.best_epoch = epoch - (self.W - 1 - argmin)
                self.best_raw_val = self.val_window[argmin]

    def restore(self, model: nn.Module) -> Optional[int]:
        """Load best_state into model. Returns best_epoch (or None if never set)."""
        if self.best_state is None:
            return None
        model.load_state_dict(self.best_state)
        return self.best_epoch


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        if x.ndim > 2:
            x = x.view(x.size(0), -1)
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)




def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            if x.ndim > 2:
                x = x.view(x.size(0), -1)
            x, y = x.to(device), y.to(device)
            preds = model(x)
            total_loss += criterion(preds, y).item()
    return total_loss / len(loader)


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epochs: int = 10,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    restore_best_weights: bool = True,
    best_weights_smoothing_window: int = 20,
):
    """
    Runs all `epochs` epochs. Val is checked each epoch.

    restore_best_weights (default True): at the end of training, load model
    weights from the epoch that had the lowest raw val_loss inside the
    window with the lowest smoothed (rolling-mean) val_loss. Guards against
    picking a lucky one-epoch fluke while still using the actual best-epoch
    weights (not a weight-space average). window=1 → raw single-epoch min.
    Set restore_best_weights=False to keep last-epoch weights (old behaviour).
    """
    train_losses, val_losses = [], []
    tracker = _BestWeightsTracker(best_weights_smoothing_window) if restore_best_weights else None
    pbar = tqdm(total=epochs, desc="Training", unit="iter")

    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)

        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        # Step scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if tracker is not None:
            tracker.update(epoch, val_loss, model)

        # Progress
        pbar.update(1)
        if epoch % 500 == 0:
            pbar.set_postfix(iter=epoch, loss=f"{train_loss:.4f}")
            pbar.write(f"[Iter {epoch:4d}] validation loss: {val_loss:.4f}")

    pbar.close()

    if tracker is not None:
        best_epoch = tracker.restore(model)
        if best_epoch is not None:
            print(f"[best-weights] restored epoch {best_epoch}  "
                  f"raw val_loss={tracker.best_raw_val:.4f}  "
                  f"smoothed val (W={tracker.W})={tracker.best_smoothed:.4f}  "
                  f"vs last-epoch val_loss={val_losses[-1]:.4f}")

    return train_losses, val_losses


def fit_with_epoch_noise(
    model: nn.Module,
    train_loader: Union[DataLoader, None],
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epochs: int = 10,
    train_loader_fn: Optional[Callable[[], DataLoader]] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    restore_best_weights: bool = True,
    best_weights_smoothing_window: int = 20,
):
    """
    Same as fit() but re-generates the train loader each epoch (for per-epoch
    noise resampling). See fit() docstring for restore_best_weights /
    best_weights_smoothing_window semantics -- both do rolling-window smoothed
    val loss to pick the best-epoch weights and restore them at the end.

    Val loader is passed once and unchanged (val stays CLEAN per pipeline
    design), so its measurements are directly comparable across epochs.
    """
    train_losses, val_losses = [], []
    tracker = _BestWeightsTracker(best_weights_smoothing_window) if restore_best_weights else None
    pbar = tqdm(total=epochs, desc="Training", unit="iter")

    for epoch in range(1, epochs + 1):
        # Dynamically use loader function (for noise) if provided
        current_loader = train_loader_fn() if train_loader_fn else train_loader

        train_loss = train_one_epoch(
            model=model,
            loader=current_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device
        )
        train_losses.append(train_loss)

        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if tracker is not None:
            tracker.update(epoch, val_loss, model)

        pbar.update(1)
        if epoch % 500 == 0:
            pbar.set_postfix(iter=epoch, loss=f"{train_loss:.4f}")
            pbar.write(f"[Iter {epoch:4d}] validation loss: {val_loss:.4f}")

    pbar.close()

    if tracker is not None:
        best_epoch = tracker.restore(model)
        if best_epoch is not None:
            print(f"[best-weights] restored epoch {best_epoch}  "
                  f"raw val_loss={tracker.best_raw_val:.4f}  "
                  f"smoothed val (W={tracker.W})={tracker.best_smoothed:.4f}  "
                  f"vs last-epoch val_loss={val_losses[-1]:.4f}")

    return train_losses, val_losses

