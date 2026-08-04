import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm.auto import tqdm
from typing import Optional, Union, Callable


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
    Runs all `epochs` epochs. If restore_best_weights=True, snapshots the
    model when the EMA of val_loss (with span `best_weights_smoothing_window`)
    hits a new minimum, and loads that snapshot at the end. The EMA guards
    against picking a lucky one-epoch fluke. window=1 → no smoothing (raw
    per-epoch min). restore_best_weights=False → keep last-epoch weights.
    """
    train_losses, val_losses = [], []
    alpha = 2.0 / (best_weights_smoothing_window + 1)
    smoothed = None
    best_smoothed = float("inf")
    best_state = None
    best_epoch = -1
    pbar = tqdm(total=epochs, desc="Training", unit="iter")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if restore_best_weights:
            smoothed = val_loss if smoothed is None else alpha * val_loss + (1 - alpha) * smoothed
            if smoothed < best_smoothed:
                best_smoothed = smoothed
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch

        pbar.update(1)
        if epoch % 500 == 0:
            pbar.set_postfix(iter=epoch, loss=f"{train_loss:.4f}")
            pbar.write(f"[Iter {epoch:4d}] validation loss: {val_loss:.4f}")

    pbar.close()

    if restore_best_weights and best_state is not None:
        model.load_state_dict(best_state)
        print(f"[best-weights] restored epoch {best_epoch}  "
              f"raw val={val_losses[best_epoch-1]:.4f}  "
              f"ema (W={best_weights_smoothing_window})={best_smoothed:.4f}  "
              f"vs last-epoch={val_losses[-1]:.4f}")

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
    noise resampling). See fit() for restore_best_weights /
    best_weights_smoothing_window semantics.

    Val loader is passed once and unchanged (val stays CLEAN per pipeline
    design), so its measurements are directly comparable across epochs.
    """
    train_losses, val_losses = [], []
    alpha = 2.0 / (best_weights_smoothing_window + 1)
    smoothed = None
    best_smoothed = float("inf")
    best_state = None
    best_epoch = -1
    pbar = tqdm(total=epochs, desc="Training", unit="iter")

    for epoch in range(1, epochs + 1):
        current_loader = train_loader_fn() if train_loader_fn else train_loader
        train_loss = train_one_epoch(model, current_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if restore_best_weights:
            smoothed = val_loss if smoothed is None else alpha * val_loss + (1 - alpha) * smoothed
            if smoothed < best_smoothed:
                best_smoothed = smoothed
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch

        pbar.update(1)
        if epoch % 500 == 0:
            pbar.set_postfix(iter=epoch, loss=f"{train_loss:.4f}")
            pbar.write(f"[Iter {epoch:4d}] validation loss: {val_loss:.4f}")

    pbar.close()

    if restore_best_weights and best_state is not None:
        model.load_state_dict(best_state)
        print(f"[best-weights] restored epoch {best_epoch}  "
              f"raw val={val_losses[best_epoch-1]:.4f}  "
              f"ema (W={best_weights_smoothing_window})={best_smoothed:.4f}  "
              f"vs last-epoch={val_losses[-1]:.4f}")

    return train_losses, val_losses

