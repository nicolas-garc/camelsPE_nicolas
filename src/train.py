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
):
    train_losses, val_losses = [], []
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

        # Progress
        pbar.update(1)
        if epoch % 500 == 0:
            pbar.set_postfix(iter=epoch, loss=f"{train_loss:.4f}")
            pbar.write(f"[Iter {epoch:4d}] validation loss: {val_loss:.4f}")

    pbar.close()
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
):
    train_losses, val_losses = [], []
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

        pbar.update(1)
        if epoch % 500 == 0:
            pbar.set_postfix(iter=epoch, loss=f"{train_loss:.4f}")
            pbar.write(f"[Iter {epoch:4d}] validation loss: {val_loss:.4f}")

    pbar.close()
    return train_losses, val_losses

