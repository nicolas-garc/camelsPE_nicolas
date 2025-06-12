import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
base_path = "src/" ## this should be adjusted depending on where you are running this.
sys.path.append(base_path)
from models import *
from train import *
from losses import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # Hyperparameters
    input_dim    = 28 * 28
    hidden_dims  = [128, 64]
    output_dim   = 1
    dropout_rate = 0.2
    lr           = 1e-3
    epochs       = 100
    val_fraction = 0.2 
    batch_size   = 64


    # Create a single TensorDataset, then split inside `fit()`
    x = torch.randn(1000, input_dim)
    y = torch.randn(1000, output_dim)
    full_dataset = TensorDataset(x, y)

    # -- split into train / val --
    n_val   = int(len(full_dataset) * val_fraction)
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    # model, optimizer, loss function
    model     = SimpleMLP(input_dim, hidden_dims, output_dim, dropout_rate).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = MSELoss()

    fit(model, train_loader, val_loader, optimizer, criterion, device, epochs)

