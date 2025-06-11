import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int, dropout_rate):
        """
        input_dim   : size of input features (e.g. 28*28=784 for flattened MNIST)
        hidden_dims : list of ints, sizes of hidden layers
        output_dim  : number of classes (for classification) or 1 for regression
        """
        super().__init__()
        layers = []
        in_dim = input_dim
        # build hidden layers
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = h
        # final output layer (no activation here if using CrossEntropyLoss)
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)