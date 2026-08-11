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

class VarMLP(nn.Module):
    def _init_(self, input_dim: int, hidden_dims: list, output_dim: int, dropout_rate):
        """
        input_dim   : size of input features (e.g. 28*28=784 for flattened MNIST)
        hidden_dims : list of ints, sizes of hidden layers
        output_dim  : number of classes (for classification) or 1 for regression
        """
        super()._init_()
        layers = []
        in_dim = input_dim
        #build hidden layers that use Leaky RELU
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.LeakyRelu())
            layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.Softplus())
            in_dim = h
        #final output 
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.net(x)
        
        

class HeteroscedasticMLP(nn.Module):
    """Heteroscedastic-Gaussian MLP: shared trunk with two heads (μ and log σ²).

    Predicts (μ, log σ²) per parameter — the Gaussian posterior parameters directly,
    following Kendall & Gal 2017. Trained with Gaussian NLL loss (with an MSE
    warmup phase in `pipeline.fit_hetero_case` to avoid σ-collapse at init).

    Output: two tensors [batch, output_dim] — mu and log_var.
    log_var is clamped to [-6, 6] during forward to prevent numerical blow-up
    (variance ∈ [~0.0025, ~403]).
    """
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate,
                 log_var_clamp=(-6.0, 6.0)):
        super().__init__()
        self.output_dim = output_dim
        self._lo, self._hi = log_var_clamp
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = h
        self.trunk = nn.Sequential(*layers)
        self.mu_head = nn.Linear(in_dim, output_dim)
        self.log_var_head = nn.Linear(in_dim, output_dim)
        # Bias log σ² toward a moderate initial value so NLL doesn't blow up
        # from a random tiny variance at step 0
        nn.init.constant_(self.log_var_head.bias, 0.0)

    def forward(self, x: torch.Tensor):
        h = self.trunk(x)
        mu = self.mu_head(h)
        log_var = torch.clamp(self.log_var_head(h), min=self._lo, max=self._hi)
        return mu, log_var
