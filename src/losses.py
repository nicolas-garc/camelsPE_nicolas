import torch
import torch.nn as nn
import torch.nn.functional as F


class MSELoss(nn.Module):
    """
    MSE + (lambda_w) * L2_weight_penalty + (lambda_o) * output_magnitude_penalty
    """
    def __init__(self):
        super().__init__()

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 1) base MSE
        mse = F.mse_loss(outputs, targets)

        #out_pen = torch.mean(outputs.pow(2))

        return mse