import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TimeSelectionLayer(nn.Module):
    """
    Custom PyTorch layer for time selection.

    Args:
        num_outputs (int): Number of output units.
        regularization (float, optional): Regularization strength. Defaults to 0.001.
    """
    def __init__(self, input_shape: tuple, num_outputs: int, regularization: float = 0.001, flatten=False):
        super(TimeSelectionLayer, self).__init__()
        self.num_outputs = num_outputs
        self.regularization = regularization
        self.flatten = flatten
        self.mask = nn.Parameter(torch.full(input_shape, 0.01))
    
    def regularization_loss(self) -> torch.Tensor:
        """
        Custom regularization function for the layer.

        Args:
            weights (torch.Tensor): Layer weights.

        Returns:
            torch.Tensor: Regularization term.
        """
        weight = self.regularization / (10**math.log2(self.num_outputs))
        return torch.sum(weight * self.get_mask())
        
    def get_mask(self):

        mask = F.hardsigmoid(self.mask)
        rounded = torch.round(mask)
        
        return mask + (rounded - mask).detach()
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        
        inputs_masked = inputs * self.get_mask()

        if self.flatten:
            inputs_masked = inputs_masked.view(inputs_masked.size(0), -1)
        
        return inputs_masked