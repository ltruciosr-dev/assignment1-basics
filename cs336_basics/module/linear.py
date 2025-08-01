import numpy as np
import torch.nn as nn
import torch
from einops import einsum


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        """
        Construct a linear transformation module.

        Args:
            in_features: int final dimension of the input
            out_features: int final dimension of the output
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        self.factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.weight = self._init_weight(in_features, out_features)

    def _init_weight(self, in_features, out_features) -> nn.Parameter:
        """Initialize the weight."""
        weight = nn.Parameter(torch.empty((out_features, in_features), **self.factory_kwargs))
        std = np.sqrt(2 / (out_features + in_features))
        nn.init.trunc_normal_(weight, mean=0, std=std, a=-3 * std, b=3 * std)
        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
