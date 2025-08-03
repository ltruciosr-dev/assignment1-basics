import numpy as np
import torch.nn as nn
import torch
from einops import einsum


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        """
        Construct a SwiGLU transformation module.

        Args:
            in_features: int final dimension of the input
            out_features: int final dimension of the output
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        self.factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.w1_weight = self._init_weight(d_ff, d_model)
        self.w2_weight = self._init_weight(d_model, d_ff)
        self.w3_weight = self._init_weight(d_ff, d_model)

    def _init_weight(self, out_features, in_features) -> nn.Parameter:
        """Initialize the weight."""
        weight = nn.Parameter(torch.empty((out_features, in_features), **self.factory_kwargs))
        std = np.sqrt(2 / (out_features + in_features))
        nn.init.trunc_normal_(weight, mean=0, std=std, a=-3 * std, b=3 * std)
        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_x = einsum(x, self.w1_weight, "... d_model, d_ff d_model -> ... d_ff")
        w3_x = einsum(x, self.w3_weight, "... d_model, d_ff d_model -> ... d_ff")
        gate = w1_x * torch.sigmoid(w1_x)
        inner = einsum(gate, w3_x, "... d_ff, ... d_ff -> ... d_ff")
        out = einsum(inner, self.w2_weight, "... d_ff, d_model d_ff -> ... d_model")
        return out
