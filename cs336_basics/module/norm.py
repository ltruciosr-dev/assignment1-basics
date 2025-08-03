import torch
import torch.nn as nn
from einops import einsum


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        Construct the RMSNorm layer normalization module.

        Args:
            d_model: int Hidden dimension of the model
            eps: float = 1e-5 Epsilon value for numerical stability
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        self.factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gain = nn.Parameter(torch.empty(d_model, **self.factory_kwargs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(einsum(x, x, "... d_model, ... d_model -> ...") / self.d_model + self.eps)
        rms_2 = einsum(x, 1 / rms, "... d_model, ... -> ... d_model")
        rms_3 = einsum(rms_2, self.gain, "... d_model, d_model -> ... d_model")
        return rms_3.to(in_dtype)
