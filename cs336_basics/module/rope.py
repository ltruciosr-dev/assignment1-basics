import torch.nn as nn
import torch
from einops import einsum


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, dtype=None):
        """
        Construct the relative positional embeddings that help us to ingest positional information.

        Args:
            theta: float Θ value for the RoPE
            d_k: int dimension of query and key vectors
            max_seq_len: int Maximum sequence length that will be inputted
            device: torch.device | None = None Device to store the buffer on
        """
        self.factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        position = torch.arange(max_seq_len)
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2) / d_k))
        angles = einsum(position, inv_freq, "d_t, d_k -> d_t d_k")

        # cache the cos and sin pseudo-matrix
        self.register_buffer("cos_cached", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cached", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """
        Return the transformed tensor

        Args:
            x (Float[Tensor, "... sequence_length d_k"]): Input tensor for either the key or query tokens.
            token_positions (Int[Tensor, "... sequence_length"]): Token positions along the sequence dimension.

        Return:
            (Float[Tensor, "... sequence_length d_k"]): Output tensor for rotational positional embeddings.
        """
        if token_positions is None:
            cos = self.cos_cached
            sin = self.sin_cached
        else:
            cos = self.cos_cached[token_positions]  # (sequence_length d_k//2)
            sin = self.sin_cached[token_positions]  # (sequence_length d_k//2)

        # Compute the rotated positional embeddings
        x_even = x[..., 0::2]  # (... d_k//2)
        x_odd = x[..., 1::2]  # (... d_k//2)
        x_even_rotated = x_even * cos - x_odd * sin
        x_odd_rotated = x_even * sin + x_odd * cos

        return torch.stack([x_even_rotated, x_odd_rotated], dim=-1).flatten(-2)
