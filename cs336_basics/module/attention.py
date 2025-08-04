import numpy as np
import torch.nn as nn
import torch
from jaxtyping import Float, Int
from einops import einsum
from cs336_basics.module.rope import RoPE
from cs336_basics.module.utils import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None):
        """
        Construct a causal multi-head self-attention as defined in https://arxiv.org/abs/1706.03762.

        Args:
            d_model (int): Final dimension of the input and output.
            num_heads (int): Number of attention heads.
            device (torch.device | None, optional): Device to store the parameters on.
            dtype (torch.dtype | None, optional): Data type of the parameters.
        """
        self.factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_heads = num_heads
        self.d_k = self.d_v = d_model
        self.d_h = d_model // num_heads
        self.q_proj_weight = self._init_weight(self.d_k, d_model, num_heads)  # (d_k, d_in)
        self.k_proj_weight = self._init_weight(self.d_k, d_model, num_heads)  # (d_k, d_in)
        self.v_proj_weight = self._init_weight(self.d_v, d_model, num_heads)  # (d_v, d_in)
        self.o_proj_weight = self._init_weight(d_model, self.d_v, 1)  # (d_in, d_v)

    def _init_weight(self, in_features, out_features, num_heads) -> nn.Parameter:
        """Initialize the weight."""
        weight = nn.Parameter(torch.empty((out_features, in_features), **self.factory_kwargs))
        std = np.sqrt(2 / (out_features / num_heads + in_features))
        nn.init.trunc_normal_(weight, mean=0, std=std, a=-3 * std, b=3 * std)
        return weight

    def forward(self, x: Float[torch.Tensor, "batch sequence_length d_in"]) -> torch.Tensor:
        # Direct reshape to multi-head format
        q = einsum(x, self.q_proj_weight, "... d_in, d_k d_in -> ... d_k").view(*x.shape[:-1], self.num_heads, self.d_h)
        k = einsum(x, self.k_proj_weight, "... d_in, d_k d_in -> ... d_k").view(*x.shape[:-1], self.num_heads, self.d_h)
        v = einsum(x, self.v_proj_weight, "... d_in, d_v d_in -> ... d_v").view(*x.shape[:-1], self.num_heads, self.d_h)

        # compute the mask
        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len, **self.factory_kwargs))

        # apply dot product transformation
        att_out = (
            scaled_dot_product_attention(
                Q=q.permute(2, 0, 1, 3), K=k.permute(2, 0, 1, 3), V=v.permute(2, 0, 1, 3), mask=mask
            )
            .permute(1, 2, 0, 3)
            .contiguous()
        )  # [B, seq_len, num_heads, d_h]
        output = einsum(
            att_out.view(*att_out.shape[:-2], self.num_heads * self.d_h),
            self.o_proj_weight,
            "... d_v, d_in d_v -> ... d_in",
        )
        return output


class MultiHeadAttentionRoPE(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: float, max_seq_len: int, device=None, dtype=None):
        """
        Construct a causal multi-head self-attention module with Rotary Positional Embeddings (RoPE)
        as described in https://arxiv.org/abs/1706.03762.

        Args:
            d_model (int): Input and output feature dimension of the model.
            num_heads (int): Number of attention heads.
            theta (float): RoPE base parameter (typically 10000.0).
            max_seq_len (int): Maximum sequence length for pre-caching RoPE.
            device (torch.device or None, optional): Device to store the parameters on.
            dtype (torch.dtype or None, optional): Data type of the parameters.
        """
        self.factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_heads = num_heads
        self.d_k = self.d_v = d_model
        self.d_h = d_model // num_heads
        self.q_proj_weight = self._init_weight(self.d_k, d_model, num_heads)  # (d_k, d_in)
        self.k_proj_weight = self._init_weight(self.d_k, d_model, num_heads)  # (d_k, d_in)
        self.v_proj_weight = self._init_weight(self.d_v, d_model, num_heads)  # (d_v, d_in)
        self.o_proj_weight = self._init_weight(d_model, self.d_v, 1)  # (d_in, d_v)
        self.rope = RoPE(theta, self.d_h, max_seq_len, **self.factory_kwargs)

    def _init_weight(self, in_features, out_features, num_heads) -> nn.Parameter:
        """Initialize the weight."""
        weight = nn.Parameter(torch.empty((out_features, in_features), **self.factory_kwargs))
        std = np.sqrt(2 / (out_features / num_heads + in_features))
        nn.init.trunc_normal_(weight, mean=0, std=std, a=-3 * std, b=3 * std)
        return weight

    def forward(
        self,
        x: Float[torch.Tensor, "... sequence_length d_in"],
        token_positions: Int[torch.Tensor, "... sequence_length"],
    ) -> torch.Tensor:
        # Direct reshape to multi-head format
        q = einsum(x, self.q_proj_weight, "... d_in, d_k d_in -> ... d_k").view(*x.shape[:-1], self.num_heads, self.d_h)
        k = einsum(x, self.k_proj_weight, "... d_in, d_k d_in -> ... d_k").view(*x.shape[:-1], self.num_heads, self.d_h)
        v = einsum(x, self.v_proj_weight, "... d_in, d_v d_in -> ... d_v").view(*x.shape[:-1], self.num_heads, self.d_h)

        # compute the mask
        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len, **self.factory_kwargs))

        # apply rope transformation to q, k tensors
        q_rope = self.rope(q.permute(2, 0, 1, 3), token_positions)
        k_rope = self.rope(k.permute(2, 0, 1, 3), token_positions)

        # apply dot product transformation
        att_out = (
            scaled_dot_product_attention(Q=q_rope, K=k_rope, V=v.permute(2, 0, 1, 3), mask=mask)
            .permute(1, 2, 0, 3)
            .contiguous()
        )  # [B, seq_len, num_heads, d_h]
        output = einsum(
            att_out.view(*att_out.shape[:-2], self.num_heads * self.d_h),
            self.o_proj_weight,
            "... d_v, d_in d_v -> ... d_in",
        )
        return output
