import math
import torch

from jaxtyping import Float
from torch import Tensor
from einops import einsum


def apply_softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Apply the softmax operation on the defined dimension of the tensor.

    For numerical stability we compute the substract by the max value on the desired dimension.

    Args:
        x (torch.Tensor): input tensor to transform.
        dim (int): dimension along which to apply the normalization.

    Returns:
        torch.Tensor: tensor after applying softmax along the specified dimension.
    """
    x_max, _ = torch.max(x, dim=dim, keepdim=True)
    exp = torch.exp(x - x_max)
    sum_exp = torch.sum(exp, dim=dim, keepdim=True)
    softmax = exp / sum_exp

    return softmax


def scaled_dot_product_attention(
    Q: Float[Tensor, "... d_in_q d_k"],
    K: Float[Tensor, "... d_in_v d_k"],
    V: Float[Tensor, "... d_in_v d_v"],
    mask: Float[Tensor, "... d_in_q d_in_v"] | None = None,
) -> Float[Tensor, "... d_in d_v"]:
    d_k = Q.shape[-1]
    att_scores = einsum(Q, K, "... d_in_q d_k, ... d_in_v d_k -> ... d_in_q d_in_v") / (math.sqrt(d_k))  # QK/sqrt(d_k)
    if mask is not None:
        masked_scores = att_scores.masked_fill(mask == 0, value=-float("inf"))
    att_weights = apply_softmax(masked_scores, dim=-1)
    return einsum(att_weights, V, "... d_in_q d_in_v, ... d_in_v d_v -> ... d_in_q d_v")
