import torch


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
