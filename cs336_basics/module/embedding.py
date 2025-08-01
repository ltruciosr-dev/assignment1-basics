import torch.nn as nn
import torch


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """
        Construct an embedding module.

        Args:
            num_embeddings: int Size of the vocabulary
            embedding_dim: int Dimension of the embedding vectors, i.e., dmodel
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        self.factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.weight = self._init_weight(num_embeddings, embedding_dim)

    def _init_weight(self, num_embeddings, embedding_dim) -> nn.Parameter:
        """Initialize the weight."""
        weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), **self.factory_kwargs))
        nn.init.trunc_normal_(weight, mean=0, std=1, a=-3, b=3)
        return weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
