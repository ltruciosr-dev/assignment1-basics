import torch
from jaxtyping import Float
from torch import nn
from cs336_basics.module.attention import MultiHeadAttentionRoPE
from cs336_basics.module.positionwise_feedforward import SwiGLU
from cs336_basics.module.norm import RMSNorm
from cs336_basics.module.embedding import Embedding
from cs336_basics.module.linear import Linear


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
        device: str | torch.device | None = None,
        dtype: str | torch.dtype | None = None,
    ):
        """
        Basic Transformer language model using multi-head self-attention and feedforward layers.

        Args:
            vocab_size (int): Size of the vocabulary.
            context_length (int): Maximum sequence length (context window).
            d_model (int): The dimensionality of the Transformer block input and output.
            num_layers (int): Number of Transformer blocks (layers).
            num_heads (int): Number of attention heads. Must divide d_model evenly.
            d_ff (int): Dimensionality of the feed-forward inner layer.
            rope_theta (float, optional): RoPE base parameter (default: 10000.0). Used for pre-buffering
                the rotary positional embeddings.
            device (str or torch.device, optional): Device to store the parameters on.
            dtype (str or torch.dtype, optional): Data type of the parameters.
        """
        self.factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.tf_blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, **self.factory_kwargs)
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(
        self, seq: Float[torch.Tensor, "batch sequence_length"]
    ) -> Float[torch.Tensor, "batch sequence_length vocab_size"]:
        """
        Forward pass for a pre-norm Transformer block.

        Args:
            x (Float[Tensor, "batch sequence_length d_model"]): Input tensor.

        Returns:
            Float[Tensor, "batch sequence_length d_model"]: Output tensor after transformer block.
        """
        # Embedding
        x = self.embedding(seq)

        # Iterate over all transformer blocks
        for i in range(len(self.tf_blocks)):
            x = self.tf_blocks[i](x)

        # Final Norm + Linear
        x = self.ln_final(x)  # [B, S, D]
        x = self.lm_head(x)  # [B, S, vocab_size]

        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int | None = 128,
        theta: float | None = 10000.0,
        device: str | None = None,
        dtype: str | None = None,
    ):
        """
        Args:
            d_model (int): The dimensionality of the Transformer block input and output.
            num_heads (int): Number of attention heads. Must divide d_model evenly.
            d_ff (int): Dimensionality of the feed-forward inner layer.
            max_seq_len (int, optional): Maximum sequence length for pre-buffering RoPE transformations.
                This is required to precompute and cache the rotary positional embeddings.
            theta (float, optional): RoPE base parameter (default: 10000.0). Used for pre-buffering
                the rotary positional embeddings.
            device (str or torch.device, optional): Device to store the parameters on.
            dtype (str or torch.dtype, optional): Data type of the parameters.
        """
        self.factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.attn = MultiHeadAttentionRoPE(d_model, num_heads, theta, max_seq_len, **self.factory_kwargs)
        self.ffn = SwiGLU(d_model, d_ff, **self.factory_kwargs)

    def forward(self, x: Float[torch.Tensor, "batch sequence_length d_model"]) -> torch.Tensor:
        """
        Forward pass for a pre-norm Transformer block.

        Args:
            x (Float[Tensor, "batch sequence_length d_model"]): Input tensor.

        Returns:
            Float[Tensor, "batch sequence_length d_model"]: Output tensor after transformer block.
        """
        # Pre-norm and self-attention
        x_residual = x
        x = self.ln1(x)
        seq_len = x.shape[1]
        token_positions = torch.arange(seq_len, **self.factory_kwargs)
        x = x_residual + self.attn(x, token_positions)

        # Pre-norm and feedforward
        x_residual2 = x
        x = self.ln2(x)
        x = x_residual2 + self.ffn(x)

        return x
