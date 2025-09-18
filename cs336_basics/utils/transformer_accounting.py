def count_trainable_parameters(
    vocab_size: int,
    num_layers: int,
    d_model: int,
    d_ff: int,
) -> int:
    """
    Count trainable parameters in TransformerLM model.

    Args:
        vocab_size: Size of vocabulary
        context_length: Maximum sequence length
        num_layers: Number of transformer blocks
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension

    Returns:
        Total number of trainable parameters
    """
    # Embedding
    embedding_params = vocab_size * d_model

    # Per transformer block parameters
    block_params = (
        2 * d_model  # Layer norms (2 per block)
        + 4 * d_model * d_model  # Attention projections (4 matrices)
        + 3 * d_ff * d_model  # FFN (3 matrices for SwiGLU)
    )

    # Final layer norm
    ln_final_params = d_model

    # LM head
    lm_head_params = d_model * vocab_size

    total_params = embedding_params + (num_layers * block_params) + ln_final_params + lm_head_params
    return total_params


def calculate_model_memory_mb(
    vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int
) -> float:
    """
    Calculate total memory required to load TransformerLM model including RoPE buffers.
    Assumes single precision (4 bytes per parameter).

    Args:
        vocab_size: Size of vocabulary
        context_length: Maximum sequence length
        num_layers: Number of transformer blocks
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension

    Returns:
        Total memory in MB
    """
    # Trainable parameters
    trainable_params = count_trainable_parameters(vocab_size, num_layers, d_model, d_ff)

    # RoPE buffers (non-trainable but stored in memory)
    d_head = d_model // num_heads
    rope_buffer_size_per_layer = 2 * context_length * (d_head // 2)
    total_rope_buffers = num_layers * rope_buffer_size_per_layer

    # Total parameters (trainable + buffers)
    total_memory_params = trainable_params + total_rope_buffers

    # Convert to MB (4 bytes per float32)
    memory_mb = (total_memory_params * 4) / (1024**2)

    return memory_mb


def calculate_forward_pass_flops(
    vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int
) -> int:
    """
    Calculate total FLOPs for a forward pass through TransformerLM.

    Args:
        vocab_size: Size of vocabulary
        context_length: Sequence length for forward pass
        num_layers: Number of transformer blocks
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension

    Returns:
        Total FLOPs for forward pass
    """
    # Per transformer block FLOPs
    attention_proj_flops = 8 * context_length * d_model * d_model
    attention_compute_flops = 4 * context_length * context_length * d_model
    ffn_flops = 6 * context_length * d_model * d_ff

    block_flops = attention_proj_flops + attention_compute_flops + ffn_flops

    # Final layer FLOPs
    lm_head_flops = 2 * context_length * d_model * vocab_size

    # Total FLOPs
    total_flops = num_layers * block_flops + lm_head_flops

    return total_flops


def main():
    # Parameters from transformer_accounting.md
    vocab_size = 50257
    context_length = 1024
    num_layers = 48
    d_model = 1600
    num_heads = 25
    d_ff = 6400

    n_parameters = count_trainable_parameters(vocab_size, num_layers, d_model, d_ff)
    print(f"Total number of parameters: {n_parameters / 1e9:.2f}B")

    memory_mb = calculate_model_memory_mb(vocab_size, context_length, num_layers, d_model, num_heads, d_ff)
    print(f"Total model memory (MB): {memory_mb:.2f}")

    flops = calculate_forward_pass_flops(vocab_size, context_length, num_layers, d_model, num_heads, d_ff)
    print(f"Forward pass FLOPs: {flops / 1e9:.2f}B")


if __name__ == "__main__":
    main()
