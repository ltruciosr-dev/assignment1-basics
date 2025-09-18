## Transformer LM resource accounting

(a) Consider GPT-2 XL, which has the following configuration:

- vocab_size : 50,257
- context_length : 1,024
- num_layers : 48
- d_model : 1,600
- num_heads : 25
- d_ff : 6,400

**Parameter counting breakdown**:

1. **Embedding layer**: `vocab_size * d_model`
   - Token embedding matrix maps each vocabulary token to a d_model dimensional vector
   - Shape: [vocab_size, d_model] = [50,257, 1,600] = **80,411,200 parameters**

2. **Per transformer block** (repeated `num_layers` times):
   - **Layer normalization (ln1 + ln2)**: `2 * d_model`
     - RMSNorm has learned scale parameters for each dimension
     - 2 layer norms per block × 1,600 dimensions = **3,200 parameters per block**

   - **Multi-head attention**:
     - **Projection matrices**: `4 * d_model * d_model`
       - Q projection: [d_model, d_model] = [1,600, 1,600]
       - K projection: [d_model, d_model] = [1,600, 1,600]
       - V projection: [d_model, d_model] = [1,600, 1,600]
       - Output projection: [d_model, d_model] = [1,600, 1,600]
       - Total: **10,240,000 parameters per block**
     - **RoPE buffers**: `2 * (d_model // num_heads // 2) * context_length`
       - Pre-computed rotary embeddings for efficiency
       - Per head dimension: d_model // num_heads = 1,600 // 25 = 64
       - RoPE operates on half the dimensions: 64 // 2 = 32
       - Cos/sin buffers: 2 × 32 × 1,024 = **65,536 buffer elements per block**

   - **Feed-forward network (SwiGLU)**: `3 * d_ff * d_model`
     - W1 gate projection: [d_model, d_ff] = [1,600, 6,400]
     - W2 output projection: [d_ff, d_model] = [6,400, 1,600]
     - W3 value projection: [d_model, d_ff] = [1,600, 6,400]
     - Total: **30,720,000 parameters per block**

   **Total trainable parameters per block**: 3,200 + 10,240,000 + 30,720,000 = **40,963,200**
   **Total buffer elements per block**: **65,536**

3. **Final layer normalization**: `d_model`
   - RMSNorm scale parameters: **1,600 parameters**

4. **Language modeling head**: `d_model * vocab_size`
   - Linear projection from hidden states to vocabulary logits
   - Shape: [d_model, vocab_size] = [1,600, 50,257] = **80,411,200 parameters**

**Total trainable parameters**:
- Embedding: 80,411,200
- Transformer blocks: 48 × 40,963,200 = 1,966,233,600
- Final LayerNorm: 1,600
- LM head: 80,411,200
- **Grand total: 2,127,057,600 parameters (~2.13B)**

**Total memory buffers**: 48 × 65,536 = **3,145,728 elements**

(b) FLOP accounting for matrix multiplications:

For a forward pass with sequence length `context_length`, the matrix multiplication FLOPs are:

1. **Embedding lookup**: No matrix multiplication FLOPs (just indexing)

2. **Per transformer block** (repeated `num_layers` times):
   - **Attention projections**: 4 matrix multiplications
     - Q projection: `2 * context_length * d_model * d_model`
     - K projection: `2 * context_length * d_model * d_model`
     - V projection: `2 * context_length * d_model * d_model`
     - Output projection: `2 * context_length * d_model * d_model`
     - **Total per block**: `8 * context_length * d_model^2`

   - **Attention computation**:
     - QK^T: `2 * num_heads * context_length^2 * (d_model // num_heads)`
     - Attention weights × V: `2 * num_heads * context_length^2 * (d_model // num_heads)`
     - **Total per block**: `4 * context_length^2 * d_model`

   - **Feed-forward network (SwiGLU)**: 3 matrix multiplications
     - W1 projection: `2 * context_length * d_model * d_ff`
     - W3 projection: `2 * context_length * d_model * d_ff`
     - W2 projection: `2 * context_length * d_ff * d_model`
     - **Total per block**: `6 * context_length * d_model * d_ff`

3. **Final layer**:
   - LM head: `2 * context_length * d_model * vocab_size`

**Total FLOPs per forward pass**:
```
num_layers * (8 * context_length * d_model^2 + 4 * context_length^2 * d_model + 6 * context_length * d_model * d_ff) + 2 * context_length * d_model * vocab_size
```

For GPT-2 XL with context_length tokens:
- Attention projections: `48 * 8 * context_length * 1600^2 = 983,040,000 * context_length`
- Attention computation: `48 * 4 * context_length^2 * 1600 = 307,200 * context_length^2`
- Feed-forward: `48 * 6 * context_length * 1600 * 6400 = 2,949,120,000 * context_length`
- LM head: `2 * context_length * 1600 * 50257 = 160,825,600 * context_length`

**Total**: `4,092,985,600 * context_length + 307,200 * context_length^2` FLOPs
