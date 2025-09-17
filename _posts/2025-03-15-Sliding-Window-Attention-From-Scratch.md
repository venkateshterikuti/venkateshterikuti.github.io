---
layout: single
classes: wide
author_profile: true
title: "Sliding-Window Attention From Scratch: Taming Quadratic Costs for Longer Sequences"
seo_title: "Sliding-Window Attention implementation from scratch - PyTorch, transformer optimization, and memory efficiency"
published: true
---

TL;DR: *"Attention is all you need"*...until the sequence length doubles and your GPU fans sound like a jet engine. This post documents my journey building sliding-window attention (SWA) from scratch—a practical alternative that keeps the core expressiveness of attention while taming its quadratic memory and compute costs. Perfect for running longer sequences on everyday hardware like a MacBook Air.

---

## What I Built (at a glance)

- A vectorized sliding-window attention kernel with configurable window sizes and causal masking
- Comparative benchmarking framework showing SWA vs full attention across sequence lengths
- A complete transformer implementation with SWA integrated into the attention mechanism
- Training pipeline with Metal (MPS) acceleration for Apple Silicon
- Hands-on demo training a character-level language model on Edith Wharton's *The Verdict*

![Full attention vs sliding-window attention comparison](/assets/images/swa-blog/full vs swa.png)

## Why Sliding-Window Attention?

Full self-attention is wonderfully expressive—every token can attend to every other token in the sequence. But this flexibility comes at a steep cost: memory and compute requirements grow as O(N²) with sequence length N. For a 2048-token sequence, you're looking at 4 million attention scores per head.

Sliding-window attention offers a compelling trade-off:
- **Linear scaling**: Cost becomes O(N·W) where W is the window size
- **Local context preservation**: Most language patterns are local anyway
- **Hardware friendly**: Fits comfortably on consumer GPUs and laptops
- **Configurable trade-offs**: Adjust window size based on your sequence length and hardware constraints

The key insight: many language modeling tasks care more about nearby context than distant dependencies. SWA keeps the gossip local while maintaining the attention mechanism's core strengths.

---

## The Mathematics Behind Sliding-Window Attention

Standard attention computes:
```
Attention(Q,K,V) = softmax(QK^T / √d)V
```

For a sequence of length N, this creates an N×N attention matrix—expensive for long sequences.

Sliding-window attention constrains each query to only attend to keys within a window of size W:
- **Causal window**: Each token attends to the previous W-1 tokens and itself
- **Bidirectional window**: Each token attends to W/2 tokens on each side

This reduces the attention matrix from N×N to N×W, achieving linear scaling in sequence length.

![Transformer blocks showing fixed vs sliding window](/assets/images/swa-blog/Two-consecutive-transformer-blocks-the-left-is-a-fixed-window-the-right-is-a-sliding.png)

---

## Building the Sliding-Window Kernel: Getting the Indices Right

The trickiest part of implementing SWA is efficiently gathering the right keys and values for each query position. I solved this with vectorized index tensors:

```python
@torch.no_grad()
def _make_indices(n: int, window_size: int, causal: bool, device: torch.device):
    """Generate indices for sliding window attention."""
    w = min(window_size, n)
    
    # Create offset pattern based on causality
    if causal:
        # Causal: look back at previous W-1 tokens + current token
        offsets = torch.arange(-w + 1, 1, device=device)
    else:
        # Bidirectional: W/2 tokens on each side
        offsets = torch.arange(-w // 2, w // 2 + 1, device=device)
    
    # Broadcast to create index matrix
    base = torch.arange(n, device=device).view(-1, 1)
    idx_raw = base + offsets.view(1, -1)
    
    # Handle boundary conditions
    valid = (idx_raw >= 0) & (idx_raw < n)
    idx_clamped = idx_raw.clamp(0, n - 1)
    
    return idx_clamped, ~valid  # indices and mask for invalid positions
```

The core attention computation then becomes:

```python
def sliding_window_attention(q, k, v, window_size, *, causal=True, scale=None):
    """Sliding window attention with configurable window size."""
    bsz, n_heads, seq_len, head_dim = q.shape
    scale = scale or 1.0 / math.sqrt(head_dim)
    device = q.device
    
    # Get windowed indices
    idx, mask = _make_indices(seq_len, window_size, causal, device)
    window = idx.size(-1)
    
    # Expand indices for gathering
    gather_idx = idx.view(1, 1, seq_len, window, 1).expand(bsz, n_heads, seq_len, window, head_dim)
    
    # Gather windowed keys and values
    k_win = torch.gather(k.unsqueeze(3).expand_as(gather_idx), 2, gather_idx)
    v_win = torch.gather(v.unsqueeze(3).expand_as(gather_idx), 2, gather_idx)
    
    # Compute attention scores within windows
    scores = (q.unsqueeze(3) * k_win).sum(dim=-1) * scale
    scores = scores.masked_fill(mask.view(1, 1, seq_len, window), float("-inf"))
    
    # Apply softmax and compute output
    attn = F.softmax(scores, dim=-1)
    output = (attn.unsqueeze(-1) * v_win).sum(dim=3)
    
    return output
```

Key engineering decisions:
- **Vectorized gathering**: Uses `torch.gather` for efficiency instead of loops
- **Boundary handling**: Clamps indices and masks invalid positions
- **Memory layout**: Carefully manages tensor shapes for broadcasting
- **Numerical stability**: Proper masking with `-inf` before softmax

---

## Benchmarking: When Does SWA Win?

I built a comprehensive benchmarking framework to compare SWA against full attention across different sequence lengths and window sizes:

```bash
python scripts/benchmark.py \
  --seq 128 256 512 1024 2048 \
  --window 32 64 128 256 \
  --heads 4 --dim 64 --batch 2 --iters 20 --warmup 5 --device auto \
  --out benchmarks/swattn_bench.csv
```

The results tell a compelling story:

![Tokens per second comparison](/assets/images/swa-blog/swattn_tokens_per_s.png)

![Speedup comparison](/assets/images/swa-blog/swattn_speedup.png)

**Key findings:**
- **Short sequences (≤256 tokens)**: Full attention wins due to its simpler implementation
- **Medium sequences (512-1024 tokens)**: SWA becomes competitive, especially with smaller windows
- **Long sequences (≥2048 tokens)**: SWA dominates, with speedups of 2-4x depending on window size
- **Memory efficiency**: SWA uses significantly less GPU memory, enabling longer sequences on the same hardware

What this taught me:
- The crossover point depends on both sequence length and window size
- Hardware characteristics (memory bandwidth vs compute) affect the trade-offs
- Benchmarking on real workloads is essential—theoretical complexity doesn't tell the whole story

---

## Training a Tiny Language Model: Putting SWA to Work

To demonstrate SWA in action, I trained a character-level language model on Edith Wharton's *The Verdict* (~5k words). This provided a controlled environment to validate the attention mechanism while generating interpretable results.

Training setup:
```bash
python scripts/train_toy_language_model.py \
  --context 256 --window 64 --steps 2000 --batch 24 \
  --data data/tiny.txt \
  --log-csv logs/the_verdict_train.csv --sample-file logs/the_verdict_sample.txt
```

Model architecture:
- **Context length**: 256 characters
- **Window size**: 64 characters (25% of context)
- **Model size**: 4 layers, 4 heads, 128 dimensions
- **Tokenization**: Character-level (simple but effective for this demo)

![Training progress](/assets/images/swa-blog/train_loss.png)

Training dynamics:
- Loss drops from 4.29 to 0.26 over 2000 steps
- Convergence is smooth and stable
- No gradient explosion or vanishing issues

Generated samples show the model learning character-level patterns:

```
Sample after 2000 steps:
"Never bratly.
"Nevever ou that cther dract--o he dofoullin't ing?"
"Be he br flashe or necomiar ne nshof ad sheragut imin n ar hisl aundin tha bspofor ther..."
```

While the words are garbled (character-level modeling is challenging), the model captures:
- Proper punctuation patterns
- Dialogue structure with quotes
- Sentence rhythm and spacing
- English-like character combinations

![Training and sampling results](/assets/images/swa-blog/the_verdict_train and verdict_sample.png)

---

## Scaling Up: Handling Larger Corpora

To showcase SWA's scalability, I documented an optional "larger corpus" experiment with ~100k words (≈600k characters):

```bash
python scripts/train_toy_language_model.py \
  --context 512 --window 128 --steps 4000 --batch 12 \
  --d-model 256 --layers 4 --heads 8 --lr 2e-3 \
  --data data/the_verdict_large.txt \
  --log-csv logs/the_verdict_large_train.csv
```

Scaling insights:
- **Longer contexts**: 512 characters provide richer context for generation
- **Larger windows**: 128-character windows capture more dependencies
- **Model capacity**: Deeper/wider models better utilize the additional context
- **Training stability**: SWA remains stable even with longer sequences

This demonstrates SWA's practical advantage: enabling longer context training on consumer hardware without memory explosions.

---

## Metal Acceleration: Making It Fast on Apple Silicon

One unexpected benefit was how well SWA performs with Metal Performance Shaders (MPS) on Apple Silicon. The training script auto-detects and uses MPS when available:

```python
def get_device():
    """Auto-detect best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
```

Performance observations:
- **MacBook Air M2**: Comfortably handles 512-token sequences with SWA
- **Memory efficiency**: No thermal throttling during training
- **Batch processing**: 12-24 samples per batch without issues

This makes SWA particularly attractive for researchers and practitioners working on consumer hardware.

![Training tokens per second](/assets/images/swa-blog/train_tokens_per_s.png)

---

## Practical Engineering Insights (that saved me debugging time)

**Index management is critical:**
- Off-by-one errors in window boundaries cause silent failures
- Proper boundary clamping prevents out-of-bounds access
- Mask discipline ensures invalid positions don't contribute to attention

**Memory layout matters:**
- Careful tensor reshaping avoids unnecessary copies
- Broadcasting rules can be tricky with 5D tensors (batch, heads, seq, window, dim)
- GPU memory usage spikes during gather operations

**Numerical stability:**
- Masking with `-inf` before softmax is essential
- Scale factor (1/√d) prevents attention score explosion
- Gradient clipping helps with longer sequences

**Testing strategies:**
- Compare outputs with full attention on short sequences
- Verify attention weights sum to 1 within windows
- Test boundary conditions (sequence length < window size)

Common gotchas I navigated:
- Forgetting to expand indices for multi-head attention
- Incorrect mask broadcasting across batch and head dimensions
- Window size larger than sequence length edge cases
- Causal vs bidirectional window confusion

---

## What "Good" Looked Like at Each Stage

**Index generation:**
- Indices stay within bounds [0, seq_len)
- Mask correctly identifies invalid positions
- Causal constraint properly enforced (no future peeking)

**Attention kernel:**
- Output shapes match input query shapes
- Attention weights sum to 1 within valid windows
- Gradients flow correctly through the computation

**Integration testing:**
- Model trains without memory explosions
- Loss decreases on small datasets
- Generated samples show learning progress

**Performance validation:**
- Speedup measurements match theoretical expectations
- Memory usage scales linearly with sequence length
- Benchmarks reproduce across different hardware

---

## Results and What I'd Build Next

**Current achievements:**
- Working SWA implementation with configurable windows
- 2-4x speedup on long sequences compared to full attention
- Successful training of language models on consumer hardware
- Comprehensive benchmarking framework

**Immediate extensions:**
- **Dilated windows**: Skip patterns for even longer-range dependencies
- **Learned windows**: Adaptive window sizes based on content
- **Hybrid approaches**: Combine SWA with sparse attention patterns

**Production considerations:**
- **Subword tokenization**: Move beyond character-level for efficiency
- **Mixed precision**: FP16/BF16 for additional speedups
- **Gradient accumulation**: Handle larger effective batch sizes
- **Checkpointing**: Memory-efficient training for very long sequences

**Research directions:**
- **Window size scheduling**: Dynamic windows during training
- **Multi-scale attention**: Different window sizes per head
- **Retrieval augmentation**: Combine SWA with external memory

---

## The Bigger Picture: Efficient Attention for Everyone

This project reinforced a key insight: the future of language modeling isn't just about bigger models—it's about smarter architectures that democratize access to powerful capabilities.

Sliding-window attention represents one point in a rich design space of efficient attention mechanisms. By implementing it from scratch, I gained intuition for:
- The trade-offs between expressiveness and efficiency
- How hardware constraints shape algorithmic choices  
- The importance of empirical validation over theoretical analysis

Most importantly, SWA makes longer-context modeling accessible on everyday hardware. Whether you're a researcher with a laptop or a startup with limited compute budget, efficient attention mechanisms open new possibilities.

---

Thank you for reading! You can find the complete code, benchmarks, and training scripts for this project on GitHub: **[sliding-window-attention](https://github.com/venkateshterikuti/sliding-window-attention)**

This exploration was inspired by the growing need for efficient attention mechanisms in the era of longer contexts and larger models.

---
