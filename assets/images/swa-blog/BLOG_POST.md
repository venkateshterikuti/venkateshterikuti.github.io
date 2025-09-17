# Sliding-Window Attention

*“Attention is all you need”…until the sequence length doubles and your GPU fans sound like a jet engine.* For this project I wanted to show how sliding-window attention (SWA) offers a quieter alternative—especially on a MacBook Air—by keeping the core idea of attention while taming its quadratic cost.

---

## What’s the Difference Between Full and Sliding-Window Attention?

Full self-attention lets every token peek at every other token. It’s wonderfully expressive, but the cost grows as `O(N²)` with sequence length `N`. Sliding-window attention keeps the gossip local: each token only talks to its nearby neighbors inside a window of width `W`, so the cost becomes `O(N·W)`.

![Image placeholder: schematic comparing full vs sliding-window attention](images/placeholder-full-vs-swa.png)

On short sequences full attention is unbeatable, but for longer runs—the kind that blow up on laptops—SWA buys you speed and memory headroom. The trade-off is obvious: you might miss distant dependencies, yet many language tasks care more about what happens nearby.

---

## Building the Sliding-Window Kernel

Here, I gather the relevant keys/values with vectorized index tensors:

```python
@torch.no_grad()
def _make_indices(n: int, window_size: int, causal: bool, device: torch.device):
    w = min(window_size, n)
    offsets = torch.arange(-w + 1, 1, device=device) if causal else torch.arange(-w // 2, w // 2 + 1, device=device)
    base = torch.arange(n, device=device).view(-1, 1)
    idx_raw = base + offsets.view(1, -1)
    valid = (idx_raw >= 0) & (idx_raw < n)
    return idx_raw.clamp(0, n - 1), ~valid
```

```python
def sliding_window_attention(q, k, v, window_size, *, causal=True, scale=None):
    scale = scale or 1.0 / math.sqrt(q.size(-1))
    idx, mask = _make_indices(seq_len, window_size, causal, device)
    gather_idx = idx.view(1, 1, seq_len, window, 1).expand(bsz, n_heads, seq_len, window, head_dim)
    k_win = torch.gather(k.unsqueeze(3).expand_as(gather_idx), 2, gather_idx)
    v_win = torch.gather(v.unsqueeze(3).expand_as(gather_idx), 2, gather_idx)
    scores = (q.unsqueeze(3) * k_win).sum(dim=-1) * scale
    scores = scores.masked_fill(mask.view(1, 1, seq_len, window), float("-inf"))
    attn = F.softmax(scores, dim=-1)
    return (attn.unsqueeze(-1) * v_win).sum(dim=3)
```

Those helpers plug into a transformer block and ultimately the training script, which auto-detects Metal (MPS).

![Image placeholder: diagram of transformer block with sliding-window attention](images/placeholder-transformer-block.png)

---

## Benchmarking Full vs Sliding-Window

I started by timing both kernels:

```bash
python scripts/benchmark.py \
  --seq 128 256 512 1024 2048 \
  --window 32 64 128 256 \
  --heads 4 --dim 64 --batch 2 --iters 20 --warmup 5 --device auto \
  --out benchmarks/swattn_bench.csv

python scripts/plot_benchmarks.py --csv benchmarks/swattn_bench.csv
```

The plots (`swattn_tokens_per_s.png`, `swattn_speedup.png`) tell a neat story: full attention wins for tiny windows, but SWA scales gracefully as sequences lengthen—exactly what you’d want to highlight in a blog post.

![Image placeholder: benchmark plot showing tokens/sec vs window size](images/placeholder-benchmark-plot.png)

---

## Training a Tiny Language Model on *The Verdict*

For a hands-on demo with Edith Wharton’s *The Verdict* (~5k words) and ran:

```bash
python scripts/train_toy_language_model.py \
  --context 256 --window 64 --steps 2000 --batch 24 \
  --data data/tiny.txt \
  --log-csv logs/the_verdict_train.csv --sample-file logs/the_verdict_sample.txt
```

Loss drops from 4.29 to 0.26 within 2000 steps, and the generated passage (saved to `the_verdict_train and verdict_sample.png`) captures punctuation and rhythm—even if the words melt together in true character-level fashion:

```
Sample:
Igabt und atrig in evefourly here.
"Neve bratly.
"Nevever ou that cther dract--o he dofoullin't ing?"Be he br flashe or necomiar ne nshof ad sheragut imin n ar hisl aundin tha bspofor ther. . . .
```

![Image placeholder: training loss curve from benchmarks/train_loss.png](images/placeholder-training-curve.png)

---

## Scaling Up the Story

To headline how SWA handles bigger workloads, I documented an optional “larger corpus” path. Grab ~100k words (≈600k characters), and train a beefier model:

```bash
python scripts/train_toy_language_model.py \
  --context 512 --window 128 --steps 4000 --batch 12 \
  --d-model 256 --layers 4 --heads 8 --lr 2e-3 \
  --data data/the_verdict_large.txt \
  --log-csv logs/the_verdict_large_train.csv --sample-file logs/the_verdict_large_sample.txt

python scripts/plot_benchmarks.py --train-csv logs/the_verdict_large_train.csv
```

This run hits even lower loss and produces more coherent samples—perfect for a “what happens when we scale” section.

![Image placeholder: comparison of small vs large training curves](images/placeholder-small-vs-large.png)

---

## Wrap-Up

- Sliding-window attention keeps transformers practical on everyday hardware by trading distant dependencies for linear scaling.
- Metal acceleration (MPS) on Apple silicon makes these experiments feasible on a MacBook Air—no workstation required.
- The repo ships with benchmarks, logs, and sample outputs you can weave into your write-up.

Next time I’d like to add prompt-based sampling and maybe a subword tokenizer, but as-is this story already shows how attention mechanics translate into real code and artifacts.
