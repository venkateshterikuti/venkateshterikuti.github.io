---
layout: single
classes: wide
author_profile: true
title: "Building My First LLM From Scratch"
seo_title: "Building an LLM from scratch - GPT implementation with PyTorch, attention mechanisms, and instruction finetuning"
published: true
---

TL;DR: I didn’t try to build the biggest model—I set out to understand one end‑to‑end. This post is the lab notebook of that journey: the design choices, the checks that saved me hours, and the small wins that stacked into a useful, instructable language model.

---

## What I Built (at a glance)

- A byte-level tokenizer pipeline and a "shift-by-one" next-token dataset.
- Scaled dot-product attention with causal masking and multi-heads.
- A minimal GPT stack: embeddings, positional encodings, transformer blocks, language head, and a generate loop.
- A pretraining routine on unlabeled text with robust batching, stabilization (gradient clipping), and checkpoints.
- Finetuning heads for classification tasks with careful masking and evaluation.
- Instruction finetuning (SFT) on instruction–response pairs with loss masking and quick qualitative evaluation.

![Overall architecture: text → tokenizer → GPT → heads](/assets/images/llm-blog/architecture.jpg)

## Why From Scratch?

- Depth over cargo-culting: I wanted to know why shapes align, where masking applies, and how decoding loops work.
- Transferable skills: debugging attention masks, stabilization tricks, and data hygiene are relevant even with larger frameworks.
- Research agility: when I can open the hood, I can test unconventional ideas without waiting on library support.

---

## Working With Text: Teaching the Model to Read

Computers don't "see" words—they see bytes. I started with a pragmatic choice: a byte-level tokenizer. It's universally valid (supports any UTF-8 text and emojis) and simple to implement, which let me iterate quickly. Later, subword tokenization (like BPE) can compress sequences for efficiency, but "bytes first" kept the early pipeline stable.

Key steps:
- Byte-level encode/decode round-trip test to guarantee reversibility.
- Turn long text into overlapping slices: inputs are tokens [t0..t_{n-2}], labels are [t1..t_{n-1}]. That one-position shift defines the next-token prediction game.
- Deterministic data splits and tiny asserts (shape checks, off-by-one checks) to catch bugs early.

Optional code snippet (kept minimal for readability):
```python
# Sliding window next-token dataset
class NextTokenDataset:
    def __init__(self, token_ids, block_size):
        self.ids, self.bs = token_ids, block_size
    def __len__(self): return len(self.ids) - self.bs
    def __getitem__(self, i):
        x = self.ids[i : i+self.bs]
        y = self.ids[i+1 : i+1+self.bs]
        return x, y
```

![Input processing workflow: raw text → encode → tokens → x/y shift](/assets/images/llm-blog/input-processing-workflow.jpg)

What this taught me:
- Data rigor prevents silent failures (padding, off‑by‑one, reproducibility).
- Translating product text into training‑ready tensors is half the battle.

---

## Attention: Teaching the Model to Look Around

Attention lets each token weigh the importance of other tokens. The mental recipe:
- Project embeddings into queries (Q), keys (K), and values (V).
- Compute scaled dot products QKᵀ / sqrt(d) to stabilize variance.
- Softmax into attention weights; multiply by V to blend context.

Causal masking blocks the future so the model can't peek at tokens it hasn't "seen" yet.

Optional code snippet:
```python
def scaled_dot_product_attention(q, k, v, mask=None):
    d = q.size(-1)
    scores = (q @ k.transpose(-2, -1)) / (d ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ v, weights
```

![Causal mask (lower triangular)](/assets/images/llm-blog/lower-triangular-casual-mask.jpg)

Multi-head attention:
- Multiple heads attend to different patterns (syntax, long-range dependencies, entities).
- Heads are concatenated and projected back to the model dimension.

Quick validations:
- Weights sum to 1 across the last axis.
- Causal mask zeros out upper-triangular positions before softmax.
- Outputs change sensibly when inputs change.

---

## Assembling a Tiny GPT: Giving the Model a Voice

A GPT stack is clean and composable:
- Token and positional embeddings (positions break permutation symmetry).
- N transformer blocks, each: LayerNorm → Attention → residual; LayerNorm → MLP → residual.
- Final LayerNorm and a linear "language head" projecting to vocabulary logits.

Greedy generate loop (kept simple initially) feeds the predicted next token back in and repeats.

![Token flow through GPT block](/assets/images/llm-blog/gpt-predicting-next-token.jpg)

What I looked for:
- No shape/mask mismatches across the stack.
- Loss decreases on small subsets.
- The generate loop produces plausible continuations without index errors.

Optional code snippet:
```python
@torch.no_grad()
def generate(model, idx, max_new_tokens, block_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits = model(idx_cond)
        next_token = logits[:, -1].argmax(-1, keepdim=True)
        idx = torch.cat([idx, next_token], dim=1)
    return idx
```

![Autoregressive generate loop](/assets/images/llm-blog/autoregressive-loop.jpg)

---

## Pretraining on Unlabeled Data: Learning by Immersion

Objective:
- Minimize cross-entropy between predicted next-token distribution and the true next token.

Engineering details that stabilized training:
- Batching: slice long token streams into `block_size` segments, shuffle, and batch to decorrelate.
- Gradient clipping to prevent spikes.
- Periodic evaluation on a held-out split.
- Checkpoint model and optimizer states to resume and compare runs.

Knobs that mattered:
- `block_size`: context window vs. memory trade-off.
- `batch_size`: smoother gradients vs. GPU memory.
- `learning_rate`: warmup or cosine decay helped after reaching a stable baseline.
- `weight_decay`: small values improved generalization slightly.

Qualitative checks:
- Sample short generations at intervals; quick human inspection exposed data or masking bugs faster than metrics alone.

![Pretraining loop overview](/assets/images/llm-blog/pretraining-loop-overview.jpg)

![Loss curve](/assets/images/llm-blog/loss-plot.jpg)

---

## Finetuning for Classification: Teaching Specific Answers

Approach:
- Keep the pretrained backbone.
- Add a small classification head (e.g., MLP) using a pooled representation (last token, mean, or a dedicated CLS token).
- Train with cross-entropy over class labels.

Pitfalls and fixes:
- Padding leakage: ensure attention masks or valid-token pooling to avoid contaminating the representation.
- Learning rate strategy: higher LR on the head, lower on the backbone, or freeze the backbone initially.
- Can it overfit 50 examples? If yes, your setup probably works.

Evaluation:
- Track accuracy and F1.
- Use stratified splits and confusion matrices for small datasets.

![Classification finetuning on GPT backbone](/assets/images/llm-blog/classification-finetuning.jpg)

---

## Instruction Finetuning (SFT): Teaching the Model to Be Helpful

Data:
- Instruction–response pairs with a consistent prompt template.

Loss masking:
- Concatenate the formatted text; compute loss only on response tokens.
- This teaches the model to produce helpful, on-topic answers without penalizing it for the fixed prompt structure.

Generation:
- Use temperature and top-k/top-p sampling to improve diversity and reduce repetition.
- Fix the prompt template for comparability across checkpoints.

![Temperature vs. diversity](/assets/images/llm-blog/temperature-plot.jpg)

Lightweight evaluation:
- A small set of instructions with reference answers.
- Heuristics (token overlap) or an external judge model if available, plus human inspection.

![SFT flow: instruction + response → tokenize → masked loss](/assets/images/llm-blog/sft-flow.jpg)

---

## Practical Engineering Practices (that saved me hours)

- Determinism: fixed seeds and deterministic splits ensured apples-to-apples comparisons.
- Tiny asserts everywhere: shape checks and off-by-one checks saved hours.
- Logging qualitative samples early: surfaced regressions faster than aggregate metrics.
- Regular checkpoints: enabled ablation studies and safe experimentation.
- Mask discipline: causal masks for generation; attention masks for padding; loss masks for SFT.

Common gotchas I navigated:
- Off-by-one errors in dataset shifting.
- Forgetting causal masks and inadvertently letting tokens "see the future."
- Exploding gradients without clipping during early training.
- Padding tokens leaking into pooled representations.
- Accidental overlap between training and evaluation text.

---

## What "Good" Looked Like at Each Stage

- Data: byte-level encode/decode round-trip; dataset x[1:] == y[:-1]; consistent batch shapes.
- Attention: rows of attention weights sum to 1; causal mask effects verified.
- GPT stack: forward pass without errors; logits with expected shapes; generate loop runs.
- Pretraining: loss decreases on a small subset; periodic eval loss captured; checkpoints saved.
- Classification finetuning: can quickly overfit a tiny sample; validation metrics trend upward; no padding leakage.
- Instruction finetuning: responses become more on-topic; qualitative samples improve; quick-eval scores rise.

---

## Short Optional Snippets

Tokenizer placeholder to emphasize universality:
```python
class ByteLevelTokenizer:
    vocab_size = 256
    def encode(self, s): return list(s.encode("utf-8"))
    def decode(self, ids): return bytes(ids).decode("utf-8")
```

Greedy decode loop for clarity:
```python
@torch.no_grad()
def generate(model, idx, max_new_tokens, block_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits = model(idx_cond)
        next_token = logits[:, -1].argmax(-1, keepdim=True)
        idx = torch.cat([idx, next_token], dim=1)
    return idx
```

---

## Results, Demos, and What I'd Build Next

Results highlights:
- Pretrained model generates coherent, on-topic text for short prompts.
- Classification head reaches competitive accuracy on small benchmarks after light tuning.
- Instruction finetuning yields noticeably more helpful and structured answers under a consistent prompt.

![Next token prediction example](/assets/images/llm-blog/next-token-generation.jpg)

Next steps for scale and production:
- Tokenization: migrate to BPE/Unigram for efficiency and shorter sequences.
- Training scale: longer context windows, mixed precision, gradient accumulation, and distributed training.
- Evaluation: richer automatic metrics, human preference ratings, adversarial evaluation sets.
- Post-training: RLHF or DPO for preference alignment.
- Serving: quantization for edge deployment, caching for fast decode, guardrails and logging.

---

Thank you for reading! You can find the complete code and notebooks for this project on GitHub: **[llm-from-scratch](https://github.com/venkateshterikuti/llm-from-scratch)**

This project was inspired by Sebastian Raschka's book "Build a Large Language Model (From Scratch)", which provided invaluable guidance during the journey.

---
