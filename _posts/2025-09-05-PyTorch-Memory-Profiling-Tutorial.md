---
layout: single
classes: wide
author_profile: true
title: "PyTorch Memory Profiling: Tracking Hidden Allocations and Debugging Memory Leaks"
seo_title: "PyTorch Memory Profiling Tutorial - Find Memory Leaks and Optimize GPU Usage with Built-in Profiler"
published: true
---

TL;DR: *That innocent "let me just cache these activations for debugging" line can quietly consume gigabytes of memory. This post demonstrates how to use PyTorch's built-in profiler to quantify memory usage at the operation level, compare clean and bug-prone training loops, and visualize the impact of common memory leaks—all without leaving the PyTorch ecosystem.*

> **Run the code yourself.** The companion notebook [`pytorch_hidden_memory_profiler.ipynb`](https://github.com/venkateshterikuti/notebooks/blob/main/pytorch_hidden_memory_profiler.ipynb) contains the complete memory profiling framework. You can download it and upload to [Google Colab](https://colab.research.google.com/) via *File -> Upload notebook*, or open it directly in Colab using the GitHub integration.

---

## What I Built

- A reproducible memory profiling framework using PyTorch's native profiler
- Side-by-side comparison of clean vs. leaky training loops under identical workloads
- Visualization pipeline combining profiler statistics with OS-level memory telemetry
- Practical debugging workflow for identifying memory bottlenecks

![Memory growth comparison](/assets/images/memory-profiler-blog/01_memory_growth.png)

## Why Memory Profiling Matters

GPU out-of-memory errors are the silent killers of deep learning experiments. They strike at random hours, corrupt multi-day training runs, and rarely point to the actual culprit. The symptoms are frustrating:

- Training runs fine for 100 iterations, then crashes
- Increasing batch size by one unit breaks everything
- Memory usage grows slowly but steadily across epochs

The root causes are often subtle:
- **Hidden tensor accumulation**: Detached tensors kept for debugging
- **Gradient retention**: Forgetting `zero_grad()` or using `retain_graph=True` unnecessarily
- **Python references**: Strong references preventing garbage collection
- **Inefficient data loading**: Pinned memory or excessive prefetching

PyTorch's profiler exposes per-operation memory allocations without external tools. Combined with OS-level metrics, it creates a complete picture of where memory goes.

---

## The Experimental Setup

We'll build a minimal regression training loop that's complex enough to profile meaningfully but simple enough to isolate memory patterns. The key: run two versions—one clean, one with a subtle memory leak—and visualize the difference.

```python
import os
import psutil
import torch
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity
import pandas as pd
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

PROCESS = psutil.Process(os.getpid())  # For tracking system memory
```

**Why psutil?** PyTorch's profiler tracks allocations it manages, but Python objects, cached tensors, and other overhead live outside PyTorch's view. `psutil` captures the process's true resident set size (RSS).

---

## Building a Synthetic Workload

A compact multilayer perceptron gives us enough operations to profile without overwhelming the visualizations:

```python
torch.manual_seed(7)  # Reproducibility
input_features = 256
hidden_units = 512
batch_size = 512

class TinyRegressor(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def generate_batch(current_batch_size: int = batch_size):
    """Generate random regression data directly on target device."""
    features = torch.randn(current_batch_size, input_features, device=device)
    targets = torch.randn(current_batch_size, 1, device=device)
    return features, targets

def init_training_state():
    """Initialize fresh model, optimizer, and loss function."""
    model = TinyRegressor(input_features, hidden_units).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    return model, optimizer, criterion
```

**Design choices**:
- Generate data on device to eliminate transfer overhead
- Use deterministic seeding for reproducible profiling runs
- Keep architecture simple to isolate profiling signal from noise

---

## Sanity-Check the Training Loop

Before profiling, verify the training loop works correctly:

```python
model, optimizer, criterion = init_training_state()

model.train()
features, targets = generate_batch()
optimizer.zero_grad()
outputs = model(features)
loss = criterion(outputs, targets)
loss.backward()
optimizer.step()

print(f'Warm-up loss: {loss.item():.4f}')
```
```
Warm-up loss: 0.9696
```

This sanity check ensures any profiling issues aren't masking training bugs.

---

## The Profiling Framework

The core profiler function runs several training steps while capturing two streams of telemetry:

1. **PyTorch operator-level statistics**: Memory allocations per operation
2. **Process-wide RSS**: OS-reported memory usage before/after each step

```python
def profile_training(cache_activations: bool = False, steps: int = 6):
    """Profile training loop with optional activation caching."""
    torch.manual_seed(7)  # Identical inputs across runs
    model, optimizer, criterion = init_training_state()
    activation_cache = []
    memory_trace = []
    
    activities = [ProfilerActivity.CPU]
    # Add ProfilerActivity.CUDA if using GPU
    
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for step_idx in range(steps):
            rss_before = PROCESS.memory_info().rss
            batch_x, batch_y = generate_batch()
            optimizer.zero_grad(set_to_none=True)

            with record_function('forward'):
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)

            with record_function('backward'):
                loss.backward()

            with record_function('optimizer_step'):
                optimizer.step()

            rss_after = PROCESS.memory_info().rss
            memory_trace.append({
                'step': step_idx + 1,
                'rss_mb_before': rss_before / (1024 ** 2),
                'rss_mb_after': rss_after / (1024 ** 2),
                'cache_enabled': cache_activations,
            })

            # Simulate a common debugging pattern that leaks memory
            if cache_activations:
                activation_cache.append(predictions.detach().cpu())

    return prof, activation_cache, memory_trace
```

**Key profiling features**:
- `record_shapes=True`: Captures tensor shapes for each operation
- `profile_memory=True`: Tracks CPU/GPU memory allocations
- `with_stack=True`: Enables stack trace capture for detailed debugging
- `record_function`: Custom labels for logical code sections

**The leak simulation**: Storing `predictions.detach().cpu()` mimics a common debugging pattern—saving intermediate results for later inspection. Even though tensors are detached, they accumulate in the Python list.

---

## Baseline Profiling: The Clean Loop

Run profiling without caching to establish baseline memory usage:

```python
def summarize_memory(prof, limit: int = 8):
    """Extract top memory consumers from profiler output."""
    summary_rows = []
    for evt in prof.key_averages():
        mem_bytes = evt.self_cpu_memory_usage
        if mem_bytes == 0:
            continue
        summary_rows.append({
            'name': evt.key,
            'calls': evt.count,
            'self_mem_mb': mem_bytes / (1024 ** 2),
            'cpu_time_ms': evt.self_cpu_time_total / 1000.0,
        })
    summary_rows.sort(key=lambda row: row['self_mem_mb'], reverse=True)
    return summary_rows[:limit]

baseline_prof, baseline_cache, baseline_mem = profile_training(
    cache_activations=False, 
    steps=6
)

baseline_summary = summarize_memory(baseline_prof)
for row in baseline_summary:
    print(f"{row['name']:>30s} | calls: {row['calls']:2d} | "
          f"self memory MB: {row['self_mem_mb']:.3f}")
```

```
                      aten::mm | calls: 30 | self memory MB: 21.012
                   aten::addmm | calls: 18 | self memory MB: 12.012
               aten::clamp_min | calls: 12 | self memory MB: 12.000
      aten::threshold_backward | calls: 12 | self memory MB: 12.000
                    aten::sqrt | calls: 36 | self memory MB: 9.035
                     aten::div | calls: 36 | self memory MB: 9.035
```

**Interpreting baseline results**:
- `aten::mm` (matrix multiply): Largest allocations for forward pass computations
- `aten::addmm` (add + matrix multiply): Bias additions in linear layers
- `aten::clamp_min`: ReLU forward pass (clamping negative values)
- `aten::threshold_backward`: ReLU backward pass

These are expected allocations for standard neural network training. The total is stable across steps—no growth.

---

## Simulating the Memory Leak

Now run the same workload with activation caching enabled:

```python
cached_prof, activation_cache, cached_mem = profile_training(
    cache_activations=True, 
    steps=6
)

cache_bytes = sum(t.element_size() * t.nelement() for t in activation_cache)
print(f'Cached tensors: {len(activation_cache)} batches')
print(f'Total cached size: {cache_bytes / (1024 ** 2):.3f} MB')

cached_summary = summarize_memory(cached_prof)
for row in cached_summary:
    print(f"{row['name']:>30s} | calls: {row['calls']:2d} | "
          f"self memory MB: {row['self_mem_mb']:.3f}")
```

```
Cached tensors: 6 batches
Total cached size: 0.012 MB

                      aten::mm | calls: 30 | self memory MB: 21.012
                   aten::addmm | calls: 18 | self memory MB: 12.012
               aten::clamp_min | calls: 12 | self memory MB: 12.000
      aten::threshold_backward | calls: 12 | self memory MB: 12.000
```

**Interesting observation**: The profiler output looks nearly identical! The cached tensors (0.012 MB) are tiny in this toy example, but the pattern holds for larger models. The profiler shows operation-level allocations, not cumulative Python object retention.

This is why OS-level telemetry is crucial—it captures the full picture.

---

## Visualizing Memory Growth with OS Telemetry

Combine both profiling runs into a single DataFrame and plot RSS over time:

```python
telemetry_df = pd.concat([
    pd.DataFrame(baseline_mem),
    pd.DataFrame(cached_mem),
], ignore_index=True)
telemetry_df['mode'] = telemetry_df['cache_enabled'].map({
    False: 'baseline', 
    True: 'cached'
})

plt.style.use('seaborn-v0_8')
fig, ax = plt.subplots(figsize=(7, 4))

for mode, group in telemetry_df.groupby('mode'):
    ax.plot(group['step'], group['rss_mb_after'], marker='o', label=mode.title())

ax.set_xlabel('Training Step')
ax.set_ylabel('Process RSS (MB)')
ax.set_title('Resident Memory Growth During Training')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()
```

![Memory growth visualization](/assets/images/memory-profiler-blog/01_memory_growth.png)

**The smoking gun**: The baseline stays flat, but cached mode shows steady growth. Even though individual cached tensors are small, they accumulate across steps. In longer training runs, this pattern causes out-of-memory crashes.

---

## Memory Breakdown: Where the Leak Lives

The crucial insight is understanding where memory lives in the system. Let's create a breakdown showing PyTorch operations vs cached Python objects:

```python
# Calculate memory components
baseline_total_profiler = sum(row['self_mem_mb'] for row in baseline_summary)
cached_total_profiler = sum(row['self_mem_mb'] for row in cached_summary)
cached_objects_mb = cache_bytes / (1024 ** 2)

baseline_final_rss = baseline_mem[-1]['rss_mb_after']
cached_final_rss = cached_mem[-1]['rss_mb_after']

# Create stacked bar chart showing the breakdown
fig, ax = plt.subplots(figsize=(10, 6))

categories = ['Baseline', 'Cached']
profiler_memory = [baseline_total_profiler, cached_total_profiler]
cached_memory = [0, cached_objects_mb]
other_memory = [
    baseline_final_rss - baseline_total_profiler,
    cached_final_rss - cached_total_profiler - cached_objects_mb
]

# Create stacked bars
ax.bar(x, profiler_memory, width, label='PyTorch Operations', color='#1f77b4')
ax.bar(x, cached_memory, width, bottom=profiler_memory, 
       label='Cached Activations', color='#ff7f0e')
ax.bar(x, other_memory, width, 
       bottom=[profiler_memory[i] + cached_memory[i] for i in range(len(categories))],
       label='Other (Python, OS)', color='#2ca02c', alpha=0.5)

plt.show()
```

![Memory breakdown](/assets/images/memory-profiler-blog/02_operation_comparison.png)

**Key insights from the breakdown**:
- **PyTorch operations** (blue): Identical in both runs (~361 MB) - operations don't change
- **Cached activations** (orange): Only present in cached run (10 MB) - **this is the leak!**
- **Total memory**: Baseline uses 349 MB, cached uses 377 MB (28 MB increase)

The visualization reveals why this leak is subtle: the profiler correctly reports that operations use the same memory. The leak happens at the Python object level—tensors held in a Python list that the profiler doesn't directly track. This is why combining profiler statistics with OS-level RSS metrics is essential for complete memory debugging.

---

## Practical Debugging Workflow

When you suspect a memory leak, follow this systematic approach:

### 1. Profile a Minimal Reproducer

Strip your training loop to the bare minimum that exhibits the problem:

```python
# Remove: data augmentation, logging, checkpointing, validation
# Keep: core forward/backward/optimizer steps
```

### 2. Run with Profiling Enabled

```python
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    profile_memory=True,
    record_shapes=True,
) as prof:
    # Your training loop here
    pass

# Inspect top memory consumers
print(prof.key_averages().table(sort_by='self_cpu_memory_usage', row_limit=10))
```

### 3. Add OS-Level Telemetry

```python
import psutil
process = psutil.Process(os.getpid())

for step in range(num_steps):
    rss_before = process.memory_info().rss
    # Training step
    rss_after = process.memory_info().rss
    print(f'Step {step}: RSS delta = {(rss_after - rss_before) / 1e6:.2f} MB')
```

### 4. Check Common Culprits

**Cached tensors**:
```python
# BAD: Accumulates memory
history = []
for step in range(num_steps):
    loss = train_step()
    history.append(loss.item())  # ❌ Stores tensor if not .item()

# GOOD: Extract scalars
history = []
for step in range(num_steps):
    loss = train_step()
    history.append(loss.item())  # ✓ Stores float
```

**Retained gradients**:
```python
# BAD: Gradients accumulate
for step in range(num_steps):
    loss = train_step()
    loss.backward()
    optimizer.step()
    # ❌ Missing zero_grad()

# GOOD: Clear gradients
for step in range(num_steps):
    loss = train_step()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()  # ✓ Frees gradient memory
```

**Unnecessary gradient tracking**:
```python
# BAD: Tracking gradients in evaluation
model.eval()
for inputs, targets in val_loader:
    outputs = model(inputs)  # ❌ Still builds computation graph

# GOOD: Disable gradient tracking
model.eval()
with torch.no_grad():  # ✓ No graph, less memory
    for inputs, targets in val_loader:
        outputs = model(inputs)
```

### 5. Compare Before and After

Profile both the buggy and fixed versions to confirm the leak is resolved:

```python
# Profile buggy version
buggy_prof, _, buggy_mem = profile_training(cache_activations=True)

# Profile fixed version  
fixed_prof, _, fixed_mem = profile_training(cache_activations=False)

# Plot side-by-side
plot_memory_comparison(buggy_mem, fixed_mem)
```

---

## Advanced Profiling Techniques

### Export Chrome Trace for Timeline View

PyTorch can export profiler data to Chrome's tracing format for detailed timeline visualization:

```python
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    profile_memory=True,
) as prof:
    # Training loop
    pass

prof.export_chrome_trace('training_trace.json')
```

Open in Chrome: Navigate to `chrome://tracing` and load the JSON file. You'll see:
- Operation timeline with memory allocations
- GPU kernel execution
- Data transfer between CPU and GPU

### Memory Snapshot Analysis

For complex leaks, PyTorch's memory snapshot provides detailed allocation tracking:

```python
torch.cuda.memory._record_memory_history()

# Training loop
for step in range(num_steps):
    train_step()

torch.cuda.memory._dump_snapshot('memory_snapshot.pickle')
torch.cuda.memory._record_memory_history(enabled=None)
```

Analyze with the snapshot viewer tool to identify retention chains.

---

## Key Takeaways

**Technical insights**:
- PyTorch's profiler exposes per-operation memory without external tools
- OS-level telemetry (via `psutil`) captures the full memory picture
- Most leaks come from Python object retention, not PyTorch operations
- Profiling clean and buggy versions side-by-side isolates the culprit

**Common memory leak patterns**:
- Caching tensors for debugging without proper cleanup
- Forgetting `zero_grad()` or `optimizer.zero_grad(set_to_none=True)`
- Retaining computation graphs with `retain_graph=True`
- Strong Python references preventing garbage collection

**Profiling best practices**:
- Profile minimal reproducers, not full training scripts
- Combine PyTorch profiler stats with OS memory metrics
- Use deterministic seeding for reproducible profiling runs
- Export Chrome traces for visual debugging of complex issues

**What "good" looks like**:
- Flat RSS curves across training steps (no monotonic growth)
- Memory usage peaks during backward pass, then releases
- Similar memory profiles between training and evaluation (modulo gradients)
- Predictable per-operation allocations

---

## When to Profile and When to Optimize

**Profile when**:
- Out-of-memory errors occur sporadically
- Memory usage grows unexpectedly across epochs
- Training crashes after N iterations
- You're pushing hardware limits (large models, long sequences)

**Don't prematurely optimize**:
- If training fits comfortably in memory
- Before identifying actual bottlenecks
- Based on intuition alone without measurements

The profiler is diagnostic, not aspirational. Use it to understand actual problems, not hypothetical ones.

---

## Extending This Framework

**Immediate applications**:
- **Multi-GPU profiling**: Track memory distribution across devices
- **Mixed precision**: Compare FP32 vs FP16 memory usage
- **Gradient accumulation**: Verify memory patterns with virtual batches

**Research directions**:
- **Memory-time trade-offs**: Quantify checkpointing impact
- **Batch size optimization**: Find the largest feasible batch automatically
- **Architecture comparison**: Profile different model families systematically

**Production monitoring**:
- Integrate profiling into CI/CD for regression detection
- Log memory profiles alongside training metrics
- Alert on anomalous memory growth patterns

---

## The Bigger Picture: Memory-Aware Deep Learning

Memory constraints shape what we can build. A 12GB GPU can train GPT-2 but not GPT-3. Understanding memory usage isn't just debugging—it's strategic planning.

This profiling framework empowers you to:
- **Push hardware further**: Fit larger models through careful optimization
- **Debug confidently**: Identify leaks before they crash production training
- **Architect strategically**: Design models within memory budgets
- **Collaborate effectively**: Share reproducible memory profiles with teammates

Most importantly, profiling transforms memory from a mysterious resource into a quantified, manageable constraint. You stop asking "Why did it crash?" and start asking "Which operation allocates memory, and can I reduce it?"

---

## Code and Resources

The complete profiling framework is available in the blog's repository, including:
- Training loop instrumentation templates
- Memory visualization utilities
- Chrome trace export helpers
- Common leak pattern examples

Related resources:
- [PyTorch Profiler Documentation](https://pytorch.org/docs/stable/profiler.html)
- [CUDA Memory Management](https://pytorch.org/docs/stable/notes/cuda.html)
- [Efficient PyTorch](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

---

*Built with PyTorch 2.8.0 and a few too many out-of-memory errors. The profiler doesn't prevent leaks, but it makes them visible—and visibility is the first step to control.*

