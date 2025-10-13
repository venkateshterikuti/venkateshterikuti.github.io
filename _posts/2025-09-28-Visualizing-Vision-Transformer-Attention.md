---
layout: single
classes: wide
author_profile: true
title: "Visualizing Vision Transformer Attention: From Raw Weights to Beautiful Heatmaps"
seo_title: "How to Visualize ViT Attention Maps with PyTorch and Transformers - Tutorial with Code"
published: true
---

TL;DR: *Vision Transformers have taken the computer vision world by storm, but understanding what they "see" remains challenging. This post walks through building a complete attention visualization pipeline—from extracting raw attention weights to generating publication-quality heatmaps that reveal how ViTs process images layer by layer.*

> **Run the code yourself.** The companion notebook [`vit_attention_visualization.ipynb`](https://github.com/venkateshterikuti/notebooks/blob/main/vit_attention_visualization.ipynb) contains the complete attention visualization pipeline. You can download it and upload to [Google Colab](https://colab.research.google.com/) via *File -> Upload notebook*, or open it directly in Colab using the GitHub integration.

---

## What I Built

- A reusable attention extraction pipeline for any Hugging Face Vision Transformer
- Layer-wise attention overlays showing how focus evolves from textures to semantics
- Multi-head attention visualization revealing complementary attention patterns
- Quantitative analysis using attention entropy to measure focus distribution

![Layer-wise attention evolution](/assets/images/vit-attention-blog/02_layer_wise_attention.png)

## Why Visualize Attention?

Vision Transformers process images by splitting them into patches and applying self-attention across these patches. Unlike CNNs where receptive fields are architecturally defined, ViTs learn where to look. This flexibility is powerful but opaque—we need visualization to understand:

- **How attention evolves**: Do early layers focus broadly while later layers sharpen?
- **Head specialization**: Do different attention heads capture different visual features?
- **Attention efficiency**: Is the model distributing focus effectively or wasting capacity?

The key insight: the `CLS` token's attention to image patches reveals which regions the model considers important for classification. By visualizing this attention across layers and heads, we can peek into the model's decision-making process.

---

## Setting Up the Environment

We'll use PyTorch, Hugging Face Transformers, and scikit-image for sample data. The visualization stack includes matplotlib and seaborn for clean, publication-ready plots.

```python
import os
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import ViTModel, ViTImageProcessor
import torch.nn.functional as F

sns.set_theme(style='white', context='talk')
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
```

---

## Loading a Pretrained Vision Transformer

We'll use Google's `vit-base-patch16-224` model—a standard ViT with 12 layers and 12 attention heads per layer. The model splits each 224×224 image into 196 patches (14×14 grid of 16×16 patches).

```python
MODEL_NAME = 'google/vit-base-patch16-224'
processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
model = ViTModel.from_pretrained(MODEL_NAME)
model.set_attn_implementation('eager')  # Required to access attention weights
model.to(device)
model.eval()

print(f'Loaded model with {model.config.num_hidden_layers} layers')
print(f'and {model.config.num_attention_heads} heads per layer')
```

**Critical detail**: Setting `attn_implementation='eager'` ensures attention weights are computed explicitly rather than fused into optimized kernels. This makes them accessible for visualization.

---

## Curating Sample Images

We'll use scikit-image's built-in dataset for reproducibility—it includes diverse images perfect for exploring attention patterns:

```python
from skimage import data

IMAGE_SPECS = [
    {
        'name': 'astronaut',
        'array': data.astronaut(),
        'caption': 'Astronaut selfie floating in space',
    },
    {
        'name': 'chelsea-cat',
        'array': data.chelsea(),
        'caption': 'Chelsea the cat lounging on a carpet',
    },
    {
        'name': 'rocket',
        'array': data.rocket(),
        'caption': 'NASA test rocket above the desert',
    },
]

raw_images = []
model_inputs = []
for spec in IMAGE_SPECS:
    image = Image.fromarray(spec['array']).convert('RGB')
    raw_images.append({'name': spec['name'], 'caption': spec['caption'], 'image': image})
    inputs = processor(images=image, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model_inputs.append(inputs)
```

![Sample image gallery](/assets/images/vit-attention-blog/01_image_gallery.png)

These images offer different visual characteristics:
- **Astronaut**: Complex human features with space suit details
- **Cat**: Furry textures with distinct facial features
- **Rocket**: High-contrast mechanical object against desert background

---

## Extracting and Processing Attention

The core challenge is transforming raw attention tensors into spatial heatmaps. Attention weights are stored as `(batch, heads, tokens, tokens)` tensors. We focus on the `CLS` token's attention to patch tokens—this reveals where the model looks for classification.

```python
def gather_attention(inputs, layer_indices=None):
    """Extract and process attention weights from specified layers."""
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    # Stack attention from all layers: (layers, batch, heads, tokens, tokens)
    attentions = torch.stack(outputs.attentions)
    
    if layer_indices is not None:
        attentions = attentions[layer_indices]
    
    # Average across selected layers
    attn = attentions.mean(dim=0)  # (batch, heads, tokens, tokens)
    
    # Extract CLS token attention to patches (skip CLS in target)
    cls_attn = attn[:, :, 0, 1:]  # (batch, heads, num_patches)
    
    # Reshape to spatial grid
    num_heads = cls_attn.shape[1]
    num_patches = cls_attn.shape[-1]
    height = width = int(np.sqrt(num_patches))  # 14x14 for base ViT
    cls_attn = cls_attn.reshape(-1, num_heads, height, width)
    
    # Normalize for visualization (sum to 1 per head)
    cls_attn = cls_attn / (cls_attn.flatten(2).sum(dim=-1, keepdim=True)
                           .reshape(-1, num_heads, 1, 1) + 1e-6)
    
    return cls_attn  # (batch, heads, H, W)
```

**Key engineering decisions**:
- Slice `0, 1:` to get CLS→patch attention (not patch→patch)
- Normalize per-head for fair comparison across layers
- Reshape to spatial grid for intuitive visualization

---

## Upscaling Attention to Image Resolution

Raw attention maps are 14×14 (for patch16 models), but we need 224×224 for clean overlays. Bicubic interpolation provides smooth upscaling without artifacts:

```python
def upscale_attention_map(attn_map, target_hw=(224, 224)):
    """Upscale attention map to image resolution."""
    attn_tensor = torch.from_numpy(attn_map).unsqueeze(0).unsqueeze(0)
    attn_resized = F.interpolate(
        attn_tensor, 
        size=target_hw, 
        mode='bicubic', 
        align_corners=False
    )
    attn_resized = attn_resized.squeeze().numpy()
    
    # Normalize to [0, 1] for visualization
    attn_resized = (attn_resized - attn_resized.min()) / \
                   (attn_resized.max() - attn_resized.min() + 1e-6)
    
    return attn_resized
```

---

## Layer-Wise Attention Evolution

One of the most insightful visualizations shows how attention changes across network depth. We compare early (layer 2), middle (layer 7), and late (layer 12) layers:

```python
LAYER_SPLITS = {
    'Early (Layer 2)': slice(1, 2),
    'Middle (Layer 7)': slice(6, 7),
    'Late (Layer 12)': slice(11, 12),
}

fig, axes = plt.subplots(len(raw_images), len(LAYER_SPLITS) + 1, figsize=(16, 12))

for row_idx, (item, inputs) in enumerate(zip(raw_images, model_inputs)):
    # Show original image
    axes[row_idx, 0].imshow(item['image'])
    axes[row_idx, 0].set_title('Original', fontsize=13)
    axes[row_idx, 0].axis('off')

    # Show attention at different depths
    for col_idx, (label, layer_slice) in enumerate(LAYER_SPLITS.items(), start=1):
        attn = gather_attention(inputs, layer_indices=layer_slice)
        averaged = attn.mean(dim=1)[0].cpu().numpy()  # Average across heads
        overlay = upscale_attention_map(averaged)
        
        axes[row_idx, col_idx].imshow(item['image'])
        axes[row_idx, col_idx].imshow(overlay, cmap='magma', alpha=0.6)
        axes[row_idx, col_idx].set_title(label, fontsize=13)
        axes[row_idx, col_idx].axis('off')

plt.tight_layout()
plt.show()
```

![Layer-wise attention progression](/assets/images/vit-attention-blog/02_layer_wise_attention.png)

**Observations**:
- **Early layers**: Distribute attention broadly, surveying textures and edges
- **Middle layers**: Begin concentrating on semantically relevant regions
- **Late layers**: Sharply focus on the main subject (astronaut's face, cat's head, rocket body)

This progression mirrors hierarchical feature learning in CNNs but emerges from learned attention rather than fixed receptive fields.

---

## Multi-Head Attention Diversity

Different attention heads within the same layer often specialize in complementary patterns. To visualize this, we select the most diverse heads from the final layer using cosine distance:

```python
from scipy.spatial.distance import pdist, squareform

def pick_diverse_heads(attn_maps, top_k=6):
    """Select heads with maximally diverse attention patterns."""
    # Flatten spatial dimensions
    flattened = attn_maps.reshape(attn_maps.shape[0], -1)
    
    # Compute pairwise cosine distances
    distances = squareform(pdist(flattened, metric='cosine'))
    
    # Greedily select diverse heads
    seed = np.argmax(distances.sum(axis=0))  # Most different from others
    selected = [seed]
    
    while len(selected) < top_k:
        remaining = [idx for idx in range(attn_maps.shape[0]) if idx not in selected]
        scores = []
        for idx in remaining:
            # Minimum distance to already selected heads
            diversity = min(distances[idx, sel] for sel in selected)
            scores.append((diversity, idx))
        scores.sort(reverse=True)
        selected.append(scores[0][1])
    
    return selected
```

Visualizing the selected heads:

```python
sample_idx = 0  # Use astronaut image
sample_item = raw_images[sample_idx]
sample_inputs = model_inputs[sample_idx]

last_layer_attn = gather_attention(sample_inputs, layer_indices=slice(-1, None))
head_maps = last_layer_attn[0].cpu().numpy()
selected_heads = pick_diverse_heads(head_maps, top_k=6)

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for ax, head_idx in zip(axes, selected_heads):
    overlay = upscale_attention_map(head_maps[head_idx])
    ax.imshow(sample_item['image'])
    ax.imshow(overlay, cmap='viridis', alpha=0.55)
    ax.set_title(f'Head {head_idx}', fontsize=12)
    ax.axis('off')

fig.suptitle(f"Diverse attention heads — {sample_item['caption']}", fontsize=16)
plt.tight_layout()
plt.show()
```

![Multi-head attention diversity](/assets/images/vit-attention-blog/03_multihead_attention.png)

**What we learn**:
- Some heads focus on the central subject (astronaut's helmet)
- Others attend to contextual elements (space background, suit edges)
- A few heads distribute attention globally (potentially for scene understanding)

This diversity suggests the model uses an ensemble of attention strategies rather than redundant computation.

---

## Quantifying Attention with Entropy

Visual inspection is powerful, but quantitative metrics provide objective comparisons. We use Shannon entropy to measure attention spread:

```python
def attention_entropy(attn_tensor):
    """Compute entropy of attention distribution."""
    # Treat each head's attention as a probability distribution
    probs = attn_tensor.reshape(attn_tensor.shape[0], attn_tensor.shape[1], -1)
    probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-6)
    
    # Shannon entropy: -Σ p log(p)
    entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1)
    
    return entropy.mean(dim=1)  # Average over heads

# Compute entropy across all layers
layer_entropies = []
for layer_idx in range(model.config.num_hidden_layers):
    attn = gather_attention(model_inputs[0], layer_indices=slice(layer_idx, layer_idx + 1))
    entropy = attention_entropy(attn).item()
    layer_entropies.append(entropy)

# Plot entropy profile
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, len(layer_entropies) + 1), layer_entropies, marker='o', color='#1f77b4')
ax.set_xlabel('Layer')
ax.set_ylabel('Attention Entropy')
ax.set_title('Layer-wise Attention Spread')
ax.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()
```

![Attention entropy across layers](/assets/images/vit-attention-blog/04_attention_entropy.png)

**Interpretation**:
- **High entropy** (≈5-6): Attention distributed broadly across many patches
- **Low entropy** (≈3-4): Attention concentrated on few patches
- **The pattern**: Entropy generally decreases in deeper layers as the model narrows focus

Interestingly, entropy sometimes increases in the final layers—possibly when the model needs broader context for final classification decisions.

---

## Practical Engineering Insights

**What worked well**:
- Using `eager` attention implementation exposes weights cleanly
- Averaging across heads gives interpretable summary visualizations
- Bicubic upsampling preserves attention smoothness without artifacts
- Diverse head selection reveals complementary attention strategies

**Common pitfalls I encountered**:
- Forgetting to skip the CLS token in attention targets (creates self-attention noise)
- Not normalizing attention per head (makes cross-layer comparisons invalid)
- Using nearest-neighbor upsampling (creates blocky artifacts)
- Visualizing all heads equally (most are similar; diversity selection is key)

**Performance considerations**:
- Attention extraction is fast (dominated by forward pass)
- Upsampling is cheap with PyTorch's `F.interpolate`
- The main bottleneck is saving high-resolution plots

---

## Key Takeaways

**Technical insights**:
- Vision Transformers progressively refine attention from broad texture surveys to focused semantic regions
- Different attention heads within layers capture complementary visual features
- Attention entropy provides a quantitative measure of focus distribution
- The CLS token's attention serves as a natural importance map for classification

**Practical applications**:
- **Model debugging**: Identify layers where attention fails to focus appropriately
- **Architecture design**: Determine optimal layer counts by observing when attention stabilizes
- **Interpretability**: Explain model predictions by showing which image regions drive decisions
- **Data quality**: Detect when models attend to spurious correlations or artifacts

**Extending this work**:
- **Cross-attention visualization**: For models with encoders and decoders
- **Attention flow**: Track how information propagates through attention paths
- **Comparative analysis**: Visualize attention differences between correctly and incorrectly classified images
- **Video transformers**: Extend to temporal attention in video understanding models

---

## Where to Go From Here

This visualization pipeline works with any Hugging Face Vision Transformer—just change `MODEL_NAME`. Some interesting models to try:

- **DeiT**: Data-efficient training variants with distillation tokens
- **BEiT**: Self-supervised pretraining with masked image modeling
- **Swin Transformer**: Hierarchical architecture with shifted windows
- **ViT-MAE**: Masked autoencoder variants with different attention patterns

The code is modular and reusable—drop it into your research workflow, adapt the color schemes to your publication needs, or extend it with new quantitative metrics.

**The broader lesson**: Attention weights are not just computational mechanisms; they're interpretable summaries of what models consider important. By visualizing them thoughtfully, we bridge the gap between mathematical operations and human understanding.

---

## Code and Resources

The complete code for this project is available in the blog's repository, including:
- Full attention extraction and processing pipeline
- Helper functions for diverse head selection and entropy computation
- Scripts for generating all visualizations shown in this post

Related reads:
- [Attention is All You Need](https://arxiv.org/abs/1706.03762) - Original transformer paper
- [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) - Vision Transformer paper
- [Visualizing Attention in Transformer-Based Language Models](https://arxiv.org/abs/1904.02679) - Attention analysis techniques

---

*Built with PyTorch 2.8.0, Transformers 4.57.0, and a curiosity about what models really see. The visualizations reveal that Vision Transformers don't just process images—they learn to look at them strategically, much like we do.*

