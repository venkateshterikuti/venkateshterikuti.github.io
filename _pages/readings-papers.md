---
permalink: /resources/papers/
title: "Research Papers"
seo_title: "Research papers collection - Venkatesh Terikuti"
excerpt: "Curated collection of influential research papers in AI, ML, and deep learning."
layout: single
classes: wide
toc: false
---

<div class="breadcrumb">
<a href="/resources/">← Back to Resources</a>
</div>

# Research Papers

Curated collection of influential research papers that have shaped the fields of artificial intelligence, machine learning, and deep learning.

---

<div class="category-container">

<div class="resource-item">
<h3><i class="fas fa-file-alt"></i> <a href="https://arxiv.org/abs/2307.09288" target="_blank"><strong>Llama 2: Open Foundation and Fine-Tuned Chat Models</strong></a></h3>
<p>Introduces Grouped-Query Attention (GQA), which strikes a balance between Multi-Head Attention (MHA) and Multi-Query Attention (MQA). GQA groups query heads and shares key-value heads within each group, reducing KV cache memory requirements while maintaining better quality than pure MQA. This technique provides a configurable trade-off between memory efficiency and model quality.</p>
</div>

<div class="resource-item">
<h3><i class="fas fa-file-alt"></i> <a href="https://arxiv.org/abs/2305.13245" target="_blank"><strong>Fast Transformer Decoding: One Write-Head is All You Need</strong></a></h3>
<p>Introduces Multi-Query Attention (MQA), which shares key and value heads across all query heads while maintaining separate query projections. This dramatically reduces the KV cache memory requirements during inference, enabling faster decoding with minimal quality degradation. MQA achieves significant memory savings (up to 32x reduction in KV cache size) and faster inference speeds, especially for long sequences.</p>
</div>

<div class="resource-item">
<h3><i class="fas fa-file-alt"></i> <a href="https://arxiv.org/abs/2205.14135" target="_blank"><strong>FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness</strong></a></h3>
<p>IO-aware exact attention algorithm that significantly reduces memory usage and wall-clock time for transformer models. The key innovation is tiling the attention computation to fit within fast on-chip memory (SRAM), reducing the number of memory reads/writes to high-bandwidth memory (HBM). This approach achieves 2-4x speedup and enables training on much longer sequences.</p>
</div>

<div class="resource-item">
<h3><i class="fas fa-file-alt"></i> <a href="https://arxiv.org/abs/2005.14165" target="_blank"><strong>Language Models are Few-Shot Learners</strong></a></h3>
<p>The GPT-3 paper demonstrated that scaling language models to 175 billion parameters enables remarkable few-shot learning capabilities without gradient updates. The model can perform various tasks by simply providing a few examples in the prompt, showcasing emergent abilities that weren't explicitly trained for. This work highlighted the importance of scale in language models and introduced in-context learning as a powerful paradigm.</p>
</div>

<div class="resource-item">
<h3><i class="fas fa-file-alt"></i> <a href="https://arxiv.org/abs/1810.04805" target="_blank"><strong>BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding</strong></a></h3>
<p>BERT introduced bidirectional training of Transformers, enabling the model to learn context from both left and right sides simultaneously. Unlike previous models that were either left-to-right or right-to-left, BERT uses masked language modeling and next sentence prediction during pre-training. This approach achieved significant improvements across a wide range of NLP tasks and established the foundation for many subsequent language models.</p>
</div>

<div class="resource-item">
<h3><i class="fas fa-file-alt"></i> <a href="https://arxiv.org/abs/1706.03762" target="_blank"><strong>Attention Is All You Need</strong></a></h3>
<p>Groundbreaking paper that introduced the Transformer architecture, completely revolutionizing natural language processing by replacing recurrent and convolutional layers with self-attention mechanisms. The authors demonstrated that attention mechanisms alone can achieve state-of-the-art results on translation tasks while being more parallelizable. The key innovation is the multi-head self-attention mechanism that allows the model to jointly attend to information from different representation subspaces.</p>
</div>

<div class="resource-item">
<h3><i class="fas fa-file-alt"></i> <a href="https://arxiv.org/abs/1512.03385" target="_blank"><strong>Deep Residual Learning for Image Recognition</strong></a></h3>
<p>This paper introduced residual connections (skip connections) that solved the degradation problem in very deep neural networks. The key insight is that it's easier to learn residual mappings than the original mappings, allowing networks to be much deeper while avoiding vanishing gradients. ResNet architectures achieved unprecedented accuracy on ImageNet and other computer vision benchmarks.</p>
</div>

<div class="resource-item">
<h3><i class="fas fa-file-alt"></i> <a href="https://arxiv.org/abs/1603.02754" target="_blank"><strong>XGBoost: A Scalable Tree Boosting System</strong></a></h3>
<p>XGBoost introduced significant algorithmic and system optimizations to gradient boosting, making it highly scalable and efficient. Key innovations include a novel tree learning algorithm that handles sparse data, weighted quantile sketch for approximate learning, and parallel and distributed computing capabilities. The system incorporates regularization techniques to prevent overfitting and provides excellent performance on structured/tabular data.</p>
</div>

<div class="resource-item">
<h3><i class="fas fa-file-alt"></i> <a href="https://jmlr.org/papers/v9/vandermaaten08a.html" target="_blank"><strong>Visualizing Data using t-SNE</strong></a></h3>
<p>t-SNE (t-Distributed Stochastic Neighbor Embedding) provides a powerful technique for visualizing high-dimensional data by preserving local structure while revealing global patterns. The method converts similarities between data points into joint probabilities and minimizes the divergence between these probabilities in high and low dimensions. It's particularly effective at revealing clusters and local neighborhoods in complex datasets.</p>
</div>

</div>

<style>
.breadcrumb {
  margin-bottom: 30px;
  padding: 10px 0;
  border-bottom: 1px solid #e1e4e8;
}

.breadcrumb a {
  color: #0366d6;
  text-decoration: none;
}

.breadcrumb a:hover {
  text-decoration: underline;
}

.category-container {
  display: grid;
  gap: 20px;
  margin-bottom: 20px;
}

.resource-item {
  padding: 20px;
  border: 1px solid #e1e4e8;
  border-radius: 8px;
  background: #fafbfc;
}

.resource-item h3 {
  margin-top: 0;
  margin-bottom: 15px;
  color: #24292e;
}

.resource-item h3 i {
  margin-right: 8px;
  color: #0366d6;
}

.resource-item p {
  margin-bottom: 0;
  line-height: 1.6;
  color: #586069;
}
</style>
