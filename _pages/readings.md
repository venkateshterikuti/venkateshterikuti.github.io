---
permalink: /readings/
title: "Readings"
seo_title: "Venkatesh Terikuti readings and research papers"
excerpt: "A curated list of papers, articles, and resources worth revisiting."
layout: single
classes: wide
toc: false
---

<div id="tag-filter" style="margin-bottom: 20px;">
  <strong>Filter by tags:</strong>
  <button class="tag-btn active" onclick="filterPapers('all')">All</button>
</div>

A curated collection of papers, articles, and resources that have influenced my work in data science, machine learning, and LLMs. Each entry includes detailed summaries and key insights.

<div id="papers-container">

<div class="paper-item" data-tags="transformers,nlp,attention">
<h3><a href="https://arxiv.org/abs/1706.03762" target="_blank"><strong>Attention Is All You Need</strong></a></h3>
<p>This groundbreaking paper introduced the Transformer architecture, completely revolutionizing natural language processing by replacing recurrent and convolutional layers with self-attention mechanisms. The authors demonstrated that attention mechanisms alone can achieve state-of-the-art results on translation tasks while being more parallelizable. The key innovation is the multi-head self-attention mechanism that allows the model to jointly attend to information from different representation subspaces.</p>
</div>

<div class="paper-item" data-tags="nlp,pre-training,bert">
<h3><a href="https://arxiv.org/abs/1810.04805" target="_blank"><strong>BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding</strong></a></h3>
<p>BERT introduced bidirectional training of Transformers, enabling the model to learn context from both left and right sides simultaneously. Unlike previous models that were either left-to-right or right-to-left, BERT uses masked language modeling and next sentence prediction during pre-training. This approach achieved significant improvements across a wide range of NLP tasks and established the foundation for many subsequent language models.</p>
</div>

<div class="paper-item" data-tags="llm,gpt,few-shot">
<h3><a href="https://arxiv.org/abs/2005.14165" target="_blank"><strong>Language Models are Few-Shot Learners</strong></a></h3>
<p>The GPT-3 paper demonstrated that scaling language models to 175 billion parameters enables remarkable few-shot learning capabilities without gradient updates. The model can perform various tasks by simply providing a few examples in the prompt, showcasing emergent abilities that weren't explicitly trained for. This work highlighted the importance of scale in language models and introduced in-context learning as a powerful paradigm.</p>
</div>

<div class="paper-item" data-tags="computer-vision,resnet,deep-learning">
<h3><a href="https://arxiv.org/abs/1512.03385" target="_blank"><strong>Deep Residual Learning for Image Recognition</strong></a></h3>
<p>This paper introduced residual connections (skip connections) that solved the degradation problem in very deep neural networks. The key insight is that it's easier to learn residual mappings than the original mappings, allowing networks to be much deeper while avoiding vanishing gradients. ResNet architectures achieved unprecedented accuracy on ImageNet and other computer vision benchmarks.</p>
</div>

<div class="paper-item" data-tags="visualization,dimensionality-reduction,tsne">
<h3><a href="https://jmlr.org/papers/v9/vandermaaten08a.html" target="_blank"><strong>Visualizing Data using t-SNE</strong></a></h3>
<p>t-SNE (t-Distributed Stochastic Neighbor Embedding) provides a powerful technique for visualizing high-dimensional data by preserving local structure while revealing global patterns. The method converts similarities between data points into joint probabilities and minimizes the divergence between these probabilities in high and low dimensions. It's particularly effective at revealing clusters and local neighborhoods in complex datasets.</p>
</div>

<div class="paper-item" data-tags="machine-learning,gradient-boosting,xgboost">
<h3><a href="https://arxiv.org/abs/1603.02754" target="_blank"><strong>XGBoost: A Scalable Tree Boosting System</strong></a></h3>
<p>XGBoost introduced significant algorithmic and system optimizations to gradient boosting, making it highly scalable and efficient. Key innovations include a novel tree learning algorithm that handles sparse data, weighted quantile sketch for approximate learning, and parallel and distributed computing capabilities. The system incorporates regularization techniques to prevent overfitting and provides excellent performance on structured/tabular data.</p>
</div>

<div class="paper-item" data-tags="conference,research,neurips">
<h3><a href="https://neurips.cc/" target="_blank"><strong>Neural Information Processing Systems (NeurIPS)</strong></a></h3>
<p>NeurIPS is one of the most prestigious conferences in machine learning and computational neuroscience, featuring cutting-edge research across theoretical foundations, algorithms, and applications. The conference proceedings provide insights into the latest developments in deep learning, reinforcement learning, optimization, and neuroscience-inspired computing. It serves as a barometer for emerging trends and breakthrough research in AI.</p>
</div>

<div class="paper-item" data-tags="transformers,attention,optimization,flash-attention">
<h3><a href="https://arxiv.org/abs/2205.14135" target="_blank"><strong>FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness</strong></a></h3>
<p>FlashAttention introduces an IO-aware exact attention algorithm that significantly reduces memory usage and wall-clock time for transformer models. The key innovation is tiling the attention computation to fit within fast on-chip memory (SRAM), reducing the number of memory reads/writes to high-bandwidth memory (HBM). This approach achieves 2-4x speedup and enables training on much longer sequences.</p>
</div>

<div class="paper-item" data-tags="transformers,attention,efficiency,multi-query">
<h3><a href="https://arxiv.org/abs/2305.13245" target="_blank"><strong>Fast Transformer Decoding: One Write-Head is All You Need</strong></a></h3>
<p>This paper introduces Multi-Query Attention (MQA), which shares key and value heads across all query heads while maintaining separate query projections. This dramatically reduces the KV cache memory requirements during inference, enabling faster decoding with minimal quality degradation. MQA achieves significant memory savings (up to 32x reduction in KV cache size) and faster inference speeds, especially for long sequences.</p>
</div>

<div class="paper-item" data-tags="llm,llama,grouped-query,attention">
<h3><a href="https://arxiv.org/abs/2307.09288" target="_blank"><strong>Llama 2: Open Foundation and Fine-Tuned Chat Models</strong></a></h3>
<p>The Llama 2 paper introduces Grouped-Query Attention (GQA), which strikes a balance between Multi-Head Attention (MHA) and Multi-Query Attention (MQA). GQA groups query heads and shares key-value heads within each group, reducing KV cache memory requirements while maintaining better quality than pure MQA. This technique provides a configurable trade-off between memory efficiency and model quality.</p>
</div>

<div class="paper-item" data-tags="visualization,explainability,distill">
<h3><a href="https://distill.pub/" target="_blank"><strong>Distill.pub</strong></a></h3>
<p>Distill represents a new paradigm in scientific publishing, focusing on clear explanations and interactive visualizations of machine learning concepts. The platform emphasizes visual and interactive elements that help readers build intuition about complex algorithms and mathematical concepts. Articles often include live code, interactive diagrams, and novel visualization techniques that make abstract concepts accessible.</p>
</div>

</div>

<script>
function filterPapers(tag) {
  const papers = document.querySelectorAll('.paper-item');
  const buttons = document.querySelectorAll('.tag-btn');
  
  // Remove active class from all buttons
  buttons.forEach(btn => btn.classList.remove('active'));
  
  // Add active class to clicked button
  event.target.classList.add('active');
  
  papers.forEach(paper => {
    if (tag === 'all') {
      paper.style.display = 'block';
    } else {
      const paperTags = paper.getAttribute('data-tags').split(',');
      if (paperTags.includes(tag)) {
        paper.style.display = 'block';
      } else {
        paper.style.display = 'none';
      }
    }
  });
}

// Generate tag filter buttons dynamically
document.addEventListener('DOMContentLoaded', function() {
  const allTags = new Set();
  const papers = document.querySelectorAll('.paper-item');
  
  papers.forEach(paper => {
    const tags = paper.getAttribute('data-tags').split(',');
    tags.forEach(tag => allTags.add(tag));
  });
  
  const filterDiv = document.getElementById('tag-filter');
  const sortedTags = Array.from(allTags).sort();
  
  sortedTags.forEach(tag => {
    const btn = document.createElement('button');
    btn.className = 'tag-btn';
    btn.textContent = tag.charAt(0).toUpperCase() + tag.slice(1).replace('-', ' ');
    btn.onclick = () => filterPapers(tag);
    filterDiv.appendChild(btn);
  });
});
</script>

<style>
.paper-item {
  margin-bottom: 30px;
  padding: 20px;
  border: 1px solid #e1e4e8;
  border-radius: 8px;
  background: #fafbfc;
}

.paper-item h3 {
  margin-top: 0;
  margin-bottom: 15px;
  color: #24292e;
}

.paper-item p {
  margin-bottom: 15px;
  line-height: 1.6;
  color: #586069;
}

.tag-btn {
  display: inline-block;
  padding: 4px 8px;
  margin: 2px 4px 2px 0;
  background-color: #f1f8ff;
  color: #0366d6;
  border: 1px solid #c8e1ff;
  border-radius: 3px;
  font-size: 12px;
  cursor: pointer;
  text-decoration: none;
  transition: all 0.2s ease;
}

.tag-btn:hover {
  background-color: #0366d6;
  color: white;
}

.tag-btn.active {
  background-color: #0366d6;
  color: white;
}

#tag-filter {
  margin-bottom: 30px;
  padding: 15px;
  background: #f6f8fa;
  border-radius: 8px;
  border: 1px solid #e1e4e8;
}

#tag-filter strong {
  margin-right: 10px;
}
</style>

---

*This collection is regularly updated with new discoveries and insights. Each paper includes detailed summaries highlighting key contributions and practical applications.*
