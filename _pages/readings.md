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

<div class="paper-item" data-tags="visualization,explainability,distill">
<h3><i class="fas fa-palette"></i> <a href="https://distill.pub/" target="_blank"><strong>Distill.pub</strong></a></h3>
<p>A new paradigm in scientific publishing, focusing on clear explanations and interactive visualizations of machine learning concepts. Features live code, interactive diagrams, and novel visualization techniques.</p>
</div>

<div class="paper-item" data-tags="book,llm,deep-learning">
<h3><i class="fas fa-book"></i> <a href="https://www.manning.com/books/build-a-large-language-model-from-scratch" target="_blank"><strong>Build a Large Language Model (From Scratch)</strong></a></h3>
<p>Comprehensive guide to building LLMs from the ground up by Sebastian Raschka. Covers tokenization, attention mechanisms, training pipelines, and fine-tuning techniques with hands-on PyTorch implementation.</p>
</div>

<div class="paper-item" data-tags="youtube,education,neural-networks">
<h3><i class="fab fa-youtube"></i> <a href="https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&si=-j5TMHzIkQmwfHQC" target="_blank"><strong>3Blue1Brown - Neural Networks</strong></a></h3>
<p>Exceptional visual explanations of neural networks, backpropagation, and deep learning concepts. Grant Sanderson's intuitive animations make complex mathematical concepts accessible and engaging.</p>
</div>

<div class="paper-item" data-tags="youtube,education,ai">
<h3><i class="fab fa-youtube"></i> <a href="https://www.youtube.com/@AndrejKarpathy" target="_blank"><strong>Andrej Karpathy</strong></a></h3>
<p>Deep learning insights from former OpenAI and Tesla AI director. Features detailed tutorials on building neural networks from scratch, GPT implementations, and AI research discussions.</p>
</div>

<div class="paper-item" data-tags="youtube,education,visualization">
<h3><i class="fab fa-youtube"></i> <a href="https://www.youtube.com/@WelchLabsVideo" target="_blank"><strong>Welch Labs</strong></a></h3>
<p>High-quality educational videos on machine learning, neural networks, and data science with excellent visualizations. Known for clear explanations of complex mathematical concepts.</p>
</div>

<div class="paper-item" data-tags="youtube,education,ai">
<h3><i class="fab fa-youtube"></i> <a href="https://youtube.com/playlist?list=PLujxSBD-JXglGL3ERdDOhthD3jTlfudC2&si=BF3VL8WSurlE8Mc_" target="_blank"><strong>Two Minute Papers - AI & Deep Learning</strong></a></h3>
<p>Quick, engaging summaries of latest AI research papers. Károly Zsolnai-Fehér presents cutting-edge developments in computer graphics, machine learning, and AI with enthusiasm and clarity.</p>
</div>

<div class="paper-item" data-tags="youtube,education,machine-learning">
<h3><i class="fab fa-youtube"></i> <a href="https://www.youtube.com/@krishnaik06" target="_blank"><strong>Krish Naik</strong></a></h3>
<p>Comprehensive machine learning and data science tutorials covering everything from basics to advanced topics. Practical approach with real-world projects and industry insights.</p>
</div>

<div class="paper-item" data-tags="llm,llama,grouped-query,attention">
<h3><i class="fas fa-file-alt"></i> <a href="https://arxiv.org/abs/2307.09288" target="_blank"><strong>Llama 2: Open Foundation and Fine-Tuned Chat Models</strong></a></h3>
<p>Introduces Grouped-Query Attention (GQA), balancing Multi-Head and Multi-Query Attention. Reduces KV cache memory requirements while maintaining better quality than pure MQA.</p>
</div>

<div class="paper-item" data-tags="transformers,attention,efficiency,multi-query">
<h3><i class="fas fa-file-alt"></i> <a href="https://arxiv.org/abs/2305.13245" target="_blank"><strong>Fast Transformer Decoding: One Write-Head is All You Need</strong></a></h3>
<p>Introduces Multi-Query Attention (MQA), sharing key and value heads across query heads. Achieves up to 32x reduction in KV cache size with faster inference speeds.</p>
</div>

<div class="paper-item" data-tags="transformers,attention,optimization,flash-attention">
<h3><i class="fas fa-file-alt"></i> <a href="https://arxiv.org/abs/2205.14135" target="_blank"><strong>FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness</strong></a></h3>
<p>IO-aware attention algorithm that tiles computation to fit within fast on-chip memory (SRAM). Achieves 2-4x speedup and enables training on much longer sequences.</p>
</div>

<div class="paper-item" data-tags="llm,gpt,few-shot">
<h3><i class="fas fa-file-alt"></i> <a href="https://arxiv.org/abs/2005.14165" target="_blank"><strong>Language Models are Few-Shot Learners</strong></a></h3>
<p>GPT-3 paper demonstrating that scaling to 175B parameters enables remarkable few-shot learning capabilities. Introduced in-context learning as a powerful paradigm.</p>
</div>

<div class="paper-item" data-tags="nlp,pre-training,bert">
<h3><i class="fas fa-file-alt"></i> <a href="https://arxiv.org/abs/1810.04805" target="_blank"><strong>BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding</strong></a></h3>
<p>Introduced bidirectional training of Transformers using masked language modeling and next sentence prediction. Established foundation for many subsequent language models.</p>
</div>

<div class="paper-item" data-tags="transformers,nlp,attention">
<h3><i class="fas fa-file-alt"></i> <a href="https://arxiv.org/abs/1706.03762" target="_blank"><strong>Attention Is All You Need</strong></a></h3>
<p>Groundbreaking paper that introduced the Transformer architecture, revolutionizing NLP by replacing recurrent layers with self-attention mechanisms. Key innovation: multi-head self-attention.</p>
</div>

<div class="paper-item" data-tags="computer-vision,resnet,deep-learning">
<h3><i class="fas fa-file-alt"></i> <a href="https://arxiv.org/abs/1512.03385" target="_blank"><strong>Deep Residual Learning for Image Recognition</strong></a></h3>
<p>Introduced residual connections (skip connections) solving the degradation problem in very deep networks. Enabled much deeper architectures while avoiding vanishing gradients.</p>
</div>

<div class="paper-item" data-tags="machine-learning,gradient-boosting,xgboost">
<h3><i class="fas fa-file-alt"></i> <a href="https://arxiv.org/abs/1603.02754" target="_blank"><strong>XGBoost: A Scalable Tree Boosting System</strong></a></h3>
<p>Significant algorithmic and system optimizations to gradient boosting with novel tree learning, weighted quantile sketch, and parallel computing capabilities.</p>
</div>

<div class="paper-item" data-tags="visualization,dimensionality-reduction,tsne">
<h3><i class="fas fa-file-alt"></i> <a href="https://jmlr.org/papers/v9/vandermaaten08a.html" target="_blank"><strong>Visualizing Data using t-SNE</strong></a></h3>
<p>Powerful technique for visualizing high-dimensional data by preserving local structure while revealing global patterns. Particularly effective at revealing clusters and neighborhoods.</p>
</div>

<div class="paper-item" data-tags="conference,research,neurips">
<h3><i class="fas fa-university"></i> <a href="https://neurips.cc/" target="_blank"><strong>Neural Information Processing Systems (NeurIPS)</strong></a></h3>
<p>Premier conference in machine learning and computational neuroscience. Features cutting-edge research across theoretical foundations, algorithms, and applications.</p>
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
