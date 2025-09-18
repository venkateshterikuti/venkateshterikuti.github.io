---
layout: single
classes: wide
author_profile: true
title: ''
seo_title: "Venkatesh Terikuti - Data Science & Machine Learning"
excerpt: "Projects, notes, and readings across Data Science, ML, LLMs, and Visualization"
seo_description: "Venkatesh Terikuti — portfolio and notes covering data science, machine learning, LLM applications, and data visualization. Explore featured projects and recent readings."

# toc: true
---

## About

👋 I'm Venkatesh Terikuti, a Senior Data Scientist & Machine Learning Engineer with 6+ years building ML systems that scale to millions of users. Starting my career as a Data Analyst, I've evolved into specializing in production ML systems. Currently pursuing applied AI research at University of Arizona, focusing on LLM-powered semantic search and conversational AI.

🚀 I specialize in taking models from research to production—whether it's deploying real-time anomaly detection for 6M+ daily events, building MLOps platforms that serve 10+ models, or fine-tuning transformers for 90% better search precision. I've led distributed teams across three continents and authored a research paper submitted to CIKM conference.

**⚡ What excites me:** The intersection of cutting-edge AI research and scalable engineering solutions that drive real business impact.

## Blog 
📝 I maintain a small blog where I share code tutorials and insights on various deep learning topics, feel free to [take a look](/blog/)!

{% assign regular_posts = site.posts | where_exp: "post", "post.pinned != true" %}
{% assign pinned_posts = site.posts | where: "pinned", true %}

{% for post in regular_posts limit:1 %}
{% if post.title and post.title != "" %}
- [{{ post.title }}]({{ post.url }}) — {{ post.date | date: "%B %Y" }}  
  {{ post.excerpt | strip_html | truncatewords: 15 }}
{% endif %}
{% endfor %}

{% for post in pinned_posts limit:1 %}
{% if post.title and post.title != "" %}
- [📌 {{ post.title }}]({{ post.url }}) — {{ post.date | date: "%B %Y" }}  
  {{ post.excerpt | strip_html | truncatewords: 15 }}
{% endif %}
{% endfor %}

[View all posts →](/blog/)

## Projects

- **Transformer From Scratch: Scaling Journey** — `Deep Learning` `Transformers` `PyTorch` `Scaling`  
  Complete transformer implementation scaling from 2.4M to 52M parameters. Built character-level and BPE models, compared MacBook vs H100 training, with comprehensive scaling analysis and production infrastructure. <a href="https://github.com/venkateshterikuti/transformer-from-scratch" target="_blank" rel="noopener"><i class="fab fa-github"></i></a>

- **Building LLM From Scratch** — `Deep Learning` `LLM` `PyTorch`  
  Complete GPT-style language model implementation from ground up. Built tokenizer, attention mechanisms, transformer blocks, and training pipelines with pretraining and instruction finetuning. <a href="https://github.com/venkateshterikuti/llm-from-scratch" target="_blank" rel="noopener"><i class="fab fa-github"></i></a>

- **Casino Intelligence Hub** — `Data Science` `Analytics` `MLOps`  
  Advanced casino analytics platform analyzing 16M+ transactions to predict player churn, segment users, and detect anomalies. Built end-to-end pipeline with PostgreSQL, Python ML stack, and interactive dashboards. <a href="https://github.com/venkateshterikuti/casino-intelligence-hub" target="_blank" rel="noopener"><i class="fab fa-github"></i></a>

### In Progress
- **LLM-Powered Code Review Assistant** — `LLM` `MLOps`  
  Building an intelligent code review system using fine-tuned language models. Integrating with GitHub Actions for automated PR analysis.

- **Real-time Anomaly Detection System** — `Streaming` `ML Engineering`  
  Developing scalable anomaly detection for IoT sensor data using Apache Kafka and online learning algorithms.

[View all projects →](/activities/)

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>