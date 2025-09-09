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

🚀 I specialize in taking models from research to production—whether it's deploying real-time anomaly detection for 6M+ daily events, building MLOps platforms that serve 10+ models, or fine-tuning transformers for 90% better search precision. I've led distributed teams across three continents and authored research submitted to CIKM conference.

**⚡ What excites me:** The intersection of cutting-edge AI research and scalable engineering solutions that drive real business impact.

📄 View my CV [here](/assets/docs/vterikuti.pdf).

## Blog 
📝 I maintain a small blog where I share code tutorials and insights on various deep learning topics, feel free to [take a look](/blog/)!

{% for post in site.posts limit:2 %}
{% if post.title and post.title != "" %}
- [{{ post.title }}]({{ post.url }}) — {{ post.date | date: "%B %Y" }}  
  {{ post.excerpt | strip_html | truncatewords: 15 }}
{% endif %}
{% endfor %}

[View all posts →](/blog/)

## Projects

- **Customer Churn Prediction for Telecom** — `Machine Learning` `Data Science`  
  Built ensemble models achieving 94% accuracy in predicting customer churn. Implemented feature engineering pipeline and deployed REST API for real-time predictions.

- **Graph Neural Networks for High Energy Physics** — `Deep Learning` `Research`  
  Applied GNN architectures to particle physics data analysis. Collaborated with CERN researchers to improve event classification accuracy by 15%.

### In Progress
- **LLM-Powered Code Review Assistant** — `LLM` `MLOps`  
  Building an intelligent code review system using fine-tuned language models. Integrating with GitHub Actions for automated PR analysis.

- **Real-time Anomaly Detection System** — `Streaming` `ML Engineering`  
  Developing scalable anomaly detection for IoT sensor data using Apache Kafka and online learning algorithms.

[View all projects →](/activities/)

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>