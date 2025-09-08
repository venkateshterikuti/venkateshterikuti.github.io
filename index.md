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

Hi, I'm Venkatesh Terikuti. I work across data science, machine learning, and applied LLMs. This site hosts selected projects, notes, and a rotating list of readings. View my CV [here](/assets/docs/vterikuti.pdf).

## Blog 
📝 I share notes, code snippets, and write‑ups — [take a look](/blog/)!

### Recent Posts
{% assign recent_posts = site.posts | limit: 3 %}
{% for post in recent_posts %}
- [{{ post.title }}]({{ post.url }}) — {{ post.date | date: "%B %Y" }}  
  {{ post.excerpt | strip_html | truncatewords: 15 }}
{% endfor %}

[View all posts →](/blog/)

## Projects

### Pinned Projects
<!-- Add pinned="true" to any project in _data/projects.yml or directly here -->
- **Customer Churn Prediction for Telecom** — `Machine Learning` `Data Science`  
  Built ensemble models achieving 94% accuracy in predicting customer churn. Implemented feature engineering pipeline and deployed REST API for real-time predictions.

- **Graph Neural Networks for High Energy Physics** — `Deep Learning` `Research`  
  Applied GNN architectures to particle physics data analysis. Collaborated with CERN researchers to improve event classification accuracy by 15%.

### Currently Working On
- **LLM-Powered Code Review Assistant** — `LLM` `MLOps`  
  Building an intelligent code review system using fine-tuned language models. Integrating with GitHub Actions for automated PR analysis.

- **Real-time Anomaly Detection System** — `Streaming` `ML Engineering`  
  Developing scalable anomaly detection for IoT sensor data using Apache Kafka and online learning algorithms.

[View all projects →](/activities/)

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>