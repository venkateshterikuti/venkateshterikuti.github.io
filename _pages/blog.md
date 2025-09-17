---
permalink: /blog/
title: "Entropy Blog"
seo_title: "Alessio Devoto's blog"
layout: archive
---


{% assign entries_layout = page.entries_layout | default: 'list' %}
{% assign pinned_posts = site.posts | where: "pinned", true %}
{% assign regular_posts = site.posts | where_exp: "post", "post.pinned != true" %}

<div class="entries-{{ entries_layout }}">
  <!-- Pinned Posts First -->
  {% for post in pinned_posts %}
    {% unless post.hidden %}
      {% if post.title and post.date and post.content != "" %}
        <div class="archive__item pinned-post">
          <h2 class="archive__item-title">
            <span class="pinned-badge">📌 PINNED</span>
            <a href="{{ post.url }}">{{ post.title }}</a>
          </h2>
          {% assign words = post.content | number_of_words %}
          {% assign reading_time = words | divided_by: 180 | plus: 1 %}
          <span class="post-meta">
            {{ post.date | date: "%B %d, %Y" }} • {{ reading_time }} min read
          </span>
          {% if post.excerpt %}
            <p class="archive__item-excerpt">
              {{ post.excerpt | markdownify | strip_html | truncate: 160 }}
            </p>
          {% endif %}
        </div>
      {% endif %}
    {% endunless %}
  {% endfor %}
  
  <!-- Regular Posts -->
  {% for post in regular_posts %}
    {% unless post.hidden %}
      {% if post.title and post.date and post.content != "" %}
        <div class="archive__item">
          <h2 class="archive__item-title">
            <a href="{{ post.url }}">{{ post.title }}</a>
          </h2>
          {% assign words = post.content | number_of_words %}
          {% assign reading_time = words | divided_by: 180 | plus: 1 %}
          <span class="post-meta">
            {{ post.date | date: "%B %d, %Y" }} • {{ reading_time }} min read
          </span>
          {% if post.excerpt %}
            <p class="archive__item-excerpt">
              {{ post.excerpt | markdownify | strip_html | truncate: 160 }}
            </p>
          {% endif %}
        </div>
      {% endif %}
    {% endunless %}
  {% endfor %}
</div>

<style>
.post-meta {
  display: inline-block;
  padding: 4px 8px;
  background-color: #f2f2f2;
  border-radius: 4px;
  font-size: 0.85em;
  color: #666;
}

.pinned-post {
  border-left: 4px solid #007acc;
  padding-left: 15px;
  margin-bottom: 2em;
  background-color: #f8f9ff;
  border-radius: 4px;
  padding: 15px;
}

.pinned-badge {
  display: inline-block;
  background: linear-gradient(135deg, #007acc, #0056b3);
  color: white;
  font-size: 0.75em;
  font-weight: bold;
  padding: 3px 8px;
  border-radius: 12px;
  margin-right: 8px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.pinned-post .archive__item-title {
  margin-bottom: 8px;
}
</style>
