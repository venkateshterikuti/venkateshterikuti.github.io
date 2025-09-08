---
layout: none
---

var store = [
  {%- comment -%} First: Blog posts from _posts {%- endcomment -%}
  {%- assign posts = site.posts | where_exp:'doc','doc.search != false' -%}
  {%- for doc in posts -%}
    {%- if doc.title and doc.title != "" -%}
      {%- if doc.header.teaser -%}
        {%- capture teaser -%}{{ doc.header.teaser }}{%- endcapture -%}
      {%- else -%}
        {%- assign teaser = site.teaser -%}
      {%- endif -%}
      {
        "title": {{ doc.title | jsonify }},
        "excerpt":
          {%- if site.search_full_content == true -%}
            {{ doc.content | newline_to_br |
              replace:"<br />", " " |
              replace:"</p>", " " |
              replace:"</h1>", " " |
              replace:"</h2>", " " |
              replace:"</h3>", " " |
              replace:"</h4>", " " |
              replace:"</h5>", " " |
              replace:"</h6>", " "|
            strip_html | strip_newlines | jsonify }},
          {%- else -%}
            {{ doc.content | newline_to_br |
              replace:"<br />", " " |
              replace:"</p>", " " |
              replace:"</h1>", " " |
              replace:"</h2>", " " |
              replace:"</h3>", " " |
              replace:"</h4>", " " |
              replace:"</h5>", " " |
              replace:"</h6>", " "|
            strip_html | strip_newlines | truncatewords: 50 | jsonify }},
          {%- endif -%}
        "categories": {{ doc.categories | jsonify }},
        "tags": {{ doc.tags | jsonify }},
        "url": {{ doc.url | relative_url | jsonify }},
        "teaser": {{ teaser | relative_url | jsonify }},
        "type": "blog"
      },
    {%- endif -%}
  {%- endfor -%}
  {%- comment -%} Second: Individual Projects from activities page {%- endcomment -%}
  {%- assign activities_page = site.pages | where: "permalink", "/activities/" | first -%}
  {%- if activities_page -%}
    {%- comment -%}
      Parse markdown list format for individual projects
    {%- endcomment -%}
    {%- assign project_items = activities_page.content | split: '- **' -%}
    {%- for item in project_items -%}
      {%- if item contains '**' and item contains '<br>' -%}
        {%- assign title_parts = item | split: '**' -%}
        {%- if title_parts.size > 1 -%}
          {%- assign title = title_parts[0] | strip -%}
          {%- assign content_after_title = title_parts[1] -%}
          {%- assign desc_parts = content_after_title | split: '<br>' -%}
          {%- if desc_parts.size > 1 -%}
            {%- assign desc = desc_parts[1] | strip_html | strip_newlines | strip -%}
            {%- if title and title != '' and desc and desc != '' -%}
              {
                "title": {{ title | jsonify }},
                "excerpt": {{ desc | jsonify }},
                "categories": [],
                "tags": ["projects", "activities"],
                "url": {{ activities_page.url | relative_url | jsonify }},
                "teaser": "",
                "type": "projects"
              },
            {%- endif -%}
          {%- endif -%}
        {%- endif -%}
      {%- endif -%}
    {%- endfor -%}
    {%- comment -%} Also include the full page for general searches {%- endcomment -%}
    {
      "title": {{ activities_page.title | jsonify }},
      "excerpt": {{ activities_page.content | strip_html | strip_newlines | truncatewords: 50 | jsonify }},
      "categories": [],
      "tags": ["projects", "activities"],
      "url": {{ activities_page.url | relative_url | jsonify }},
      "teaser": "",
      "type": "projects"
    },
  {%- endif -%}
  {%- comment -%} Third: Individual Papers from readings page {%- endcomment -%}
  {%- assign readings_page = site.pages | where: "permalink", "/readings/" | first -%}
  {%- if readings_page -%}
    {%- comment -%} Extract individual papers from the readings page {%- endcomment -%}
    {%- assign paper_sections = readings_page.content | split: '<div class="paper-item"' -%}
    {%- for paper_section in paper_sections -%}
      {%- if paper_section contains '<h3>' -%}
        {%- assign paper_title_start = paper_section | split: '<strong>' -%}
        {%- if paper_title_start.size > 1 -%}
          {%- assign paper_title_end = paper_title_start[1] | split: '</strong>' -%}
          {%- assign paper_title = paper_title_end[0] | strip -%}
          {%- assign paper_summary_start = paper_section | split: '<strong>Summary:</strong>' -%}
          {%- if paper_summary_start.size > 1 -%}
            {%- assign paper_summary_end = paper_summary_start[1] | split: '</p>' -%}
            {%- assign paper_summary = paper_summary_end[0] | strip_html | strip_newlines -%}
            {%- assign paper_tags_start = paper_section | split: 'data-tags="' -%}
            {%- if paper_tags_start.size > 1 -%}
              {%- assign paper_tags_end = paper_tags_start[1] | split: '"' -%}
              {%- assign paper_tags = paper_tags_end[0] | split: ',' -%}
              {%- if paper_title and paper_title != "" and paper_summary and paper_summary != "" -%}
                {
                  "title": {{ paper_title | jsonify }},
                  "excerpt": {{ paper_summary | jsonify }},
                  "categories": [],
                  "tags": {{ paper_tags | jsonify }},
                  "url": {{ readings_page.url | relative_url | jsonify }},
                  "teaser": "",
                  "type": "readings"
                },
              {%- endif -%}
            {%- endif -%}
          {%- endif -%}
        {%- endif -%}
      {%- endif -%}
    {%- endfor -%}
    {%- comment -%} Also include the full page for general searches {%- endcomment -%}
    {
      "title": {{ readings_page.title | jsonify }},
      "excerpt": {{ readings_page.content | strip_html | strip_newlines | truncatewords: 100 | jsonify }},
      "categories": [],
      "tags": ["readings", "papers", "research"],
      "url": {{ readings_page.url | relative_url | jsonify }},
      "teaser": "",
      "type": "readings"
    },
  {%- endif -%}
  {%- comment -%} Other collections {%- endcomment -%}
  {%- for c in site.collections -%}
    {%- unless c.label == "posts" -%}
      {%- assign docs = c.docs | where_exp:'doc','doc.search != false' -%}
      {%- for doc in docs -%}
        {%- if doc.header.teaser -%}
          {%- capture teaser -%}{{ doc.header.teaser }}{%- endcapture -%}
        {%- else -%}
          {%- assign teaser = site.teaser -%}
        {%- endif -%}
        {
          "title": {{ doc.title | jsonify }},
          "excerpt":
            {%- if site.search_full_content == true -%}
              {{ doc.content | newline_to_br |
                replace:"<br />", " " |
                replace:"</p>", " " |
                replace:"</h1>", " " |
                replace:"</h2>", " " |
                replace:"</h3>", " " |
                replace:"</h4>", " " |
                replace:"</h5>", " " |
                replace:"</h6>", " "|
              strip_html | strip_newlines | jsonify }},
            {%- else -%}
              {{ doc.content | newline_to_br |
                replace:"<br />", " " |
                replace:"</p>", " " |
                replace:"</h1>", " " |
                replace:"</h2>", " " |
                replace:"</h3>", " " |
                replace:"</h4>", " " |
                replace:"</h5>", " " |
                replace:"</h6>", " "|
              strip_html | strip_newlines | truncatewords: 50 | jsonify }},
            {%- endif -%}
          "categories": {{ doc.categories | jsonify }},
          "tags": {{ doc.tags | jsonify }},
          "url": {{ doc.url | relative_url | jsonify }},
          "teaser": {{ teaser | relative_url | jsonify }},
          "type": "other"
        },
      {%- endfor -%}
    {%- endunless -%}
  {%- endfor -%}
  {%- comment -%} Other pages (excluding already included ones) {%- endcomment -%}
  {%- if site.lunr.search_within_pages -%}
    {%- assign pages = site.pages | where_exp:'doc','doc.search != false' -%}
    {%- for doc in pages -%}
      {%- unless doc.permalink == "/activities/" or doc.permalink == "/readings/" or doc.permalink == "/blog/" -%}
        {%- if doc.title and doc.title != "" and doc.title != "null" -%}
          {%- if forloop.last -%}
            {%- assign l = true -%}
          {%- endif -%}
          {
            "title": {{ doc.title | jsonify }},
            "excerpt":
                {%- if site.search_full_content == true -%}
                  {{ doc.content | newline_to_br |
                    replace:"<br />", " " |
                    replace:"</p>", " " |
                    replace:"</h1>", " " |
                    replace:"</h2>", " " |
                    replace:"</h3>", " " |
                    replace:"</h4>", " " |
                    replace:"</h5>", " " |
                    replace:"</h6>", " "|
                  strip_html | strip_newlines | jsonify }},
                {%- else -%}
                  {{ doc.content | newline_to_br |
                    replace:"<br />", " " |
                    replace:"</p>", " " |
                    replace:"</h1>", " " |
                    replace:"</h2>", " " |
                    replace:"</h3>", " " |
                    replace:"</h4>", " " |
                    replace:"</h5>", " " |
                    replace:"</h6>", " "|
                  strip_html | strip_newlines | truncatewords: 50 | jsonify }},
                {%- endif -%}
              "url": {{ doc.url | absolute_url | jsonify }},
              "type": "page"
          },
        {%- endif -%}
      {%- endunless -%}
    {%- endfor -%}
  {%- endif -%}
]
