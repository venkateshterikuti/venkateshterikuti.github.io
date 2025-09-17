---
layout: single
classes: wide
author_profile: true
title: "PAPER: Semantic Search for Conversational E-commerce (CIKM 2025, Under Review)"
seo_title: "Semantic search for conversational e-commerce - two-stage retrieval system with fine-tuned encoders and constraint extraction"
published: true
pinned: true
---

TL;DR: Shoppers increasingly type full sentences ("show me a green iPhone 14 Pro case between $15–$20 with great reviews") instead of keyword salads. We built a two-stage search system that understands conversational queries and retrieves the right products from a 1.3M-item Amazon dataset. Our approach combines fine-tuned sentence encoders with structured constraint extraction, achieving P@1 = 0.80 and R@100 = 0.81 on a challenging benchmark—substantially ahead of baselines.

## System Overview

<pre class="mermaid">
graph TD
    A[Conversational Query<br/>"Best inexpensive folio case for Samsung S22 Plus"] --> B[Parallel Processing]
    
    B --> C[Stage 1: Semantic Retrieval<br/>Fine-tuned MiniLM Encoder]
    B --> D[Stage 2: Constraint Extraction<br/>Fine-tuned Flan-T5-small]
    
    C --> E[Vector Similarity Search<br/>FAISS Index]
    D --> F[Structured Filters<br/>Price, Rating, Category]
    
    E --> G[Top-K Candidates<br/>~1000 products]
    F --> H[Extracted Constraints<br/>inexpensive, folio, S22 Plus]
    
    G --> I[Apply Filters]
    H --> I
    
    I --> J[Final Ranked Results<br/>Semantic + Constraint Match]
    
    style A fill:#e1f5fe
    style J fill:#c8e6c9
    style C fill:#fff3e0
    style D fill:#fff3e0
    style I fill:#fce4ec
</pre>

**Training Pipeline:**
<pre class="mermaid">
graph LR
    K[Amazon Product Catalog<br/>1.3M items] --> L[LLM Synthetic Query Generation<br/>6.5M query-product pairs]
    L --> M[Fine-tune Sentence Encoder<br/>MiniLM on query-product similarity]
    L --> N[Fine-tune Constraint Extractor<br/>Flan-T5 on query→filters]
    
    M --> O[Production System]
    N --> O
    
    style K fill:#e8f5e8
    style L fill:#fff3e0
    style O fill:#c8e6c9
</pre>

*Note: This paper is currently under review at CIKM 2025. The full paper will be made available once published.*

---

## Why This Matters

Most e-commerce search still expects keywords plus manual filters. But user behavior is shifting: people now talk to search like they talk to assistants. Traditional systems miss intent hidden in phrasing, qualifiers, and "by the way" constraints.

Consider these real shopping queries:
- "Best inexpensive folio case for Samsung S22 Plus with magnetic closure and few reviews"
- "Show me wireless earbuds under $50 with noise cancellation and good battery life"
- "Looking for a durable phone charger that works with iPhone 13, fast charging preferred"

Each contains both semantic intent (what kind of product) and structured constraints (price, features, ratings). Our goal was to meet users where they are—understand the full sentence, then deliver tight, trustworthy results.

---

## The Dataset and Scale

We used the Amazon Reviews 2023 dataset, focusing on Cell Phones & Accessories—about 1.3 million products spanning 1996–2023. Each item includes:
- Product titles and descriptions  
- Features and specifications
- Price, reviews, and ratings
- Subcategory labels

We cleaned the text and corrected noisy category labels to make filtering reliable. The scale represents real-world e-commerce complexity while remaining manageable for controlled experimentation.

![Label distribution across product categories](/assets/images/label_distribution.png)

---

## What We Built: A Two-Stage Architecture

Our system combines two cooperating components that handle different aspects of conversational search:

### Stage 1: Semantic Retrieval
We fine-tuned a compact Sentence-Transformer (`multi-qa-MiniLM-L6-cos-v1`) so conversational queries and ideal products land close together in vector space. All product vectors go into a FAISS index for fast nearest-neighbor search over millions of items.

The key insight: general-purpose encoders don't understand e-commerce context. A query like "durable case for iPhone 14" should be close to products with terms like "protective," "rugged," or "drop-resistant"—relationships that emerge through domain-specific fine-tuning.

### Stage 2: Structured Filtering  
We fine-tuned a small instruction-tuned model (`Flan-T5-small`) to translate conversational sentences into structured filters:
- Price ranges (min/max)
- Rating thresholds  
- Review count levels
- Subcategory constraints

These filters are applied to the retrieved set to enforce constraints that embeddings alone don't capture well (like exact price bounds).

![Two-stage pipeline architecture](/assets/images/framework.png)

The system ranks by semantic similarity, then trims by constraints—so you get results that both *feel right* and *fit the rules*.

---

## The Key Insight: Synthetic Queries as Training Data

Real, labeled query-product pairs are scarce in e-commerce. So we generated them at scale.

Using an LLM, we wrote millions of realistic, diverse queries for our catalog—phrased as actual shopper sentences—and paired each with its source product. This gave us:
- **6.5M query–product pairs** covering ~669k products
- **~3.45M unique queries** reflecting natural shopping language
- Rich diversity in phrasing, constraints, and intent

We also enriched a subset with explicit constraints ("under $300", "thousands of reviews") to teach the filter extractor. This synthetic data approach removed the usual labeled-data bottleneck while capturing realistic shopper language patterns.

---

## A Day in the Life of a Query

Let's trace through an example: *"Best inexpensive folio case for Samsung S22 Plus with magnetic closure and few reviews"*

**Step 1 - Semantic Understanding:**  
The fine-tuned embedding model places this query near Samsung S22 Plus folio cases in vector space, understanding that "best" implies quality and "folio" indicates a specific case style.

**Step 2 - Constraint Extraction:**  
The Flan-T5-small model parses:
- "inexpensive" → price level (lower quartile)
- "few reviews" → low review count threshold  
- "folio" + "magnetic closure" → category/feature requirements

**Step 3 - Retrieve & Filter:**  
FAISS returns semantically similar products; the filter stage keeps only those meeting the price, review, and feature criteria.

The result: products that match both the *intent* (folio cases for S22 Plus) and the *constraints* (affordable, low review count, magnetic closure).

---

## Evaluation: Building a Realistic Benchmark

We designed a 100-query benchmark with realistic, multi-constraint prompts like:
- "6.8-inch screen, 5000mAh+, under $400, well-reviewed"
- "Wireless charging pad compatible with iPhone, fast charging, under $30"
- "Bluetooth earbuds with noise cancellation, 8+ hour battery, premium brand"

Ground truth was built through a human-in-the-loop process supported by LLM-assisted triage—every final match was manually verified. We report precision@k and recall@k up to k=100.

This benchmark reflects real shopping complexity: multiple constraints, natural language phrasing, and the need for both semantic understanding and precise filtering.

---

## Results: Substantial Improvements Over Baselines

We compared against two baselines:
1. **Pre-trained encoder only**: Standard sentence transformer without domain fine-tuning
2. **Pre-trained encoder + filters**: Adding structured filtering to baseline

Our approach (**fine-tuned encoder + filters**) delivered the strongest across-the-board gains:

**Precision Metrics:**
- P@1: 0.80 (vs. 0.52 pre-trained only)
- P@5: 0.74 (vs. 0.48 pre-trained only)  
- P@10: 0.71 (vs. 0.45 pre-trained only)

**Recall Metrics:**
- R@100: 0.81 (on average, 81% of all relevant products appear in top-100)

The filter extractor achieved remarkable accuracy on structured constraints:
- **~99.8–99.9%** per-field accuracy
- **99.4%** exact-match across all fields
- Vastly outperformed NER+rules baseline

These results demonstrate that domain-specific fine-tuning is essential for conversational e-commerce search, and that combining semantic understanding with structured filtering creates synergistic improvements.

---

## Practicality: Speed & Resource Efficiency

This isn't a "giant-model only" solution. We specifically chose lightweight models and optimized for real-world constraints:

**Model Choices:**
- **MiniLM** (22M parameters) for semantic encoding
- **Flan-T5-small** (80M parameters) for constraint extraction
- **FAISS** for scalable vector search

**Training Resources:**
- Single NVIDIA V100 (32GB) for all fine-tuning
- Training time: ~4 hours for encoder, ~2 hours for extractor
- Total storage: <500MB for both models

**Query-Time Performance:**
- Vector search: <50ms for 1.3M products
- Constraint extraction: <20ms per query
- End-to-end latency: <100ms

This setup reflects real-world deployment constraints while maintaining strong performance.

---

## What's Novel Here

**Conversational intent + structured rules, together:**  
We combine meaning-based retrieval with explicit constraint extraction, so results both match the semantic intent and obey explicit rules (price, ratings, features).

**Synthetic data as fuel:**  
We use LLMs to create both semantic training pairs and labeled constraints, removing the usual labeled-data bottleneck while reflecting realistic shopper language.

**Designed for scale and generalization:**  
The two-stage architecture can swap in larger encoders for tougher categories and route queries by product type as needed.

**Lightweight but effective:**  
Achieves strong results with compact models that fit real deployment budgets and latency requirements.

---

## Limitations and Future Directions

**LLM consistency in labeling:**  
Synthetic labels can be inconsistent across similar queries. We're exploring techniques to improve label quality and potentially fine-tune the labeling model itself.

**Single-category focus:**  
We focused on Cell Phones & Accessories first. Expanding to multi-category search with cross-category understanding is a natural next step.

**Multimodal opportunities:**  
Current system is text-only. Incorporating product images, specification PDFs, and visual search could enhance understanding.

**Fairness and interpretability:**  
Terms like "cheap," "expensive," or "best" can be subjective. We support configurable thresholds and plan A/B tests to optimize user experience.

**Real-world deployment:**  
While our results are promising, production deployment would require additional considerations around caching, monitoring, and handling edge cases.

---

## The Bigger Picture: Democratizing Conversational Search

This work represents a broader shift toward more natural human-computer interaction in e-commerce. Key insights:

**Users shouldn't think like databases:**  
Natural language queries are becoming the norm. Search systems must adapt to meet users where they are.

**Efficiency enables innovation:**  
By using lightweight models and synthetic data, we make advanced search capabilities accessible to companies without massive ML budgets.

**Hybrid approaches work:**  
Combining different techniques (semantic + structured) often outperforms any single approach alone.

**Evaluation matters:**  
Realistic benchmarks with human validation are essential for measuring progress on complex, subjective tasks like search relevance.

---

## Closing Thoughts

Users shouldn't have to "think like a database." With a small, fast retriever; a tiny intent-to-filters model; and synthetic data generation, we can make e-commerce search conversational, precise, and scalable—without requiring heavyweight infrastructure.

This research demonstrates that the future of e-commerce search lies not just in bigger models, but in smarter architectures that understand natural language while respecting the structured nature of shopping constraints.

The full paper is currently under review at CIKM 2025. I'll share the complete technical details and code once it's published. In the meantime, feel free to reach out with questions or ideas about conversational search, synthetic data generation, or production ML systems.

---

*This work was conducted as part of my applied AI research at University of Arizona, focusing on the intersection of natural language processing and information retrieval in e-commerce contexts.*
