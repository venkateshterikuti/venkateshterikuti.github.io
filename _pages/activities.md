---
permalink: /activities/
title: "Projects & Activities"
seo_title: "Venkatesh Terikuti projects and activities"
# excerpt: "Things I have been doing lately."
# last_modified_at: 2022-05-27T11:59:26-04:00
toc: false
---

<style>
.projects-container {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 30px;
  margin-top: 20px;
}

.project-item {
  background: #f8f9fa;
  padding: 20px;
  border-radius: 8px;
  border-left: 4px solid #007acc;
}

.project-title {
  font-weight: bold;
  font-size: 1.1em;
  margin-bottom: 10px;
  color: #2c3e50;
}

.project-description {
  line-height: 1.6;
  margin-bottom: 15px;
  color: #555;
}

.project-link {
  display: inline-block;
  padding: 8px 16px;
  background: #007acc;
  color: white !important;
  text-decoration: none;
  border-radius: 4px;
  font-size: 0.9em;
  transition: background 0.3s;
}

.project-link:hover {
  background: #005a9e;
  text-decoration: none;
}

@media (max-width: 768px) {
  .projects-container {
    grid-template-columns: 1fr;
    gap: 20px;
  }
}
</style>

<div class="projects-container">

<div class="project-item">
<div class="project-title">Cancer Diagnosis Classification Project</div>
<div class="project-description">
A data mining project demonstrating supervised learning techniques on the breast cancer diagnostic dataset. Implemented and compared Support Vector Machine and Logistic Regression models to predict whether tumors are malignant or benign. The project includes comprehensive evaluation metrics, cross-validation, and achieves 90%+ accuracy. Built with Python's scientific ecosystem including pandas, scikit-learn, and matplotlib.
</div>
<a href="https://github.com/venkateshterikuti/Cancer-Diagnosis-Classification-Project" target="_blank" rel="noopener" class="project-link">View on GitHub</a>
</div>

<div class="project-item">
<div class="project-title">Optimize Pathfinding in Dynamic Mazes</div>
<div class="project-description">
Developed Deep Q-Network (DQN) and Basic Q-Learning (BQN) algorithms for maze-solving tasks in a Python-simulated environment. Achieved 30% improvement in path efficiency with DQN over BQN in complex mazes and reduced pathfinding steps by 25% across varying maze complexities. Implemented reinforcement learning techniques for dynamic pathfinding optimization, demonstrating advanced AI problem-solving capabilities in navigational challenges.
</div>
<a href="https://github.com/venkateshterikuti/Optimize-Pathfinding-in-Dynamic-Mazes" target="_blank" rel="noopener" class="project-link">View on GitHub</a>
</div>

<div class="project-item">
<div class="project-title">Predicting Customer Churn in E-Commerce</div>
<div class="project-description">
Developed machine learning models to identify at-risk customers for a UK-based online retailer using the Online Retail II dataset. Implemented and compared Random Forest, Support Vector Machine (SVM), and XGBoost algorithms, achieving up to 75% sensitivity with XGBoost and 87% sensitivity with linear SVM. Performed comprehensive feature engineering including TotalPrice and Days Since Last Purchase metrics, with cross-validation showing SVM (RBF) achieving the best average score of ~70%. Included detailed analysis with model-specific feature importance and performance evaluation across accuracy, sensitivity, and specificity metrics.
</div>
<a href="https://github.com/venkateshterikuti/Predicting-Customer-Churn-in-E-Commerce" target="_blank" rel="noopener" class="project-link">View on GitHub</a>
</div>

<div class="project-item">
<div class="project-title">Casino Intelligence Hub</div>
<div class="project-description">
Advanced casino analytics platform analyzing 16M+ transactions to predict player churn, segment users, and detect anomalies. Built an end-to-end pipeline with PostgreSQL for data warehousing, Python (pandas, scikit-learn, XGBoost) for modeling, and interactive dashboards via Streamlit/Power BI. Includes SQL-heavy feature engineering, automated ETL scripts, and a modular architecture for dashboards and model training.
</div>
<a href="https://github.com/venkateshterikuti/casino-intelligence-hub" target="_blank" rel="noopener" class="project-link">View on GitHub</a>
</div>

</div>






