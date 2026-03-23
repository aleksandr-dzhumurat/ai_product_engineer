# ML System design: churn

[https://excalidraw.com/](https://excalidraw.com/)

ML system design task: design churn prediction system

Input: Sales table

| dt | CustomerID | NetSales, $ |
| --- | --- | --- |
| 2025-09-01 | 123 | 90 |
|  |  |  |

Product input: for every 1000 income customers after 30 days only 250 continue using service

# System design interview structure (50-65 minutes)

Design churn prediction system

### Phase 1: Problem Understanding & Requirements (5-10 minutes)

- Define churn for this business context (subscription cancellation, usage drop, etc.)
- What's the prediction window? (30 days, 90 days, 1 year ahead)
- What's the business impact of false positives vs false negatives?
- Scale requirements: How many users? Prediction frequency?
- Latency requirements: Real-time vs batch predictions?
- What success metrics matter most to the business?
    - AB experiment design

## Phase 2: TRAINING

### Data & Features (15-20 minutes)

- Data Sources Discussion
- User Demographics: Age, location, subscription tier, tenure
- Behavioral Data: Login frequency, feature usage, support tickets
- Transactional Data: Payment history, billing issues, upgrades/downgrades
- Engagement Metrics: Session duration, clicks, content consumption
- External Data: Seasonality, competitor actions, economic indicators
- Feature Engineering Deep Dive:
- Time-based aggregations (7d, 30d, 90d averages)
- Trend features (usage declining vs stable)
- Ratio features (recent vs historical usage)
- Categorical encodings for high-cardinality features
- Handling missing values and data quality issues

### Model Selection & Training (10-15 minutes)

Model Approach Discussion:

- Traditional ML: Logistic Regression, Random Forest, XGBoost and Ensemble Methods
- Deep Learning: Neural networks for complex interaction patterns

Training Strategy:

- Data Splitting: Time-based splits to prevent data leakage
- Class Imbalance: SMOTE, class weights, cost-sensitive learning
- Feature Selection: Correlation analysis, feature importance
- Hyperparameter Tuning: Grid search, Bayesian optimization
- Cross-validation: Time series cross-validation strategies

Technical Implementation:

- Training infrastructure scaling
- Experiment tracking (MLflow, Weights & Biases)
- Model versioning and reproducibility
- Distributed training considerations

## Phase 3: VALIDATION

### Model Evaluation (10-15 minutes)

Metrics Discussion:

- Primary: Precision, Recall, F1-score, AUC-ROC
- Business: Customer Lifetime Value impact, retention cost ROI
- Temporal: Performance stability over time
- Segment-wise: Performance across user cohorts

Validation Strategy:

- Temporal Validation: Out-of-time testing
- A/B Testing Framework: Treatment vs control design
- Bias Detection: Fairness across demographic groups
- Confidence Intervals: Statistical significance testing

Model Debugging:

- Feature importance analysis
- Prediction explanation (SHAP, LIME)
- Error analysis by user segments
- Data drift detection methods

## Phase 4: INFERENCE

### Deployment & Serving (10-15 minutes)

```shell
¯\_(ツ)_/¯
```

### Monitoring & Maintenance (5-10 minutes)

```shell
¯\_(ツ)_/¯
```


# Recsys ML design

# Input data

> sales table
> 
> | Datetime | UserID | CustomerID | ProductID | NetPrice |
> | --- | --- | --- | --- | --- |
> | 2024-06-08 14:23:15 | 1001 | 450123 | P789234 | 29.99 |
> | 2024-06-08 15:47:32 | 1002 | 678945 | P456891 | 45.50 |
> | 2024-06-08 16:12:08 | 1003 | 234567 | P123456 | 18.75 |
> | 2024-06-08 17:35:44 | 1001 | 891234 | P654321 | 62.30 |
> | 2024-06-08 18:56:21 | 1004 | 345678 | P987654 | 33.25 |

Questions

Q1: How would you reformulate this problem into a machine learning task? Would it be classification, regression, or something else?

Hints for intervievier:

- Think in terms of predicting clicks/purchases.
- Consider binary classification (e.g., will the user click?) or ranking (e.g., recommend top N items).

Q2: How would you transform the given interaction table into a training set suitable for a machine learning model?
Hints

- Describe features: user-level, item-level, and interaction-level.
- Explain how to create positive and negative samples.
- Consider time windows and implicit feedback.

**Bonus task:**

Write a snippet in **Polars** (or pandas) that prepares a basic train set from the raw data.

Q3: ML sys design

Once you’ve trained your model, how would you serve real-time or batch recommendations to end users?

- Discuss architecture: batch vs real-time.
- Cover components like feature store, model server, candidate generation, filtering, and ranking.
- Mention caching, latency, and scalability concerns.