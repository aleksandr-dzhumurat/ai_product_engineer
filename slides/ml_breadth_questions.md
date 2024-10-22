# Machine Learning & Statistics: 28 Essential Questions

## Q1 — Law of Large Numbers vs Central Limit Theorem

Quick Answer

**LLN** tells us WHERE the sample mean goes (→ population mean).  
**CLT** tells us HOW it gets there (→ Normal distribution).

### Law of Large Numbers (LLN)

**What it states:** Sample mean converges to expected value as $n \to \infty$

$$\bar{X}_n \to \mathbb{E}[X] \quad \text{when } n \to \infty$$

**Required Conditions:**
- Independence
- Identically distributed
- Finite expected value

**Intuition:** Flip a coin 1,000,000 times → proportion of heads ≈ 0.5

**Key point:** LLN says the mean **stabilizes** but doesn't describe its distribution.

### Central Limit Theorem (CLT)

**What it states:** Sum (or scaled mean) of i.i.d. random variables → Normal distribution

$$\frac{\sum X_i - n\mathbb{E}[X]}{\sqrt{n\text{Var}(X)}} \xrightarrow{d} N(0,1)$$

**Intuition:** Even if X is skewed/discrete, sum of many X's looks like bell curve.

**Requirements:**
- Independent observations
- Identically distributed
- **Finite variance** (excludes Cauchy distribution)
- $n \ge 30$ (rule of thumb)

Key Differences


| Aspect | LLN | CLT |
|--------|-----|-----|
| **About** | Convergence of mean | Shape of distribution |
| **Converges to** | Constant ($\mu$) | Normal distribution $N(0,1)$ |
| **Requires variance** | No (weak law) | Yes |
| **Application** | Frequencies, probability estimation | p-values, confidence intervals |



### Practical Applications in ML

**SGD Convergence (LLN)**

Mini-batch gradients converge to true gradient:

$$\frac{1}{B}\sum_{i=1}^{B} \nabla L(x_i, \theta) \xrightarrow{P} E[\nabla L(X, \theta)]$$

This justifies why SGD finds (local) minima despite noisy estimates.

**SGD Distribution (CLT)**

Explains that SGD iterates follow Gaussian distribution around optimum:

$$\sqrt{n}(\bar{\theta}_n - \theta^*) \xrightarrow{d} N(0, \Sigma_\infty)$$

**Bootstrap Methods**

- LLN: Empirical distribution converges to true distribution
- CLT: Justifies confidence interval construction

**Confidence Intervals**
CLT enables:

$$\bar{X} \pm z_{\alpha/2} \cdot \frac{s}{\sqrt{n}}$$

Valid regardless of population distribution (for large n).

**References:**
- [Law of Large Numbers - Wikipedia](https://en.wikipedia.org/wiki/Law_of_large_numbers)
- [Central Limit Theorem - Wikipedia](https://en.wikipedia.org/wiki/Central_limit_theorem)

---

## Q2 — EDA on Single Feature Dataset

### Metrics to Calculate For Numerical Feature:


- Mean: $\bar{x} = \frac{1}{n}\sum x_i$
- Median: Middle value when sorted
- Mode: Most frequent value
- Variance: $s^2 = \frac{1}{n-1}\sum(x_i - \bar{x})^2$
- Standard Deviation: $s = \sqrt{s^2}$
- IQR (Interquartile Range): $Q_3 - Q_1$
- Min, Max, Range
- Skewness: Measures asymmetry
- Kurtosis: Measures tail heaviness

**Simple case 1**: Effect of Multiplying by Constant k on Variance

If all values multiplied by k:

$$\text{Var}(kX) = k^2 \cdot \text{Var}(X)$$

Example: Multiply by 2 → variance increases by 4×

### Comparing Two Datasets

**Statistical Tests:**
- **t-test:** Compare means (if normally distributed)
- **Mann-Whitney U:** Non-parametric alternative
- **KS test (Kolmogorov-Smirnov):** Compare entire distributions
- **Chi-square:** For categorical data

**Visual Comparison:**
- Overlaid histograms
- Box plots side-by-side
- Q-Q plot (one dist vs another)

If Dataset Contains Only 0s and 1s (CTR - Click-Through Rate) - this is Binary/Bernoulli data (What sort of data distribution would be relevant to flipping heads or tails on a coin), metrics for this case:

- Mean: $p$ (proportion of 1s, CTR rate)
- Variance: $p(1-p)$

**References:**
- [Exploratory Data Analysis - Wikipedia](https://en.wikipedia.org/wiki/Exploratory_data_analysis)
- [Statistical Hypothesis Testing](https://en.wikipedia.org/wiki/Statistical_hypothesis_testing)

---

## Q3 — Correlation Types

| Type X | Type Y | Method to Use |
|--------|--------|---------------|
| **Numerical** | **Numerical** | **Pearson** (linear), **Spearman** (monotonic) |
| **Categorical** | **Categorical** | **Cramér's V**, χ² test |
| **Categorical** | **Numerical** | **Correlation Ratio (η)**, ANOVA |
| **Binary (0/1)** | **Numerical** | **Point-Biserial** |
| **Ordinal** | **Numerical** | **Spearman**, **Kendall** |
| **Categorical** | **Binary target** | **Mutual Information**, **WoE** |

### Pearson Correlation

Use when both variables numerical, relationship is linear

$$r = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum(x_i - \bar{x})^2 \sum(y_i - \bar{y})^2}}$$

**Range:** [-1, 1]
- 1 = perfect positive linear relationship
- 0 = no linear relationship
- -1 = perfect negative linear relationship

**Spearman Correlation**: Use when: Non-linear monotonic relationship, Outliers present, Ordinal data

### Cramér's V (Categorical ↔ Categorical)

$$V = \sqrt{\frac{\chi^2}{n \cdot \min(k-1, r-1)}}$$

**TO DO**: add chi-squared formula

Where:
- χ² = Chi-square statistic
- n = sample size
- k = number of columns
- r = number of rows

**Range:** [0, 1]: 0 = no association, 1 = perfect association


**References:**
- [Point-Biserial Correlation - Wikipedia](https://en.wikipedia.org/wiki/Point-biserial_correlation_coefficient)
- [Cramér's V Tutorial](https://www.spss-tutorials.com/cramers-v-what-and-why/)
- [Correlation and Dependence - Wikipedia](https://en.wikipedia.org/wiki/Correlation_and_dependence)

---

## Q4 — Entropy

Entropy = Measure of Uncertainty

Simple test: How hard is it to guess the next element?
```
Sequence: AAAAAA
Easy to predict → Low uncertainty → Low entropy
---
Sequence: AG4l9Pq!D8...
Hard to predict → High uncertainty → High entropy
```

[deep dive to entropy](../jupyter_notebooks/vol_04_deep_dive_09_trees_boosting.ipynb)

### Shannon Entropy Formula

For discrete random variable X:

$$H(X) = -\sum_{x \in \mathcal{X}} p(x) \log p(x) = E[-\log p(X)]$$

**Convention:** $0 \log 0 = 0$

Units:
- log₂ → **bits**
- ln → **nats**
- log₁₀ → **dits**

**Why Logarithm?**

1. Additivity Property: for independent events A and B: $I(A \cap B) = I(A) + I(B)$

Only logarithm satisfies: $\log(p_A \cdot p_B) = \log p_A + \log p_B$

2. Information Content:
Rare event → High information: $I(x) = -\log p(x)$

3. Bits Interpretation: $\log_2(n)$ = minimum bits needed to distinguish n outcomes

**Properties**

- Non-negative: $H(X) \ge 0$
- Maximum: $H(X) \le \log_2(n)$ for $n$ outcomes
  - Maximum when uniform distribution
- Minimum: $H(X) = 0$ when deterministic
- Conditioning reduces entropy: $H(X|Y) \le H(X)$

### Binary Entropy

For coin flip with p(heads) = p: $H(X) = -p\log_2 p - (1-p)\log_2(1-p)$

Maximum at $p = 0.5$ → 1 bit of entropy

### Applications in ML

**Decision Trees** (Information Gain)

Node Entropy: $H(t) = -\sum p_c \log_2 p_c$

Information Gain:

$$IG(S, A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} H(S_v)$$

Select attribute maximizing IG for each split - Greedy algorithm builds tree

**Random Forest**: Uses entropy in individual tree splits, aggregates for variance reduction

**Gradient Boosting** Loss functions often related to cross-entropy (classification)

**LLMs (Cross-Entropy Loss)**

$$H(p, q) = -\sum_x p(x) \log q(x)$$

Where:
- $p$ = true distribution (one-hot encoded token)
- $q$ = model prediction (softmax output)

Minimizing cross-entropy = Minimizing KL divergence:

$$D_{KL}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}$$

Fundamental Relationship:

$$H(p, q) = H(p) + D_{KL}(p \| q)$$

Since $H(p)$ is constant (true labels), minimizing H(p,q) = minimizing KL divergence

---

## Q5 — Type I and Type II Errors

Definitions

Type I Error (False Positive):
- Reject true $H_0$
- Probability = $\alpha$ (significance level)
- **Example:** Healthy patient diagnosed as sick

Type II Error (False Negative):
- Fail to reject false $H_0$
- Probability = $\beta$
- **Example:** Sick patient diagnosed as healthy

Confusion Matrix Analogy

|  | $H_0$ True | $H_0$ False |
|--|---------|----------|
| **Reject $H_0$** | Type I Error ($\alpha$) | Correct ✓ |
| **Fail to Reject $H_0$** | Correct ✓ | Type II Error ($\beta$) |


**Power of a Test** = Probability of correctly rejecting false $H_0$: $\text{Power} = 1 - \beta$

#### Which Error is More Serious - it depends on context.

Type I more serious

Criminal Justice - Convicting innocent person
- "Better 10 guilty go free than 1 innocent convicted"
- **Solution:** High burden of proof (beyond reasonable doubt)

Spam Filter - Marking important email as spam
- Might miss critical communication
- **Solution:** Conservative threshold


Type II more serious

Medical Screening - missing a sick patient
- Disease progresses untreated
- **Solution:** Set low threshold (high sensitivity/recall)

Security Screening - Missing a threat
- Security breach
- **Solution:** Sensitive detection (accept more false alarms)


Tradeoff: fundamental relationship:
- Decreasing $\alpha$ (Type I) → Increases $\beta$ (Type II)
- More strict threshold → Fewer false positives, more false negatives

How to decrease both:
- **Increase sample size** n
- Use more powerful test
- Reduce noise in data

### Optimal Threshold Selection

**ROC Curve:** Plots TPR vs FPR at various thresholds
- Helps visualize tradeoff
- Choose threshold based on cost of errors

Cost-Sensitive Learning: set threshold to minimize total cost.

$$\text{Cost} = C_{FP} \cdot FP + C_{FN} \cdot FN$$

---

## Q6 — Linear Regression

Model in Matrix Form

$$\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}$$

Assumptions (LINE)

| Assumption | Description | Diagnostic |
|------------|-------------|------------|
| **L**inearity | Linear relationship | Residual vs fitted plot |
| **I**ndependence | Independent observations | Durbin-Watson test |
| **N**ormality | Normal errors | Q-Q plot |
| **E**qual variance | Homoscedasticity | Breusch-Pagan test |

Coefficient Interpretation

- $\beta_0$ (Intercept): Expected y when all predictors = 0
- $\beta_j$ (Slope): Expected change in y for one-unit increase in x_j, holding others constant

### Optimization Methods

#### OLS (Closed-Form)

$$\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

Complexity: O(np² + p³)

Use when: n < 10,000, p < 1,000

#### Gradient Descent

Use when: Very large datasets. Variants: Batch GD, Stochastic GD, Mini-batch GD

$$\boldsymbol{\beta}^{(t+1)} = \boldsymbol{\beta}^{(t)} - \eta \nabla_{\boldsymbol{\beta}} J$$

where $J = \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2$ (Sum of Squared Errors):

$$\nabla_{\boldsymbol{\beta}} J = -2\mathbf{X}^T(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})$$

*(Note: In practice, libraries often optimize Mean Squared Error $J = \frac{1}{n}\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2$ so the learning rate $\eta$ does not need scaling by $n$, which introduces a $\frac{1}{n}$ factor to the gradient.)*


## Q7 — Regularization: L1 vs L2

Why Regularization - **without regularization, models can:**
- Memorize training data (overfitting)
- Learn noise instead of patterns
- Have unstable, large weights
- Poor generalization to new data


### L1 (Lasso) Regularization

Loss Function:

$$\min \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda\|\boldsymbol{\beta}\|_1 \quad \text{or} \quad \text{Loss} + \lambda \sum_j |\beta_j|$$

Gradient Penalty:

$$\frac{\partial}{\partial \beta_j} = \lambda \cdot \text{sign}(\beta_j)$$

Constant gradient (magnitude $\lambda$) regardless of weight size.

Effect:
- Produces sparse solutions (some $\beta_j = 0$)
- Automatic feature selection
- Interpretability: Fewer features in final model
- No closed-form solution (Iterative optimization required)

### L2 (Ridge) Regularization

Loss Function:

$$\min \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda\|\boldsymbol{\beta}\|_2^2 \quad \text{or} \quad \text{Loss} + \lambda \sum_j \beta_j^2$$

Gradient Penalty:

$$\frac{\partial}{\partial \beta_j} = 2\lambda \beta_j$$

Proportional gradient (decreases as $\beta_j \to 0$).

Closed-form solution:

$$\hat{\boldsymbol{\beta}}_{ridge} = (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$$

Effect:
- Shrinks coefficients toward zero
- Rarely exactly zero
- Handles multicollinearity well
- All features remain in model

### ElasticNet (L1 + L2)

Loss Function:

$$\min \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda_1\|\boldsymbol{\beta}\|_1 + \lambda_2\|\boldsymbol{\beta}\|_2^2$$

Best of both worlds - combines:
- Feature selection (L1)
- Stability and multicollinearity handling (L2)
- Groups correlated variables

### WHY L1 Produces Sparsity, L2 Doesn't

**Algebraic Intuition**

L1: Constant "push" toward zero: $\lambda \cdot \text{sign}(\beta_j)$ - Even tiny weights get same push $\to$ reach zero.

L2: Proportional "pull": $2\lambda \beta_j$ - Small weights get tiny pull $\to$ rarely reach zero.

**Geometric Intuition**

Constraint regions:
- L1: Diamond shape (has corners at axes)
- L2: Circle shape (smooth, no corners)

Optimization:
- Contours of loss function intersect L1 diamond **often at corners** (where $\beta_j=0$ for some dimensions)
- Contours rarely hit exactly on an axis for L2 circle

Visual: L1's sharp corners encourage solutions where many weights = 0


### Comparison Table

| Property | L1 (Lasso) | L2 (Ridge) |
|----------|------------|------------|
| **Sparsity** | Yes (zeros out weights) | No (shrinks, rarely zero) |
| **Feature Selection** | Yes | No |
| **Multicollinearity** | Poor | Excellent |
| **Interpretability** | High (fewer features) | Lower |
| **Optimization** | Iterative (no closed form) | Closed form solution |
| **Gradient** | Constant magnitude | Proportional to weight |

Use L1 when:
- Want automatic feature selection
- Many irrelevant features
- Interpretability is crucial
- Sparse models needed (memory/speed)

Use L2 when:
- All features potentially useful
- Multicollinearity present
- Smooth weight distribution preferred

Use ElasticNet when:
- Best of both needed
- High-dimensional data with groups of correlated features

**References:**
- [Regularization in Machine Learning - Wikipedia](https://en.wikipedia.org/wiki/Regularization_(mathematics))
- [Lasso vs Ridge - Towards Data Science](https://towardsdatascience.com/ridge-and-lasso-regression-a-complete-guide-with-python-scikit-learn-e20e34bcbf0b)

---

## Q8 — Linear Regression Metrics

For deeper understanding: [cracking_linear_regression](cracking_linear_regression.md)

### Mean Absolute Error (MAE)

When outliers should not dominate, want interpretable metric

$$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

Advantages:
- Easy to interpret (same units as target)
- Robust to outliers
- All errors weighted equally

Disadvantages:
- Not differentiable at zero (optimization harder)
- Less sensitive to large errors


### Root Mean Squared Error (RMSE)

Mean Squared Error (MSE)

$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

Advantages:
- Differentiable everywhere
- Penalizes large errors heavily
- Widely used, well-understood
- Differentiable

Disadvantages:
- Units are squared (hard to interpret)
- Sensitive to outliers

***Root Mean Squared Error (RMSE)*** - Most common regression metric

$$\text{RMSE} = \sqrt{\text{MSE}}$$

Advantages: Same units as target (interpretable)

### Mean Absolute Percentage Error (MAPE)

When relative error matters more than absolute

$$\text{MAPE} = \frac{100\%}{n}\sum_{i=1}^{n}\left|\frac{y_i - \hat{y}_i}{y_i}\right|$$

Advantages:
- Scale-independent (can compare across datasets)
- Easy to interpret (percentage)

Disadvantages:
- Undefined when $y=0$
- Asymmetric: Penalizes over-predictions more than under-predictions
- Biased toward under-predictions

### Huber Loss (Robust Alternative)

$$L_\delta(y, \hat{y}) = \begin{cases}
\frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\
\delta(|y - \hat{y}| - \frac{1}{2}\delta) & \text{otherwise}
\end{cases}$$

Behavior:
- Quadratic for small errors (like MSE)
- Linear for large errors (like MAE)
- **Robust to outliers** while still differentiable

### Metric Selection Guide

| Scenario | Recommended Metric |
|----------|-------------------|
| Outliers present | MAE, Huber |
| No outliers, want differentiability | MSE, RMSE |
| Need scale-independence | MAPE (if no zeros), R² |
| Large errors very costly | MSE, RMSE |
| Model comparison | R², Adjusted R² |
| Want variance explained | R², Adjusted R² |

**References:**
- [Regression Validation Metrics - scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics)

---

## Q9 — Classification Algorithms

Algorithm Comparison


| Algorithm | Training Speed | Interpretability | Noise Resistance | Data Requirements |
|-----------|---------------|------------------|------------------|-------------------|
| Logistic Regression | Fast | High | Medium | Low |
| Decision Tree | Fast | Very High | Low | Low |
| Random Forest | Medium | Medium | High | Medium |
| Gradient Boosting | Slow | Medium | Medium | Medium-High |
| Neural Networks | Very Slow | Very Low | Medium | Very High |
| KNN | None | Medium | Low | Low |
| Naive Bayes | Very Fast | High | Medium | Low |
| SVM | Slow | Low | Low | Medium |


### Logistic Regression

Advantages:
- Highly interpretable (coefficients = log-odds)
- Well-calibrated probabilities
- Robust with regularization (L1/L2)
- Fast training and prediction O(n)
- Well-calibrated probability outputs

Models probability using sigmoid function:

$$P(y=1|x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1x_1 + ... + \beta_nx_n)}}$$

Loss Function (Log-Loss):

$$J(\beta) = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i \log(p_i) + (1-y_i)\log(1-p_i)\right]$$

Regularized Versions:
- L2 (Ridge): $J(\beta) + \lambda\|\beta\|_2^2$ - shrinks coefficients, handles multicollinearity
- L1 (Lasso): $J(\beta) + \lambda\|\beta\|_1$ - produces sparse solutions, feature selection
- ElasticNet: $J(\beta) + \lambda_1\|\beta\|_1 + \lambda_2\|\beta\|_2^2$ - combines both
*Disadvantages:
- Cannot capture non-linear relationships
- Sensitive to outliers without regularization
- Requires feature scaling

Best For binary classification with interpretability requirements, high-dimensional sparse data

### Decision Trees

Splitting Criteria:

- Gini: $G = 1 - \sum p_i^2$
- Entropy: $H = -\sum p_i \log_2(p_i)$

Advantages:
- Highly interpretable (visual tree)
- No feature scaling needed
- Handles mixed data types (numerical and categorical)
- Captures non-linear relationships

Disadvantages:
- Prone to overfitting (high variance)
- Unstable (small data change = different tree)
- Cannot extrapolate
- Greedy algorithm (may not find global optimum)

### Random Forest

Advantages:
- Reduces overfitting vs single tree (variance reduction)
- Feature importance scores
- Works well "out of the box"
- Robust to outliers
- Handles missing values
- Parallelizable

Disadvantages:
- Less interpretable than single tree
- Slower inference than linear models
- Memory intensive

#### Gradient Boosting (XGBoost, LightGBM, CatBoost)

Advantages:
- Often achieves best accuracy
- Built-in regularization
- Handles missing values
- Feature importance

Disadvantages:
- Prone to overfitting without tuning
- Sequential training (slower, not parallelizable)
- Many hyperparameters to tune
- Less interpretable

### KNN (K-Nearest Neighbors)

Advantages:
- Simple, intuitive
- No training phase
- Non-parametric (no assumptions)

Disadvantages:
- Slow prediction O(n×d)
- Curse of dimensionality
- Sensitive to irrelevant features
- Requires distance metric choice

### Naive Bayes

Advantages:
- Very fast training and inference
- Works well with small datasets
- Good for text classification
- Handles high dimensions well

Disadvantages:
- Strong independence assumption (rarely true)
- Poor probability estimates

### Neural Networks

Advantages:
- Universal function approximator
- Automatic feature learning
- State-of-the-art on unstructured data (images, text, audio)
- Transfer learning possible

Disadvantages:
- "Black box" - low interpretability
- Requires large datasets
- Computationally expensive
- Many hyperparameters
- Prone to overfitting without regularization


### SVM (Support Vector Machine)

Advantages:
- Effective in high dimensions
- Memory efficient (uses only support vectors)
- Kernel trick for non-linear boundaries

Disadvantages:
- Slow training O(n² to n³)
- Requires feature scaling
- Kernel selection is tricky
- Not well-calibrated probabilities

**Best For:** Text classification, high-dimensional data with clear margins

---

## Q10 — Classification Metrics & ROC-AUC vs F1

### Core Metrics

Confusion Matrix

|  | Predicted Positive | Predicted Negative |
|--|-------------------|-------------------|
| **Actual Positive** | True Positive (TP) | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN) |


**Accuracy:**

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

Limitation: Misleading on imbalanced data—95% accuracy means nothing if 95% of data is one class


**Precision:**

$$\text{Precision} = \frac{TP}{TP + FP}$$

Intuition: "When model predicts positive, how often is it correct?"

Use when: False positives are costly (spam filter, drug approval)


**Recall (Sensitivity):** "Of all actual positives, how many did we catch?"

$$\text{Recall} = \frac{TP}{TP + FN}$$

Use when: False negatives are costly (cancer screening, fraud detection)

**Specificity**

$$\text{Specificity} = \frac{TN}{TN + FP}$$

Use when: Correctly identifying negatives matters (confirming absence of disease)

**F1-Score:**: Harmonic mean (penalizes extreme values)

$$F_\beta = (1 + \beta^2) \times \frac{\text{Precision} \times \text{Recall}}{(\beta^2 \times \text{Precision}) + \text{Recall}}$$

- β < 1: More weight to precision
- β > 1: More weight to recall


*Why F-beta(F1) is Needed*:
- Precision and recall have inverse relationship
- Single balanced score for model comparison
- Essential for imbalanced datasets


### Advanced metrics

#### ROC-AUC

ROC Curve: Plots TPR (y-axis) vs FPR (x-axis) across all thresholds

Probabilistic interpretation: Probability that random positive ranks higher than random negative

AUC (Area Under Curve): $AUC = 1.0$: Perfect classifier, $AUC = 0.5$: Random guessing, $AUC \ge 0.9$ : Excellent.

#### PR-AUC vs ROC-AUC

Key Difference:
- ROC uses TN in FPR calculation
- PR-AUC ignores TN entirely

When PR-AUC is Better:
- Highly imbalanced datasets (positive class < 10%)
- When true negatives aren't meaningful
- Fraud detection, anomaly detection

#### ROC-AUC vs F1

| Aspect | ROC-AUC | F1-Score |
|--------|---------|----------|
| **Threshold** | Threshold-independent | Threshold-dependent |
| **Imbalanced data** | Can be misleading | Better for imbalanced |
| **What it measures** | Ranking ability | Precision-recall balance |
| **Uses TN** | Yes (in FPR) | No |
| **Best for** | Model comparison, ranking | Specific threshold, imbalanced data |

When to Use What: decision Guide

| Scenario | Recommended Metric |
|----------|-------------------|
| Balanced classes, equal costs | Accuracy, ROC-AUC |
| Imbalanced, positive class matters | PR-AUC, F1 |
| FP costly (spam filter) | Precision, F0.5 |
| FN costly (cancer screening) | Recall, F2 |
| Model comparison, ranking ability | ROC-AUC |
| Threshold-specific performance | F1 at threshold |

---

Drawbacks of ROC-AUC: it does not account for the costs of Type I and Type II errors. Instead, one should use cost-sensitive precision or cost-sensitive recall.

* Accuracy: Balanced classes, equal costs
* Precision Minimize false positives (spam filter, drug approval)
* Recall: Minimize false negatives (cancer screening, fraud detection)
* F1: Balance precision and recall, imbalanced data
* Macro F1: Multi-class, treat all classes equally
* Micro F1: Multi-class, weight by class frequency

**ROC-AUC:** 
- Model comparison
- When you care about ranking/probability
- Balanced datasets

**PR-AUC (Precision-Recall AUC):**
- Highly imbalanced datasets (positive < 10%)
- When true negatives are not meaningful

## Q11 — Handling Class Imbalance

Methods:
1. Resampling:
   - SMOTE (oversampling minority)
   - Random undersampling majority

2. Class Weights:
   - Penalize minority class errors more
   - Most sklearn classifiers support `class_weight='balanced'`

3. Threshold Adjustment:
   - Lower threshold to increase recall
   - Optimize threshold using validation set

4. Ensemble Methods:
   - Balanced Random Forest
   - EasyEnsemble, BalancedBagging

5. Use Appropriate Metrics:
   - F1-score instead of accuracy
   - PR-AUC instead of ROC-AUC

**References:**
- [Confusion Matrix - Wikipedia](https://en.wikipedia.org/wiki/Confusion_matrix)
- [Receiver Operating Characteristic - Wikipedia](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)

---

## Q12 — Handling Unknown Categories in Production

Problem: Model encounters category it never saw during training.

Example:
- Training: categories [A, B, C]
- Production: new category D appears

Training
- Monitor: Track frequency of unknown categories  
- Retrain: Periodically retrain with new categories  
- Alert: Set up alerts when unknowns exceed threshold  
- A/B Test: Compare different handling strategies  

### 1. OneHotEncoder with `handle_unknown='ignore'`

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(handle_unknown='ignore')
```

Behavior: Unknown category → all zeros vector
- Effectively treats as "neutral"
- Model uses only other features

### 2. Fallback to Most Frequent Category

During inference:
```python
if category not in known_categories:
    category = most_frequent_category
```

When to use: When categories are similar enough

### 3. Add 'Unknown' Category During Training

Strategy:
- During training, add synthetic "unknown" category
- Assign small fraction of training data to it
- OR: Hold out some categories as "unknown"

Benefit: Model learns to handle unknowns explicitly

### 4. Target Encoding with Global Mean Fallback

Target Encoding:
```python
category_means = df.groupby('category')['target'].mean()
```

For unknown category:
```python
if category in category_means:
    value = category_means[category]
else:
    value = global_mean  # Fallback
```

### 5. Embedding-Based Approach

For high-cardinality categoricals:
- Use entity embeddings (neural networks)
- Approximate unknown with nearest known embedding
- OR: Use hash trick to map to fixed space

### 6. Hierarchical Categories

If categories have hierarchy:
```
Product → Category → Subcategory
"New Phone Model" → "Electronics" → "Phones"
```

Fallback to parent category if specific subcategory unknown


**References:**
- [OneHotEncoder Documentation - sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
- [Entity Embeddings of Categorical Variables](https://arxiv.org/abs/1604.06737)

---

## Q13 — Gradient Boosting

[Gradient boosting recap](../jupyter_notebooks/vol_04_deep_dive_09_trees_boosting.ipynb)

The gradient descent update rule is:

$$\theta_{new} = \theta_{old} - \eta \cdot \nabla L(\theta)$$

Where:
- $\theta_{new}$ is the updated parameter value
- $\theta_{old}$ is the current parameter value
- $\eta$ is the learning rate
- $\nabla L(\theta)$ is the gradient of the loss function with respect to $\theta$

Если функция потерь зависит от нескольких параметров $L(\theta_1, \ldots, \theta_n)$ то градиент - это вектор


$$\nabla L = \left(\frac{\partial L}{\partial \theta_1}, \frac{\partial L}{\partial \theta_2}, \ldots, \frac{\partial L}{\partial \theta_n}\right)$$

Каждая компонента показывает, насколько чувствительна ошибка к изменению конкретного параметра

The **gradient** is a mathematical concept that indicates the direction and rate of steepest increase of a function.

Now about boosting: core principle - **Sequentially add weak learners**, each correcting errors of previous ensemble, Additive Model Principle

$$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$

Where:
- $F_m$ = ensemble after m trees
- $\eta$ = learning rate (shrinkage)
- $h_m$ = new tree fitted to pseudo-residuals

Algorithm

**1. Initialize:** $F_0(x) = \text{constant}$ (e.g., mean for regression)

For MSE:

$$F_0(x) = \mathbb{E}[y]$$

For log-loss:

$$F_0(x) = \log\left(\frac{p}{1 - p}\right), \quad \text{where } p = \mathbb{E}[y]$$

**2. For m = 1 to M:**
   - **Compute pseudo-residuals:** 

$$r_{im} = -\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\bigg|_{F=F_{m-1}}$$
   
   - Fit tree $h_m$ to residuals $r_{im}$
   - Update: $F_m = F_{m-1} + \eta \cdot h_m$

**3. Output:** $F_M(x)$

### Losses

**For MSE Loss**

Loss: $L(y, F) = \frac{1}{2}(y - F)^2$

Pseudo-residual: $r = y - F_{m-1}(x)$ (ordinary residual)

Intuition: Fit next tree to what we got wrong so far

**For Log-Loss (Classification)**

Loss: $L(y, F) = -[y\log(p) + (1-y)\log(1-p)]$ where $p = \sigma(F)$

Pseudo-residual: $r = y - \sigma(F_{m-1}(x))$

### Gradient boosting Regularization Techniques

#### 1. Learning Rate (Shrinkage)

$$F_m = F_{m-1} + \eta \cdot h_m, \quad 0 < \eta \leq 1$$

Smaller $\eta$:
- Need more trees
- Better generalization
- **Typical:** 0.01 - 0.3

#### 2. Tree Constraints

- Max depth: 3-8 (shallow trees prevent overfitting)
- Min samples per leaf
- Max leaf nodes

#### 3. Subsampling (Stochastic GB)

 Reduces overfitting, adds randomness

- Row sampling: Use random 50-80% of data per tree
- Column sampling: Use random features per tree/split

#### 4. XGBoost Regularization

$$\Omega(h) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2$$

Where:
- $T$ = number of leaves, penalizes tree size
- $w_j$ = leaf weights
- $\gamma$ = complexity penalty, discourages adding new leaves: split must reduce loss by at least $\gamma$
- $\lambda$ = L2 on weights, shrinks predictions and prevents extreme leaf outputs

### Gradient Boosting vs Random Forest

| Aspect | GB | RF |
|--------|----|----|
| **Training** | Sequential | Parallel |
| **Reduces** | Bias | Variance |
| **Base learners** | Shallow trees (3-8) | Deep trees |
| **Overfitting risk** | Higher | Lower |
| **Accuracy** | Often better | Good |
| **Speed** | Slower training | Faster training |

* When to use GB: Maximum accuracy, willing to tune  
* When to use RF: Quick baseline, less tuning needed

### Modern Implementations

XGBoost:
- Uses gradient + Hessian (2nd derivative)
- Regularized objective
- Handles missing values
- Histogram-based splits

LightGBM:
- Leaf-wise growth (faster but riskier)
- GOSS (Gradient-based One-Side Sampling)
- EFB (Exclusive Feature Bundling)

CatBoost:
- Ordered boosting (prevents leakage)
- Native categorical handling
- Symmetric trees

---

## Q14 — Bias-Variance Tradeoff

* [bias-variance decomposition](https://education.yandex.ru/handbook/ml/article/bias-variance-decomposition)
* [bias-variance tradeoff explained](https://www.linkedin.com/feed/update/groupPost:961087-7266231655042838528)

**Mathematical Decomposition:**

$$\text{Expected Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

For squared error loss, the expected prediction error decomposes as:

$$\text{EPE}(x) = \underbrace{(f(x) - E[\hat{f}(x)])^2}_{\text{Bias}^2} + \underbrace{E[(\hat{f}(x) - E[\hat{f}(x)])^2]}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Irreducible}}$$

Where:
- **Bias²:** Error from **wrong assumptions** (how far average prediction is from the true function).
- **Variance:** Error from **sensitivity to small changes** (how much predictions vary across different training sets).
- **Irreducible Error:** Inherent noise in the data ($\sigma^2$).

### High Bias vs High Variance

**High Bias (Underfitting):**
- Model is too simple to capture patterns.
- Poor performance on **both training and test sets**.
- *Examples:* Linear regression on non-linear data, logistic regression with insufficient features, naive bayes, shallow decision trees.

**High Variance (Overfitting):**
- Model is too complex and fits noise in the training data.
- Very good performance on **training data**, but poor generalization on **test data**.
- *Examples:* Deep decision trees without pruning, KNN with k=1, neural networks without regularization.

| Model Complexity | Bias | Variance |
|------------------|------|----------|
| Simple model     | High | Low      |
| Complex model    | Low  | High     |

The goal is to find the sweet spot where expected error is minimized. 

### Managing Tradeoff

| Goal | Techniques |
|------|-----------|
| **Reduce Bias** | Add features, increase model capacity/complexity, reduce regularization. |
| **Reduce Variance** | Add regularization (L1/L2), reduce features, get more training data, use ensemble methods (e.g., bagging). |

**In practice:**
- Use **cross-validation** to estimate generalization error.
- Visualization tools like **learning curves** can show if you're dealing with high bias or high variance.
- *Note:* A single deep decision tree has low bias but high variance. A **Random Forest** uses bagging to dramatically lower this variance while keeping bias low.

### Ensemble Methods

#### Bagging (Reduces Variance)

Train multiple models on bootstrap samples, average predictions

$$\hat{f}_{bag}(x) = \frac{1}{B}\sum_{b=1}^{B} \hat{f}_b(x)$$

Variance Reduction:

$$\text{Var}(\hat{f}_{bag}) = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$$


Key Insight: Averaging reduces variance without increasing bias. Works best with high-variance base learners (deep trees).

**Random Forest** enhancement:
- Feature subsampling at each split reduces correlation ρ between trees
- Typically use √p features for classification, p/3 for regression
- Further variance reduction compared to plain bagging

Out-of-Bag (OOB) Error: Each tree is trained on ~63% of data, can validate on remaining ~37% without separate validation set.


#### Boosting (Reduces Bias)

Sequentially train models, each correcting previous errors. Primarily reduces bias; can increase variance if over-boosted.

Examples: Gradient Boosting, AdaBoost


Bias Reduction Mechanism:
- Start with high-bias model (shallow tree)
- Each iteration fits pseudo-residuals (errors of current ensemble)
- Systematically corrects where current model performs poorly
- Gradually builds complex decision boundary


Key Difference from Bagging:
- Bagging: Independent models, reduces variance
- Boosting: Dependent models, reduces bias


#### Stacking

Train meta-learner on predictions of base models

Architecture:
- Level-0: Train multiple diverse base models h₁(x), ..., h_K(x)
- Level-1: Train meta-learner g on Level-0 predictions

$$\hat{y}_{stack} = g(h_1(x), h_2(x), ..., h_K(x))$$

Avoiding Overfitting:
Use K-Fold cross-validation to generate out-of-fold predictions for meta-learner training.

**Critical:** Never train meta-learner on same predictions used to train base models.

Process:
1. Split training data into K folds
2. For each base model:
   - Train on K-1 folds, predict on holdout fold
   - Repeat K times to get out-of-fold predictions for entire training set
3. Train meta-learner on concatenated out-of-fold predictions
4. For test data: Average predictions from K base models trained on different folds

Base Model Selection:
Diverse models (different algorithms, hyperparameters) work best. Common combinations:
- Linear model + Tree model + Neural network
- XGBoost + LightGBM + CatBoost

Meta-Learner:
Usually simple model (Logistic Regression, Ridge) to avoid overfitting.

### When to Use Each

| Scenario | Recommended Method | Reasoning |
|----------|-------------------|-----------|
| High-variance base models (deep trees) | Bagging / Random Forest | Variance reduction |
| High-bias base models (linear, shallow trees) | Boosting | Bias reduction |
| Diverse model types available | Stacking | Combines different strengths |
| Maximum accuracy, competition | Gradient Boosting + Stacking | State-of-the-art |
| Quick baseline, robust | Random Forest | Works out-of-box |
| Need fast training | Random Forest | Parallelizable |



### High Bias vs High Variance

**High Bias, Low Variance (Underfitting):**
- Linear regression on non-linear data
- Shallow decision trees
- Strongly regularized models (high λ)
- Symptom: High train AND test error
- Solution: Increase model complexity, reduce regularization

**Low Bias, High Variance (Overfitting):**
- Deep unpruned decision trees
- KNN with k=1
- High-degree polynomial regression
- Neural networks without regularization
- Symptom: Low train error, high test error
- Solution: Regularization, more data, reduce complexity

---

## Q15 — Recommendation Systems

Goal: Predict user preference for items not yet interacted with

Interaction Matrix R:
- Rows: Users (m users)
- Columns: Items (n items)
- Entries: Ratings, clicks, purchases, or implicit feedback

Challenges:
- Sparsity: Typically >99% missing values
- Cold start: New users/items with no history
- Scalability: Millions of users × items

### Core Approaches

Collaborative Filtering
- User-based: Find similar users
- Item-based: Find similar items
- Matrix Factorization: Latent factor models

Content-Based
- Recommend items similar to those user liked
- Based on item features

Method: Cosine similarity between item feature vectors

Advantages:
- No cold start for new users (recommend based on current interaction)
- Transparent explanations
- Stable over time

Disadvantages:
- Limited diversity (filter bubble)
- Requires item features

Hybrid: Combine collaborative + content-based

### Key RecSys Algorithms

#### K-Nearest Neighbors (KNN)

* User-based: Find K most similar users, predict rating as weighted average
* Item-based: Find K most similar items user rated, predict as weighted average
* Similarity: Cosine, Pearson correlation, Jaccard

Pros: Simple, interpretable  
Cons: Doesn't scale, sensitive to sparsity


#### Matrix Factorization

$$\hat{r}_{ui} = \mu + b_u + b_i + \mathbf{p}_u^T \mathbf{q}_i$$

Where:
- μ = global mean
- b_u, b_i = user/item bias
- p_u, q_i = latent factors

Loss:
$$\min \sum_{(u,i) \in observed} (r_{ui} - \hat{r}_{ui})^2 + \lambda(||p_u||^2 + ||q_i||^2)$$

#### ALS (Alternating Least Squares)

Algorithm:
1. Fix Q, solve for P (closed-form)
2. Fix P, solve for Q (closed-form)
3. Alternate until convergence

Pros: Parallelizable, handles implicit feedback  
Cons: Assumes all missing = negative

#### BPR (Bayesian Personalized Ranking)

Objective: Maximize pairwise ranking

$$\max \sum_{(u,i,j)} \log \sigma(\hat{r}_{ui} - \hat{r}_{uj})$$

Where:
- i = positive item (observed)
- j = negative item (unobserved)

Better for implicit feedback (clicks, views)


#### Session-Based

Predict next action based on current session sequence

Method: 
- RNNs (GRU, LSTM) for sequence modeling
- Transformers (self-attention over session)
- Graph Neural Networks (session as graph)

Use Cases:
- E-commerce browsing
- Music streaming (next song)
- Video recommendations

Advantages:
- Captures short-term intent
- No user profile needed
- Works for anonymous users

### Cold Start Solutions

| Problem | Approach |
|---------|----------|
| **New users** | Popularity-based recommendations |
| | Ask for initial preferences |
| | Use demographic information |
| **New items** | Content-based features |
| | Expert curation |
| | Explore-exploit strategies |
| **Both** | Hybrid systems |
| | Transfer learning from related domains |


### Metrics


**Precision@K**, interpretation: Of K recommendations, what fraction user liked?

$$\text{Precision@K} = \frac{|\text{Relevant items in top K}|}{K}$$

**Recall@K**, Interpretation: Of all items user would like, what fraction did we recommend?

$$\text{Recall@K} = \frac{|\text{Relevant items in top K}|}{|\text{All relevant items}|}$$

**MAP (Mean Average Precision)**, Interpretation: Rewards placing relevant items higher in ranking
$$\text{MAP@K} = \frac{1}{|U|}\sum_{u=1}^{|U|} \frac{1}{\min(m_u, K)} \sum_{k=1}^{K} P(k) \cdot \text{rel}(k)$$

Where $P(k)$ = precision at position k, rel(k) = 1 if item k is relevant

**NDCG (Normalized Discounted Cumulative Gain)**, Normalized to [0, 1], Penalizes relevant items appearing lower
$$\text{DCG@K} = \sum_{i=1}^{K} \frac{2^{rel_i} - 1}{\log_2(i+1)}$$

$$\text{NDCG@K} = \frac{DCG@K}{IDCG@K}$$

Where IDCG = ideal DCG (perfect ranking)

### Cold Start Problem

New User:
- Popularity-based recommendations
- Ask for initial preferences
- Use demographic info

New Item:
- Content-based features
- Expert curation

---

## Q16 — Text Preprocessing (Sparse Retrieval)

### Preprocessing Pipeline

#### 1. Tokenization
Split text into tokens (words, subwords, characters)
- Whitespace splitting
- Rule-based (language-specific)
- Subword: BPE, WordPiece

#### 2. Lowercasing
`"Python" → "python"`

Tradeoff: Loses information (Apple company vs apple fruit)

#### 3. Cleaning
- Remove HTML tags
- Remove special characters
- Normalize whitespace
- Remove/normalize numbers

#### 4. Stop Words Removal
Remove common words: "the", "is", "a", "an"

Tradeoff:
- ✅ Reduces index size
- ❌ Can hurt phrase queries ("to be or not to be")

#### 5. Stemming / Lemmatization

Stemming (Porter, Snowball):
```
running → run
happiness → happi  # Not a real word!
```

Lemmatization (WordNet):
```
running → run
better → good
geese → goose  # Real words
```

| Aspect | Stemming | Lemmatization |
|--------|----------|---------------|
| Method | Rule-based | Dictionary + POS |
| Speed | Fast | Slow |
| Output | May not be word | Always valid word |
| Accuracy | Lower | Higher |

### Bag of Words (BoW)

Document as word frequency vector

**Properties:**
- Ignores word order
- Ignores semantics
- High-dimensional, sparse
- Simple baseline

### TF-IDF

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

Term Frequency:

$$\text{TF}(t, d) = \frac{f_{t,d}}{\sum_{t'} f_{t',d}}$$

Or: $\text{TF} = \log(1 + f_{t,d})$ (sublinear)

Inverse Document Frequency:

$$\text{IDF}(t) = \log\frac{N}{|\{d : t \in d\}|}$$

Intuition: High TF-IDF = term frequent in doc, rare in corpus

### BM25 (Best Matching 25)

Formula:

$$\text{score}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot (1 - b + b \cdot \frac{|D|}{\text{avgdl}})}$$

Where:
- $q_i$: Query terms
- $f(q_i, D)$: Term frequency of q_i in document D
- $\|D\|$: Document length
- avgdl: Average document length in corpus
- $k_1$: Term frequency saturation parameter (typical: 1.2-2.0)
- $b$: Length normalization parameter (typical: 0.75)

**IDF Component:**

$$\text{IDF}(q_i) = \log \frac{N - n(q_i) + 0.5}{n(q_i) + 0.5}$$

Where N = total documents, $n(q_i)$ = documents containing $q_i$

Key Features:

1. Term Saturation: Diminishing returns for repeated terms
   - 1 occurrence: contributes a lot
   - 100 occurrences: doesn't contribute 100× more

2. Length Normalization: Penalizes long documents
   - Prevents long documents from dominating
   - b=0: No normalization
   - b=1: Full normalization

Improvements over TF-IDF:
- Non-linear term frequency (saturation)
- Better length normalization
- Tunable parameters

Use Case: Standard baseline for text search, used in Elasticsearch, Lucene

For  High-dimensional sparse data Cosine Similarity works good

### Inverted Index

Structure: Term → List of (DocID, metadata)

Example:
```
"cat": [(Doc1, TF=2, positions=[5,12]), (Doc3, TF=1, positions=[7])]
"dog": [(Doc2, TF=1, positions=[3]), (Doc3, TF=3, positions=[2,8,15])]
```

Query Processing:
1. Look up query terms in index
2. Retrieve posting lists
3. Compute scores (BM25, TF-IDF)
4. Rank and return top-K

Optimizations:
- Skip pointers: Jump over documents that can't match
- Compression: Varbyte encoding, delta encoding
- Caching: Frequently queried terms

**Complexity:**  
- Without index: $\( O(N \times M) \)$, where $\( N \)$ = number of documents and $\( M \)$ = average document length  
- With index: $\( O(K \times L) \)$, where $\( K \)$ = number of query terms and $\( L \)$ = average posting list length

---

## Q17 — Word2Vec


| Aspect | CBOW | Skip-gram |
|--------|------|-----------|
| **Input** | Context words | Target word |
| **Output** | Target word | Context words |
| **Speed** | Faster to train | Slower |
| **Rare words** | Underperforms | Better representation |
| **Data needs** | Works with less data | Needs more data |
| **Use case** | Large corpus | Small-medium corpus |


**CBOW:** Predict target word from context  
**Skip-gram:** Predict context from target word

Key difference from BoW:
- BoW: Sparse, high-dimensional, no semantics
- Word2Vec: Dense (50-300d), captures semantic similarity

Semantic properties:
```
king - man + woman ≈ queen
Paris - France + Italy ≈ Rome
```

Limitations:
- Single vector per word (no polysemy)
- Trained on static corpus
- No handling of out-of-vocabulary words

Training Tricks:
- Negative sampling: Instead of softmax over full vocabulary, sample K negative words
- Hierarchical softmax: Binary tree structure for efficient training

---

## Q18 — Vector Search

Map text/images/other data to dense vectors that capture semantic similarity

### Pretrained Models

Text Embeddings:
- Sentence-BERT (SBERT): Siamese architecture, fine-tuned BERT for sentence similarity
- OpenAI embeddings: text-embedding-ada-002, text-embedding-3-small/large
- BGE (BAAI General Embedding): State-of-the-art open-source
- E5: Multilingual, instruction-tuned
- Cohere embed-v3: Supports task-specific compression

Multimodal:
- CLIP: Joint text-image embedding
- BLIP-2: Visual question answering


### Fine-tuning Methods

Contrastive Learning: Loss encourages similar pairs close, dissimilar pairs far

Triplet Loss:

$$L = \max(0, d(a, p) - d(a, n) + \text{margin})$$

Where a=anchor, p=positive, n=negative

Hard Negative Mining: Select challenging negatives (high similarity but irrelevant) for better discrimination

Domain Adaptation: Fine-tune on domain-specific data (medical, legal, scientific)

### Distance Metrics

#### Low dimensions

Euclidean Distance

$$d_2(\mathbf{A}, \mathbf{B}) = ||\mathbf{A} - \mathbf{B}||_2 = \sqrt{\sum_{i=1}^{n}(A_i - B_i)^2}$$

Use when:
- Magnitude is meaningful
- Low-dimensional data (images)
- Clustering (K-means uses Euclidean)

Manhattan Distance (L1)

For the grid-like spaces, robust to outliers

$$d_1(\mathbf{A}, \mathbf{B}) = \sum_{i=1}^{n} |A_i - B_i|$$


#### Cosine Similarity

$$\text{sim}(i, j) = \frac{\mathbf{v}_i \cdot \mathbf{v}_j}{||\mathbf{v}_i|| \cdot ||\mathbf{v}_j||}$$

Range: [-1, 1] (typically [0, 1] for semantic embeddings)

Cosine Distance: $d = 1 - \text{cosine similarity}$

Use when:
- Direction matters more than magnitude
- Text similarity (embeddings usually normalized)

Dot Product - same but without norms. Use when fast computation needed, vectors are normalized

$$\mathbf{A} \cdot \mathbf{B} = \sum_{i=1}^{n} A_i B_i$$

#### Curse of Dimensionality

In high dimensions, geometry breaks down. The main effect, distances converge, the max and min distances become nearly equal. The Nearest Neighbor in 100D Almost as Far as the Farthest Neighbor.

$$\lim_{d \to \infty} \frac{d_{\max} - d_{\min}}{d_{\min}} \to 0$$

Intuition: in a 1000-dimensional space, all points are roughly the same distance from each other. L2 distances stop being informative.

This phenomenon occurs due to data sparsity in high-dimensional space, which makes all distances converge (i.e., the difference between the closest and farthest neighbors becomes negligible). In low dimensions (e.g., 2D or 3D), points are **relatively close together**, and the nearest neighbor is **much closer** than the farthest. However, in **high-dimensional space (100D+), data spreads out exponentially**, causing all distances to become similar.

Key reason:
- As dimensions increase, data points occupy a larger volume.
- The average distance between points increases.
- The difference between the closest and farthest points becomes **negligibly small** relative to the total distance range.


Why cosine handles high dimensionality better:

- Cosine measures the **angle**, not the absolute distance:
- Angular differences between vectors are preserved even in high dimensions (curse of dimensionality)
- L2 suffers more because as $d$ grows, noise accumulates across all dimensions:

$$\|\mathbf{a} - \mathbf{b}\|_2 = \sqrt{\sum_{i=1}^{d}(a_i - b_i)^2}$$

Each additional dimension adds another $(a_i - b_i)^2$ term — mostly noise — pushing all pairwise distances together.

Practical implications:

- More data is needed as the number of features grows — the volume of the space scales as $O(r^d)$, so the data needed to fill it grows exponentially
- KNN becomes ineffective in high $d$ since the nearest and farthest neighbors are almost equidistant
- PCA, UMAP, t-SNE are dimensionality reduction tools designed specifically to fight this


### Vector Databases

| Database | Key Features | Best For |
|----------|-------------|----------|
| **Milvus** | Distributed, GPU support, 10+ indexes | Production, billions of vectors |
| **Pinecone** | Fully managed, serverless | Quick deployment, no ops |
| **Weaviate** | GraphQL API, hybrid search | Complex queries |
| **Qdrant** | Rust-based, payload filtering | High performance, filtering |
| **Redis Vector** | In-memory, low latency | Real-time applications |
| **Chroma** | Embedded, developer-friendly | Prototyping, small scale |

Key Capabilities:
- CRUD operations on vectors
- Metadata filtering
- Hybrid search (dense + sparse)
- Sharding and replication
- ACID transactions (some)

### ANN (Approximate Nearest Neighbor) Indices

For large datasets evalute cosine leads to a high compute costs

#### HNSW (Hierarchical Navigable Small World)

Multi-layer graph, $O(log n)$ average complexity

- Top layer: Sparse, long-range connections
- Bottom layer: Dense, full graph


Parameters:
- M: Max connections per node (typical: 16-64)
- efConstruction: Search depth during build (typical: 200-400)
- efSearch: Search depth during query (typical: 100-500)

Search Algorithm:
1. Start at top layer
2. Greedily navigate to nearest neighbor
3. Drop to next layer
4. Repeat until bottom layer

Pros:
- Excellent recall (>95% with proper parameters)
- Fast queries
- Supports incremental updates

Cons:
- High memory usage (stores full graph)
- Build time can be long


#### IVF (Inverted File Index)

K-means clustering + inverted index, $O(K + nprobe × N/K)$ Complexity


Parameters:
- **K:** Number of clusters (typical: √N to N/100)
- **nprobe:** Clusters to search (typical: 10-100)

1. Cluster vectors into K partitions (k-means)
2. Assign each vector to nearest cluster
3. Create inverted index: ClusterID → VectorList

Search: Only search top-nprobe clusters
1. Find nprobe nearest cluster centroids to query
2. Search only vectors in those clusters
3. Return top-K overall

Pros:
- Good recall-speed tradeoff
- Lower memory than HNSW
- Supports GPU acceleration

Cons:
- Requires training phase
- Sensitive to clustering quality
- Not ideal for updates (need retraining)


#### LSH (Locality Sensitive Hashing)

Similar vectors hash to same bucket

Common LSH Families:
- **Random projection:** For cosine similarity
- **MinHash:** For Jaccard similarity
- **p-stable:** For L_p distances

Algorithm:
1. Create L hash tables with K hash functions each
2. For query, hash into all L tables
3. Retrieve candidates from matching buckets
4. Rank candidates

Pros:
- Sublinear query time O(N^ρ) where ρ < 1
- Constant update time
- Works in very high dimensions

Cons:
- Lower recall than HNSW/IVF
- Needs many hash tables for good recall
- Hash function design is tricky


| Index | Recall | Speed | Memory | Build Time | Updates |
|-------|--------|-------|--------|-----------|---------|
| **HNSW** | Excellent | Excellent | High | Medium | Good |
| **IVF** | Good | Good | Medium | Fast | Poor |
| **LSH** | Moderate | Moderate | Medium | Very Fast | Excellent |
| **Flat (Exact)** | Perfect | Slow | Low | Instant | Excellent |

Guidelines:
- Static data, best recall: HNSW
- Dynamic data, frequent updates: LSH
- Large scale, GPU available: IVF with GPU
- <100K vectors: Consider flat search

### Embedding Optimization Techniques

#### 1. Dimensionality Reduction

* PCA (Principal Component Analysis): Project to top-k principal components
* UMAP/t-SNE: Non-linear reduction, preserves local structure. Use case: Reduce 1536d to 384d with <5% quality loss


#### 2. Quantization

Scalar quantization: INT8/INT4 instead of FP32:
- 2-8× size reduction
- Faster computation (INT ops)
- Minimal quality loss (<1% typically)

Process:
1. Find min/max of each dimension
2. Map range to [0, 255]
3. Store quantization parameters

Compression: 4× smaller  
Speed: Faster dot products (int8 SIMD)  
Quality: Minimal degradation (<1% recall drop)

Methods:
- Post-training quantization (PTQ)
- Quantization-aware training (QAT)

Tools: GPTQ, GGUF, bitsandbytes

**Product Quantization (PQ):**
1. Split d-dimensional vector into m subvectors
2. Quantize each subvector independently (k-means with 256 centroids)
3. Store centroid IDs (1 byte each)

Compression: d × 4 bytes → m bytes (often 32× or 64×)  
Quality: Moderate degradation (2-5% recall drop)

**Binary Quantization:**
Convert to binary: x_i → sign(x_i)

Compression: 32× smaller  
Speed: Hamming distance = POPCNT instruction  
Quality: Significant degradation (use for filtering, then rerank)

**References:**
- [Approximate Nearest Neighbor - Wikipedia](https://en.wikipedia.org/wiki/Nearest_neighbor_search)
- [vector similarity](https://redis.io/blog/vector-similarity/)
- [HNSW params](https://www.linkedin.com/posts/sarthakrastogi_ai-genai-aiagents-activity-7377665823470968832--0WB)
- [DataBricks scaled vector search](https://www.databricks.com/blog/decoupled-design-billion-scale-vector-search)

---

## Q19 — Transformers

[transformer deep dive](cracking_llm_basics.md)

Recurrent Neural Networks:
$$h_t = f(W_{hh} h_{t-1} + W_{xh} x_t)$$

Problems:

1. Vanishing Gradients:
   - Gradients decay exponentially with distance
   - Practical context length: ~100-500 tokens
   - LSTM/GRU help but don't fully solve

2. Sequential Processing:
   - Cannot parallelize: h_t depends on h_{t-1}
   - Training time O(T) for sequence length T
   - GPU underutilized

3. Limited Context:
   - Information compressed into fixed-size hidden state
   - Struggles with long-range dependencies Can handle ~100-500 tokens
   - Attention mechanisms added as patch

LSTM Improvements:
- Gates (forget, input, output) help gradient flow
- Limited context: Can handle ~1000 tokens
- Still fundamentally sequential

**Transformer Solutions:**

Key Innovation: Replace recurrence with self-attention

- Parallel processing (all positions at once)
- Direct connections (attention)
- No gradient vanishing
- Longer context (1000s of tokens)
- Training time O(1) per layer (O(log T) total with depth)

---

## Q20 — Temperature in LLM Sampling

Temperature Formula

$$P(x_i) = \frac{e^{z_i/\tau}}{\sum_j e^{z_j/\tau}}$$

Where:
- z_i = logits from model
- τ = temperature

Effect of Temperature

τ < 1 (Low temperature):
- Sharper distribution
- More deterministic
- Less creative

τ = 1:
- Original model probabilities

τ > 1 (High temperature):
- Flatter distribution
- More random
- More creative

τ → 0:
- Greedy decoding (always pick highest probability)

### Practical Guidelines

| Temperature | Use Case |
|-------------|----------|
| 0.0 - 0.3 | Factual Q&A, code, math |
| 0.4 - 0.7 | Balanced, general chat |
| 0.8 - 1.2 | Creative writing, brainstorming |
| 1.5+ | Experimental, highly random |

### Other Sampling Methods

Top-k: Keep only k most probable tokens

Keep only k most probable tokens, redistribute probability

Algorithm:
1. Sort tokens by probability
2. Keep top k
3. Renormalize
4. Sample

Effect: Eliminates long tail of unlikely tokens

Typical k: 40-50


Top-p (Nucleus): Keep smallest set with cumulative probability ≥ p
- Adaptive to model confidence
- Typical p = 0.9-0.95

Keep smallest set of tokens with cumulative probability ≥ p

Algorithm:
1. Sort tokens by probability
2. Keep adding until sum ≥ p
3. Renormalize
4. Sample

Advantage over Top-k:
- Adaptive to confidence
- High confidence → fewer tokens
- Low confidence → more tokens

Example:
- High confidence: "The capital of France is ___" → nucleus ≈ 1-2 tokens
- Low confidence: "Once upon a time ___" → nucleus ≈ 50+ tokens

Repetition Penalty: Reduce probability of already-generated tokens
Frequency Penalty: Reduce based on overall frequency in response  
Presence Penalty: Reduce if token appeared at all

**References:**
- [Temperature Sampling - Hugging Face](https://huggingface.co/blog/how-to-generate)

---

## Q21 — Context Window

Definition: Maximum number of tokens model can process

Limitations:
- Memory: O(n²) for attention
- Computation: O(n² × d) per layer
- Positional encodings may not extrapolate

Evolution
- GPT-3: 2048 tokens
- GPT-3.5: 4096 tokens
- GPT-4: 8K → 32K → 128K tokens
- Claude 3: 200K tokens
- Gemini 1.5 Pro: 1M tokens

### Extension Methods

#### RoPE (Rotary Position Embedding)

Encode position by rotating Q/K (query/key) vectors

$$f_q(x_m, m) = (W_q x_m) e^{im\theta}$$

Where θ is frequency parameter

Benefits:
- Relative position preserved in inner products
- Better extrapolation than learned embeddings

Rotary Positional Embeddings (RoPE) quietly power most modern LLMs.
They’re one of the reasons long-context models actually work.

Why this matters
- Attention cares about relative distance, not absolute position
- RoPE makes long-context generalization far more stable than learned position embeddings

How RoPE works
- Queries and keys are split into pairs of dimensions
- Each pair is treated as a 2D vector
- These vectors are rotated by an angle based on token position
- Rotation preserves magnitude but changes direction
- Tokens close together have similar rotations
- Tokens far apart have larger phase differences
- During Q·K attention, these phase differences encode relative distance naturally

Why RoPE beats absolute positional embeddings
- No fixed lookup table for positions
- Encodes relative offsets directly inside attention
- Extrapolates better to longer sequences
- Adds a strong inductive bias aligned with how attention works

* [RoPe explain](https://www.linkedin.com/posts/hoang-van-hao_machinelearning-llm-llama3-activity-7408475024220835840-7M7Y)
* [How RoPE work](https://www.linkedin.com/posts/sarthakrastogi_ai-llms-genai-activity-7423159922256809984-ou9b)

**Extension Techniques:**

Position Interpolation (PI):
- Compress positions during fine-tuning
- Train on 2048, compress to use 4096
- Maintains relative distances

YaRN (Yet another RoPE extensioN method):
- Frequency-aware interpolation
- Different frequencies extended differently
- Better extrapolation than PI

#### ALiBi (Attention with Linear Biases)

Add linear bias to attention scores based on distance

$$\text{softmax}(QK^T + m \cdot [-1, -2, -3, ..., -n])$$

Benefits:
- No positional embeddings needed
- Superior extrapolation
- Can train on 2K, inference on 10K+

References:
- [RoPE - Paper](https://arxiv.org/abs/2104.09864)

---

## Q22 — Tokenization: BPE

### Byte Pair Encoding (BPE)

Algorithm:
1. Start with character vocabulary
2. Count all adjacent pairs
3. Merge most frequent pair
4. Add to vocabulary
5. Repeat until desired vocab size

Example:
```
Initial: ["l", "o", "w", "e", "r"]
Merge "e"+"r" → "er"
Merge "er"+" " → "er "
...
Final vocab: ["low", "er", " ", "wide", "st"]
```

Advantages:
- Balances vocab size and sequence length
- Handles rare words (breaks into subwords)
- No unknown tokens (can always fall back to characters)

Used by: GPT-2, GPT-3, RoBERTa

### WordPiece

Merges based on **likelihood increase**, not frequency

Used by: BERT, DistilBERT

### SentencePiece

Language-agnostic, treats whitespace as character

Algorithms: BPE or Unigram LM

Benefits:
- Works for languages without spaces (Chinese, Japanese)
- Reversible

Used by: T5, XLNet, multilingual models

**References:**
- [BPE - Neural Machine Translation](https://arxiv.org/abs/1508.07909)

---

## Q23 — GPT vs BERT

### GPT (Generative Pre-trained Transformer)

Architecture: Decoder-only with causal (masked) attention

Attention: Can only attend to previous tokens

$$\text{Mask}_{ij} = \begin{cases} 0 & \text{if } i \geq j \\ -\infty & \text{if } i < j \end{cases}$$

Training Objective:

$$\mathcal{L} = -\sum_{t=1}^{T} \log P(x_t | x_1, ..., x_{t-1})$$

Next token prediction (left-to-right language modeling)

Strengths:
- Text generation
- Autoregressive tasks
- Few-shot learning (in-context)
- Dialogue/chat

Weaknesses:
- Can't look ahead
- Less effective for understanding tasks

Used for: ChatGPT, GPT-4, text generation, dialogue

### BERT (Bidirectional Encoder Representations from Transformers)

Architecture: Encoder-only with bidirectional attention

Attention: Can attend to all tokens (past and future)

Training Objectives:

1. MLM (Masked Language Modeling):
   - Randomly mask 15% of tokens
   - Predict masked tokens using context

$$\mathcal{L}_{MLM} = -\sum_{i \in \text{masked}} \log P(x_i | x_{\backslash i})$$

2. NSP (Next Sentence Prediction):
   - Given [CLS] Sent_A [SEP] Sent_B [SEP]
   - Predict if Sent_B follows Sent_A
   - Dropped in RoBERTa (didn't help much)

Strengths:
- Classification (sentiment, NER, QA)
- Understanding bidirectional context
- Transfer learning for NLP tasks

Weaknesses:
- Cannot generate text naturally
- Mismatch: [MASK] in pretraining, not in fine-tuning

Used for:
- Classification (sentiment, NER)
- Question answering
- Sentence similarity
- Understanding bidirectional context

### Architectural Comparison

| Aspect | GPT (Decoder) | BERT (Encoder) |
|--------|---------------|----------------|
| **Attention** | Causal (unidirectional) | Bidirectional |
| **Training** | Next token prediction | Masked LM + NSP |
| **Primary Use** | Generation | Understanding |
| **Context** | Left-to-right | Full sequence |
| **Tasks** | Text generation, chat, completion | Classification, NER, QA, embeddings |
| **Architecture** | Decoder stack | Encoder stack |
| **Output** | One token at a time | All tokens simultaneously |

**Encoder-Decoder Models:** T5, BART combine both

**References:**
- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [GPT-2 Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

---

## Q24 — SFT, RLHF, DPO (Instruct Training)

### Training Pipeline

#### 1. Pretraining

Learn language patterns from massive text corpora

Data:
- Web crawl (Common Crawl)
- Books, Wikipedia
- Code repositories (GitHub)
- Total: Trillions of tokens

Loss: Next-token prediction

$$\mathcal{L} = -\sum_{t=1}^{T} \log P_\theta(x_t \mid x_{< t})$$

Result: Base model with strong language understanding but:
- Doesn't follow instructions
- Unpredictable behavior
- May continue user query rather than answer

Example:
- User: "Write a poem about AI"
- Base model: "Write a poem about robotics Write a poem about..." (completion, not instruction following)

#### 2. SFT (Supervised Fine-Tuning)

Objective: Teach model to follow instructions

Data Format: (instruction, output) pairs
```
{
  "instruction": "Translate to French: Hello, how are you?",
  "output": "Bonjour, comment allez-vous?"
}
```

Training: Standard supervised learning

$$\mathcal{L}_{SFT} = -\sum_i \log P_\theta(y_i | x_i)$$

Dataset Size: 10K - 100K high-quality examples

Example:
```json
{
  "instruction": "Translate to French: Hello",
  "output": "Bonjour"
}
```

Key Insight: Quality >> Quantity
- 1,000 excellent examples > 100,000 mediocre ones
- Diversity matters (cover many task types)

Sources:
- Human demonstrations
- Distillation from strong models (GPT-4, Claude)
- Self-instruct (model generates its own training data)

Result: Model that follows instructions but:
- May be verbose, overly apologetic
- Not aligned with human preferences
- Doesn't know when to refuse


#### 3. RLHF (Reinforcement Learning from Human Feedback)

Two-Stage Process:

**Stage 1: Train Reward Model**

- Collect preference pairs (response A > response B)
- Train classifier to predict human preference

Data Collection:
1. Sample prompts from dataset
2. Generate multiple responses from SFT model
3. Humans rank responses (A > B > C)

Model: Classifier predicting human preference

$$r_\theta(x, y) \in \mathbb{R}$$

Loss (Bradley-Terry Model):

$$\mathcal{L}_r = -E \left[\log \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))\right]$$

Where y_w = preferred (winner), y_l = less preferred (loser)

Intuition: Reward model scores how "good" a response is

**Stage 2: RL Fine-tuning (PPO)**

Objective:

$$\mathcal{J}(\theta) = E_{x \sim D, y \sim \pi_\theta}[r_\theta(x, y)] - \beta \cdot D_{KL}(\pi_\theta || \pi_{ref})$$

Components:
- Reward: r_θ(x, y) from reward model
- KL penalty: Prevents model from diverging too far from reference (SFT) model
  - Without KL: Model might "hack" reward model
  - β controls regularization strength (typical: 0.01-0.1)

PPO (Proximal Policy Optimization):
- Stable RL algorithm
- Clips updates to prevent large policy changes
- Multiple epochs per batch

Result: Model aligned with human preferences

Challenges:
- Computationally expensive (4 models in memory)
- Unstable training
- Reward hacking (model finds shortcuts)

#### 4. DPO (Direct Preference Optimization)

Key Insight: Bypass reward model, optimize policy directly from preferences

Mathematical Foundation:
The optimal policy for RLHF satisfies:
$$\pi^*(y|x) \propto \pi_{ref}(y|x) \exp\left(\frac{r(x,y)}{\beta}\right)$$

Rearranging:
$$r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$

DPO Loss:

$$\mathcal{L}_{DPO} = -E\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

Intuition:
- Increase probability of preferred response y_w
- Decrease probability of dispreferred response y_l
- Relative to reference policy

Advantages over RLHF:
- No reward model needed
- More stable training
- Only 2 models instead of 4 (policy + reference)
- Simpler implementation

Data: Same preference data as RLHF

Result: Similar quality to RLHF with simpler training

### Comparison

| Aspect | SFT | RLHF | DPO |
|--------|-----|------|-----|
| **Data** | Demonstrations | Preferences | Preferences |
| **Training** | Supervised | RL | Supervised |
| **Models** | 1 | 4 | 2 |
| **Stability** | High | Low | Medium |
| **Compute** | Low | Very High | Medium |

Typical Pipeline:
1. Pretrain on trillions of tokens
2. SFT on 10K-100K demonstrations
3. RLHF or DPO on 100K-1M preference pairs

Recent Trends:
- RLAIF: Replace human feedback with AI feedback
- Constitutional AI: Model critiques its own outputs
- Multi-objective RLHF: Optimize for helpfulness, harmlessness, honesty
- Iterative DPO: Multiple rounds with updated reference

References:
- [InstructGPT Paper](https://arxiv.org/abs/2203.02155)
- [DPO Paper](https://arxiv.org/abs/2305.18290)

---

## Q25 — LLM Inference Optimization

[deep dive to llm ops](cracking_llm_ops.md)

#### Pruning

Remove low-magnitude weights:
- Structured pruning (entire layers/channels)
- Unstructured pruning (individual weights)

Result: Smaller, faster model

#### 4. Batching

Process multiple requests simultaneously:
- Amortize fixed costs
- Better GPU utilization

Dynamic batching: Group requests on-the-fly

#### 5. KV-Cache

Cache key/value tensors for autoregressive models:
- Avoid recomputing for already-generated tokens
- Essential for efficient generation
- Memory vs computation

#### 6. Flash Attention

Optimized CUDA kernels:
- Memory-efficient exact attention
- 2-4× speedup
- No approximation

Sparse Attention: Attend to subset of tokens  
Sliding Window: Local + global attention  
Landmark Tokens: Special tokens summarize context


#### 7. Speculative Decoding

Use small model to propose, large model to verify:
- Generate multiple tokens in parallel
- 2-3× speedup for autoregressive generation

#### 8. Hardware Optimization

Specialized accelerators:
- TensorRT (NVIDIA)
- ONNX Runtime
- A100, H100 GPUs
- Custom ASICs (TPUs, Inferentia)

### Models optimization

Model Formats for Local Inference: GGUF, Ollama, vLLM)

Optimization Stack: **Model → Quantization → Compilation → Hardware**

**Example pipeline:**
1. Quantize to INT8
2. Compile with TensorRT
3. Deploy on A100 GPU
4. Use dynamic batching
5. Enable KV-cache

Result: 10-100× speedup vs naive implementation

**References:**
- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)
- [GPTQ - Quantization](https://arxiv.org/abs/2210.17323)
- [vLLM - Inference Engine](https://github.com/vllm-project/vllm)
- [scaled dot-product attention](https://medium.com/@maxwbuckley/is-attention-all-you-need-to-prepare-for-ml-interviews-830742f6d2ba)
- [KV cache](https://www.linkedin.com/posts/sarthakrastogi_ai-llms-genai-activity-7416260128800280576-BshQ)

---

## Q26 — ML Project Stages

### 1. Problem Definition
- Understand business objective
- Define success metrics
- Assess feasibility

### 2. Data Collection
- Identify data sources
- Collect and consolidate
- Document provenance

Key Techniques/Tools:
- SQL databases, NoSQL (MongoDB), data lakes (AWS S3, Azure Data Lake)
- Web scraping (BeautifulSoup, Scrapy)
- APIs (REST, GraphQL)
- Data labeling tools (Labelbox, Amazon Mechanical Turk)
- ETL pipelines (Apache Airflow, Luigi)

Common Pitfalls:
- **Selection bias**: Collecting non-representative samples
- **Insufficient data volume**: Inadequate samples for model generalization (how many samples do you need for model training)
- **Poor data quality**: Missing values, duplicates, inconsistent formats
- **Legal/ethical issues**: Violating privacy regulations (GDPR, HIPAA)

Best Practices:
- Document data sources, collection methods, and timestamps
- Implement data versioning (DVC, MLflow)
- Ensure balanced class representation for classification tasks

EDA is the process of analyzing datasets to summarize main characteristics, detect anomalies, discover patterns, and form hypotheses using statistical and visual methods.

Key Techniques/Tools:
- Statistical analysis: Mean, median, standard deviation, correlation matrices
- Visualization: Histograms, box plots, scatter plots, heatmaps
- Tools: Pandas, Matplotlib, Seaborn, Polars


### 3. EDA (Exploratory Data Analysis)

- Statistical summaries
- Visualizations
   - Histogram
   - Box and Whisker plot - for outliers
   - Q-Q plot (normality check)
   - Violin plot

- Identify patterns, anomalies
- Formulate hypotheses

### 4. Preprocessing
- Cleaning: Handling missing values (imputation, deletion), outlier treatment
- Encoding: One-hot encoding, label encoding, target encoding
- Scaling: StandardScaler, MinMaxScaler, RobustScaler
- Feature creation: Polynomial features, binning, log transformations
- Feature selection: Filter methods, wrapper methods (RFE), embedded methods (L1 regularization)

**Critical Pitfall - Data Leakage:**
Using test data information during preprocessing invalidates evaluation.

### 5. Feature Engineering
- Create new features
- Feature selection
- Dimensionality reduction

### 6. Model Selection
- Choose candidate algorithms
- Train baselines
- Compare performance


### 7. Training & Hyperparameter Tuning

Cross-Validation Methods:
- K-Fold, Stratified K-Fold (for classification)
- Time Series Split (for temporal data)
- Leave-One-Out (small datasets)

Hyperparameter Tuning:
- Grid Search: Exhaustive but computationally expensive
- Random Search: Often more efficient
- Bayesian Optimization (Optuna): Most efficient for expensive evaluations


### 8. Evaluation
- Test on holdout set
- Error analysis
- Compare to business metrics

Go deep
- Analyze misclassifications by category
- Report confidence intervals
- Assess fairness across demographic groups

### 9. Deployment
- Containerization (Docker)
- API development (FastAPI, Flask)
- CI/CD pipeline
- A/B testing

### 10. Monitoring
- Track performance metrics
- Detect data/concept drift
- Set up alerts
- Plan retraining

Best Practices:
- Define SLAs for model performance
- Implement automated drift detection
- Establish retraining triggers and schedules

### Repository Structure

```
project/
├── data/
│   ├── raw/           # Original data (immutable)
│   ├── processed/     # Cleaned data
├── notebooks/         # EDA, experiments
├── src/
│   ├── data/          # Data processing
│   ├── features/      # Feature engineering
│   ├── models/        # Model definitions
│   ├── evaluation/    # Metrics, validation
├── models/            # Saved models
├── configs/           # Config files (YAML, JSON)
├── tests/             # Unit tests
├── requirements.txt
├── Dockerfile
└── README.md
```

**References:**
- [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/)

---

## Q27 — RAG (Retrieval-Augmented Generation)


Problems with Vanilla LLMs:
- Knowledge cutoff: Can't access recent information
- Hallucinations: Generate plausible but incorrect information
- Domain specificity: Lack specialized knowledge

RAG Benefits:
- Up-to-date information (retrieve from current docs)
- Reduced hallucinations (grounded in retrieved evidence)
- Domain adaptation without fine-tuning
- Verifiable: Can inspect retrieved sources
- Cost-effective: Cheaper than retraining
- Source attribution: Can cite sources


Pipeline: Retrieval → Ranking → Generation

```
User Query → Embedding → Vector Search → Top-K Retrieval 
           ↓
    Reranking → Context + Query → LLM → Answer
```


### 1. Chunking

**Chunking:**  Documents too long for context window, Split documents into chunks (512-1024 tokens)


Chunking Strategies

Fixed-Size Chunking
- Split every N characters/tokens
- Simple but may break mid-sentence/paragraph
- Typical: 512-1024 tokens with 50-100 token overlap

Semantic Chunking
- Split on topic boundaries
- Use sentence embeddings to detect topic shifts
- Better coherence but more complex

Recursive Chunking
- Hierarchical splitting: chapters → sections → paragraphs
- Preserves document structure
- Good for structured documents

Sentence Window
- **Index:** Individual sentences
- **Retrieve:** Surrounding sentences (window)
- Best precision/context balance

### 2. Embedding

**Embedding:** sentence-transformers, OpenAI embeddings, BGE, using **Bi-encoder:** Encode query and documents separately

**Storage:** Vector DB (Pinecone, Weaviate, FAISS, Milvus, Qdrant)

### 3. Retrieval

Retrieve top-20 to top-100. Methods: Dense (Semantic), Sparse (BM25 for exact keyword matching) or hybrid: Combine both with weighted sum

$$\text{score}_{hybrid} = \alpha \cdot \text{score}_{dense} + (1-\alpha) \cdot \text{score}_{sparse}$$

**Pre-generation filtering:**
- Remove low-relevance documents (score threshold)
- Check for contradictions between sources
- Prioritize recent/authoritative sources


### 4. Reranking

**Purpose:** Improve precision of top results using **Cross-encoder:** Jointly encode [query; document]. Reduce top-100 → top-10.

- More accurate than bi-encoder
- Slower - Must encode each pair (only for top-K)

**Popular Rerankers:**
- Cohere rerank
- bge-reranker
- Cross-encoder models (SBERT)

#### 4. Generation

**Prompt:**
```
Context:
{retrieved_doc_1}
{retrieved_doc_2}
...

Question: {user_query}

Answer based on context. If not in context, say "I don't know."
```

### Reducing Hallucinations

1. Citation Requirements:
- Require model to cite sources
- Verify citations support claims

2. Confidence Scores:
- Ask for confidence (0-10)
- If low, abstain or flag for human review

3. Self-Consistency:
- Generate multiple responses (different temperature/sampling)
- Check agreement
If responses differ significantly, flag uncertainty

4. Entailment Checking:
- Use NLI model to verify claims

**Example:**
- Claim: "The company was founded in 2015"
- Context: "Since its founding in 2015..."
- Entailment: TRUE ✓

### Advanced: 

#### HyDE (Hypothetical Document Embeddings)

Process:
1. Generate hypothetical answer to query (without retrieval)
2. Use hypothetical answer as search query
3. Retrieve documents similar to hypothetical answer
4. Generate final answer from retrieved docs

Benefit: Bridges vocabulary gap (query vs document language)

#### Query Decomposition

Process:
1. Break complex query into simpler sub-queries
2. Retrieve for each sub-query
3. Synthesize final answer

Example:
- Query: "Compare revenue growth of Apple and Microsoft 2020-2023"
- Sub-queries:
  - "Apple revenue 2020-2023"
  - "Microsoft revenue 2020-2023"

## Q28 — Agents, Tool Calling, ReAct

### What Are Agents?

[complex agents intro](https://cold-scallion-5b8.notion.site/AI-agents-1979c76f79e8808291a4e463e8224bfc?pvs=74)

**Definition:** Autonomous systems that can:
1. **Perceive:** Understand environment/task
2. **Reason:** Plan and decide actions
3. **Act:** Execute using tools/APIs
4. **Learn:** Adapt from feedback

**Contrast with Standard LLMs:**
- LLM: Stateless, single-turn, passive
- Agent: Stateful, multi-turn, proactive

### Agentic RAG vs Regular RAG

**Dynamic process with reasoning:**

```text
Query → Agent decides:
        ├─ Need retrieval? → Vector search
        ├─ Can answer directly? → Answer
        ├─ Need web search? → Web API
        └─ Need calculation? → Calculator

Retrieved context → Evaluate quality
                  → Reformulate if needed
                  → Generate & verify → Return
```

#### Actor-Critic for RAG

**Actor:** Generates retrieval queries  
**Critic:** Evaluates retrieved document quality

Iterative refinement:
1. Actor retrieves documents
2. Critic scores relevance
3. If low score: Actor refines query, retrieves again
4. Repeat until satisfactory

Key difference from basic RAG:
- Decides WHEN to retrieve (dynamic pipeline)
- Uses MULTIPLE tools (search, calculate, API calls)
- ITERATIVE refinement (self-correcting)
- EXPLICIT reasoning traces (chain-of-thought)

| Aspect | Regular RAG | Agentic RAG |
|--------|-------------|-------------|
| **Pipeline** | Fixed (retrieve → generate) | Dynamic (decides when to retrieve) |
| **Retrieval** | Single-shot | Multi-step, iterative |
| **Tools** | None | Multiple tools (search, calculate, API calls) |
| **Reasoning** | Implicit | Explicit (chain-of-thought) |
| **Adaptation** | None | Self-corrects based on results |

Example:
- **Regular RAG:** "What's the weather?" → Retrieve → Answer
- **Agentic RAG:** "What's the weather?" → Use weather API → Parse result → Answer in user's preferred units

Tool Calling: LLM invokes external functions/APIs. Agent may call multiple tools in sequence or parallel (Web search, Database queries, Code execution, Send email, Create calendar event, API calls).

#### Process

**1. Define Tools (JSON Schema)**
```json
{
  "name": "get_weather",
  "description": "Get current weather for a location",
  "parameters": {
    "type": "object",
    "properties": {
      "location": {"type": "string"},
      "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
    },
    "required": ["location"]
  }
}
```

**2. LLM Generates Function Call**
```json
{
  "tool": "get_weather",
  "arguments": {
    "location": "Paris",
    "unit": "celsius"
  }
}
```

**3. Application Executes Function**
```python
result = get_weather(location="Paris", unit="celsius")
# Returns: {"temperature": 15, "condition": "cloudy"}
```

**4. Result Added to Context**
LLM continues with function result

**5. LLM Generates Final Response**
"The current weather in Paris is 15°C and cloudy."

### ReAct (Reasoning + Acting) Pattern

**Structure:** Interleave reasoning traces with actions

**Format:**
```text
Thought: [Reasoning about what to do next]
Action: [Tool to use + arguments]
Observation: [Result from tool]
Thought: [Reasoning about observation]
Action: [Next tool call]
Observation: [Result]
...
Thought: [Final reasoning]
Answer: [Final response]
```

**Example:**
```text
User: "What's the capital of the country where the 2024 Olympics were held?"

Thought: I need to find where the 2024 Olympics were held
Action: Search["2024 Olympics location"]
Observation: Paris, France

Thought: Now I know it was in France, I need to confirm the capital
Action: Search["capital of France"]
Observation: Paris

Thought: I have the answer
Answer: The capital is Paris, which is also where the 2024 Olympics were held.
```

**Advantages:**
- **Transparent:** Can trace reasoning
- **Debuggable:** See where agent fails
- **Self-correcting:** Can revise plan based on observations
- **Works out-of-box:** No fine-tuning needed for strong models

**Implementation:** Prompt engineering—provide examples in few-shot format

### Planning Approaches

#### Plan-and-Execute

**Process:**
1. Create complete plan upfront
2. Execute steps sequentially
3. Handle errors if any step fails

**Example:**
```text
Plan:
1. Search for "2023 Nobel Prize Physics"
2. Extract winner names
3. Search for "{winner} current position"
4. Summarize findings
```

Pros: Clear structure  
Cons: Rigid, can't adapt if intermediate results unexpected

#### Tree of Thoughts (ToT)

Process:
1. Generate multiple reasoning paths
2. Evaluate each path
3. Prune low-quality paths
4. Explore most promising
5. Backtrack if needed

Example:
```text
        [Initial Query]
       /      |      \
   Path A   Path B  Path C
    /|\       |      /|\
   ...       ...    ...
```

Evaluation: LLM rates each thought (1-10) or external evaluator
Use Case: Complex reasoning, math problems, puzzles

#### Reflection

Process:
1. Generate initial solution
2. Critique solution
3. Refine based on critique
4. Iterate until satisfactory

Prompt:
```text
Task: {task}
Solution: {current_solution}

Critique this solution. What's wrong? What's missing?
```

Use Case: Writing, code generation, complex analysis

### Memory Types

Short-term (Working Memory): Current conversation context, Recent tool outputs, Intermediate results

**Implementation:** In-context (part of prompt)

Long-term Memory
- **Episodic (Event Memory):** Past interactions, Historical tool use, User feedback
- **Semantic (Knowledge Memory):** Facts learned about user, Preferences, Domain knowledge
- **Procedural (Skill Memory):** Learned workflows, Successful strategies, Instructions

**Implementation:**: Vector database (semantic search), Structured storage (SQLite, Redis)

### Agentic RAG Design

**Architecture:** LLM as orchestrator with access to multiple tools

**Tools:**
- **Vector search:** Semantic document retrieval
- **Keyword search:** BM25 for exact matches
- **Web search:** External knowledge
- **SQL database:** Structured data
- **Calculator:** Numerical computations
- **Code execution:** Complex transformations

**Multi-tool orchestration:**
```python
tools = [
    VectorSearch(),      # Semantic retrieval
    BM25Search(),        # Keyword search
    WebSearch(),         # External knowledge
    SQLQuery(),          # Structured data
    Calculator()         # Numerical computations
]

agent = Agent(llm=llm, tools=tools)
```

#### Example Flow

**Query:** "Compare our Q3 sales to industry average and explain variance"

**Agent Execution:**
```text
Thought: Need our Q3 sales data
Action: sql_query("SELECT sales FROM company_data WHERE quarter='Q3'")
Observation: $5.2M

Thought: Need industry average
Action: web_search("Q3 2024 industry sales average")
Observation: Industry average is $6.1M

Thought: Calculate variance
Action: calculate("(5.2 - 6.1) / 6.1 * 100")
Observation: -14.75%

Thought: Search for potential reasons
Action: vector_search("factors affecting Q3 sales")
Observation: [Market slowdown in region X, delayed product launch]

Thought: I have enough information
Answer: Our Q3 sales of $5.2M are 14.75% below the industry average 
of $6.1M. Key factors: [synthesized explanation]
```

#### Challenges

- **Reliability:** Tool calling errors, Hallucinated tool arguments, Infinite loops.
  **Solutions:** Retry logic with exponential backoff, Argument validation, Maximum iteration limits.
- **Cost:** Multiple LLM calls → higher cost
- **Latency:** Sequential tool calls → slower response

### Agent Frameworks

#### LangChain
- **Features:** Modular components (agents, tools, memory, chains), 100+ integrations, Expression language for workflows.
- **Strengths:** Comprehensive, mature ecosystem.  
- **Weaknesses:** Can be complex, verbose.

#### LlamaIndex
- **Features:** Data-focused (connectors to 160+ data sources), Strong RAG primitives, Query engines, agents.
- **Strengths:** Best-in-class RAG, data integration.  
- **Weaknesses:** Less general than LangChain.

#### CrewAI
- **Features:** Multi-agent orchestration, Role-based agents with specific expertise, Collaborative workflows.

**Example:**
```python
researcher = Agent(role="Researcher", goal="Find information")
writer = Agent(role="Writer", goal="Write article")
crew = Crew(agents=[researcher, writer], process="sequential")
```

**Strengths:** Multi-agent coordination.  
**Use Case:** Complex tasks requiring multiple specialized agents.

#### Comparison

| Framework | Best For | Complexity | Focus |
|-----------|----------|------------|-------|
| **LangChain** | General-purpose agents | High | Orchestration, tools |
| **LlamaIndex** | RAG systems | Medium | Data, retrieval |
| **CrewAI** | Multi-agent workflows | Medium | Agent collaboration |

**References:**
- [RAG - Paper](https://arxiv.org/abs/2005.11401)
- [ReAct Paper](https://arxiv.org/abs/2210.03629)
- [LangChain Documentation](https://python.langchain.com/)

---

## Bonus Tasks

### Task 1: Fair Coin Predictions

Dataset:
- 1000 samples
- 90 positive (9%)
- 910 negative (91%)

Prediction: Fair coin flip (50/50)

Expected outcomes:
- E[TP] = 90 × 0.5 = 45
- E[FN] = 90 × 0.5 = 45
- E[TN] = 910 × 0.5 = 455
- E[FP] = 910 × 0.5 = 455

Precision:

$$\text{Precision} = \frac{TP}{TP + FP} = \frac{45}{45 + 455} = \frac{45}{500} = 0.09 = 9\%$$

Recall:

$$\text{Recall} = \frac{TP}{TP + FN} = \frac{45}{45 + 45} = \frac{45}{90} = 0.5 = 50\%$$

Answer:
- Precision = 9% (same as base rate!)
- Recall = 50% (random guessing catches half)

---


### Task 2: Order Acceptance Probability

Setup:
- 10 drivers
- Each driver: 50% conversion (accepts order)
- Order "burns" if NOBODY accepts

Question: P(order burns)?

Solution:

P(driver declines) = 0.5

P(all 10 decline) = (0.5)^10

$$P(\text{burn}) = 0.5^{10} = \frac{1}{1024} \approx 0.000977 = 0.0977\%$$

**Answer: ~0.1%** (very low probability)

**Alternatively:**

$$P(\text{at least one accepts}) = 1 - 0.5^{10} = 1 - \frac{1}{1024} = \frac{1023}{1024} \approx 99.9\%$$

---
