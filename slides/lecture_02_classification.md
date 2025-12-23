# Logistic regression

* [Jupyter Notebook](../jupyter_notebooks/vol_00_pre_requirements_02_machine_learning_classification.ipynb)
* [Naive bayes classifier](../jupyter_notebooks/vol_04_deep_dive_00_probability_hw_2_naive_bayes_solved.ipynb)


## Logistic regression: simple case

- Выборка: 20 объектов
    - Класс 0 (dogs): вес < 4 кг
    - Класс 1 (cats): вес > 4 кг
- Признак: **вес** (x)
- Модель: **логистическая регрессия** с 1 признаком

Хотим построить логистическую регрессию, которая будет идеально разделять эти классы. Сколько будет обучающихся параметров, и какими они должны быть, чтобы разделить эти два класса?

Логистическая регрессия:

$$
P(y=1|x) = \sigma(w x + b) = \frac{1}{1 + e^{-(w x + b)}}
$$

где:

- $w$ — вес признака
- $b$ — смещение (bias)
- $\sigma$ — сигмоид

---

Шаг 1: Сколько параметров? 2 обучающихся параметра

- Один признак → 1 коэффициент $w$
- смещение $b$

---

Шаг 2: Какие значения им нужны?

* Класс 0: $x < 4 \Rightarrow \sigma(w x + b) \approx 0$
* Класс 1: $x > 4 \Rightarrow \sigma(w x + b) \approx 1$

Сигмоид = 0.5 при 

$$w x + b = 0 \Rightarrow x = -b/w$$

Чтобы идеально разделить классы:

$$ b/w = 4 \quad \Rightarrow \quad b = -4 w $$

---

Шаг 3: Условие направления

- Класс 0 (<4 кг) → 0
- Класс 1 (>4 кг) → 1

Тогда при $x < 4$ аргумент сигмоиды < 0, а при $x >4$  аргумент сигмоиды > 0 → $w > 0$

---

✅ Вывод

- 2 параметра $w$ и $b$
- Соотношение для идеального разделения:

$$b = -4 w, \quad w > 0 $$
    
- Любое положительное $w$ с соотношением $b=-4w$ разделит классы идеально.


# Classification metrics

**Binary classification problem**, confusion matrix:

|  | **Predicted Positive** | **Predicted Negative** |
| --- | --- | --- |
| **Actual Positive (1s)** | **True Positives (TP)** | **False Negatives (FN)** |
| **Actual Negative (0s)** | **False Positives (FP)** | **True Negatives (TN)** |
- **True Positives (TP)** → Model correctly predicted **positive**.
- **True Negatives (TN)** → Model correctly predicted **negative**.
- **False Positives (FP)** → Model incorrectly predicted **positive** (Type I error).
- **False Negatives (FN)** → Model incorrectly predicted **negative** (Type II error).

The **F1 score** is the harmonic mean of **precision** and **recall**:

$$F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

Compute Precision

$$\text{Precision} = \frac{TP}{TP + FP}$$

- Measures how many of the **predicted positives** were actually correct.


Compute Recall

$$\text{Recall} = \frac{TP}{TP + FN}$$

- Measures how many of the **actual positives** were correctly identified.

✅ High recall → The model catches most of the positives (few false negatives).

**When to Prioritize Recall?**

* **Medical Diagnosis (Cancer, COVID-19, etc.)** → Missing a sick patient is dangerous.
* **Fraud Detection** → Better to flag potential fraud than let it go unnoticed.
* **Security Systems (Intrusion Detection)** → Better to have false alarms than miss real threats.

🔹 **Trade-off:** High recall can increase **false positives** (low precision).

Recall, also known as **Sensitivity** or **True Positive Rate (TPR)**, is a key metric in classification models, especially when missing positive cases is costly (e.g., medical diagnosis, fraud detection).

Sensitivity and specificity measure opposite aspects of model performance:

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Sensitivity (Recall, TPR)** | $\frac{TP}{TP + FN}$ | **Ability to detect positives** |
| **Specificity (TNR)** | $\frac{TN}{TN + FP}$ | **Ability to detect negatives** |

* **High Sensitivity (Recall)** → Good at finding **positives** (e.g., cancer screening).
* **High Specificity** → Good at ruling out **negatives** (e.g., spam detection).

💡 **Trade-off**: Increasing recall often lowers specificity, and vice versa.

## Step 3: Compute F1 Score

$$F1 = 2 \times \frac{(TP/(TP + FP)) \times (TP/(TP + FN))}{(TP/(TP + FP)) + (TP/(TP + FN))}$$

---

**Where:**
- $TP$ = True Positives
- $FP$ = False Positives  
- $FN$ = False Negatives

**Use F1 when classes are imbalanced** (e.g., fraud detection, medical diagnosis).

- If **false positives & false negatives have different costs**, choose **precision or recall** instead:
    - **High Precision?** → Model avoids **false positives**.
    - **High Recall?** → Model avoids **false negatives**.

# ROC curve vs P-R curve

Both **Precision-Recall (P-R) Curves** and **ROC Curves** help evaluate classification models, but they are suited for **different scenarios**.

**ROC Curve** plots **True Positive Rate (Recall) vs. False Positive Rate (FPR)**:

- **Best for balanced datasets** where **positives & negatives are roughly equal**.
- **False positives matter less**, as FPR normalizes by total negatives.

**Use ROC Curve when**:

- Both **false positives (FP) and false negatives (FN) are equally important**.
- The dataset has **balanced classes** (e.g., disease detection where 50% have it).

ROC-AUC может выглядеть высокой на несбалансированных данных, потому что она фокусируется на ранжировании и false positive rate, которые определяются мажоритарным классом, часто скрывая плохую производительность на миноритарном классе.

**1. Нечувствительность к дисбалансу классов**
ROC-AUC основана на true positive rate (TPR) и false positive rate (FPR). Когда негативный класс доминирует, даже большое количество ложноположительных прогнозов приводит к малому FPR, из-за чего модель выглядит лучше, чем есть на самом деле.

**2. Хороший ранжир ≠ хорошие предсказания**
ROC-AUC измеряет, насколько хорошо модель ранжирует положительные примеры выше отрицательных, но не то, насколько точно она предсказывает положительный класс при используемом пороге. Модель может иметь высокий ROC-AUC, но при этом показывать очень плохие precision или recall для миноритарного класса.

**3. Игнорирует реальную рабочую точку**
На практике важна производительность при конкретном пороге принятия решения (например, частота срабатывания алертов, ограничения по затратам). ROC-AUC усредняет производительность по всем порогам, многие из которых неактуальны в условиях сильного дисбаланса.

**4. Может скрывать плохую работу на миноритарном классе**
Классификатор, который почти всегда предсказывает мажоритарный класс, всё равно может достичь обманчиво высокого ROC-AUC, если он хоть немного разделяет классы.

The **P-R Curve** plots **Precision vs. Recall**, focusing only on **positive class performance**:

- **Best for imbalanced datasets**, where **positives are rare** (e.g., fraud detection).
- **More informative when false positives are costly**.

**Use P-R Curve when**:

- **Positive class is rare** (e.g., cancer detection, fraud, spam filtering).
- You care more about **precision and recall trade-offs**.

| **Scenario** | **ROC Curve** | **P-R Curve** |
| --- | --- | --- |
| Balanced dataset (50-50) | ✅ Yes | ❌ No |
| Imbalanced dataset (e.g., 1% positives) | ❌ No | ✅ Yes |
| Medical diagnosis (minimizing false negatives) | ✅ Yes | ✅ Yes |
| Fraud detection (rare class) | ❌ No | ✅ Yes |
| When FP rate is misleading | ❌ No | ✅ Yes |

# Threshold tuning

**Default Threshold Limitations**

- The default threshold (0.5) often favors the majority class in imbalanced settings, leading to poor recall/precision for the minority class[1](https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/)[4](https://machinelearningmastery.com/tour-of-evaluation-metrics-for-imbalanced-classification/).
- Example: A model might achieve 99% accuracy by always predicting the majority class but fail to detect critical minority cases (e.g., fraud or disease)[4](https://machinelearningmastery.com/tour-of-evaluation-metrics-for-imbalanced-classification/).

## **ROC Curve Analysis**

- Identify the threshold that maximizes the **Youden’s J statistic** (J = TPR + TNR - 1) or balances True Positive Rate (TPR) and False Positive Rate (FPR)[1](https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/).
- Example: Use the **`roc_curve`** function in **`scikit-learn`** to extract thresholds and select the point closest to the top-left corner of the ROC curve.

## **Precision-Recall Curve Analysis**

- Focus on thresholds that balance precision and recall, especially useful when the minority class is critical[1](https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/)[4](https://machinelearningmastery.com/tour-of-evaluation-metrics-for-imbalanced-classification/).
- Example: Optimize for the **F1-score** (harmonic mean of precision and recall) or target a specific recall value.

Grid Search for Threshold Tuning

```python
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = LogisticRegression()
clf.fit(X_train, y_train)

y_probs = clf.predict_proba(X_test)[:, 1]

# Define costs
cost_fp = 1
cost_fn = 5

thresholds = np.linspace(0, 1, 100)
min_cost = float('inf')
best_threshold = 0.5

for threshold in thresholds:
    y_pred = (y_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    cost = cost_fp * fp + cost_fn * fn
    if cost < min_cost:
        min_cost = cost
        best_threshold = threshold

print(f"Best threshold: {best_threshold:.2f} with minimum cost: {min_cost}")
```

# Negative sampling

```python
def prepare_evaluation_df(input_df, negatives_per_one_positive=2):
    input_df = pl.from_pandas(input_df) if isinstance(input_df, pd.DataFrame) else input_df
    key_fields = ['dt', 'state_name', 'StoreID', 'IsOnline']
    # Select distinct combinations of 'dt', 'StoreID', 'CustomerID', and 'IsOnline'
    prepared_input_df = (
        input_df
        .select(key_fields + ['CustomerID', 'ProductID']).unique()
        .with_columns([pl.lit(1).alias('target').cast(pl.Int64)])
    )
    print('Transformation started...')
    cadidates_full_df = (
        input_df.select(key_fields + ['CustomerID']).unique()
        .join(
            input_df.select(key_fields + ['ProductID']).unique(),
            on=key_fields,
            how='inner'
        )
        .join(
            prepared_input_df,
            on=key_fields+['ProductID', 'CustomerID'],
            how='left'
        )
    )
    print(f"Negative candidates: {cadidates_full_df.filter(pl.col('target').is_null()).height}, Positive samples: {input_df.height}")
    negative_candidates_df = (
        cadidates_full_df.filter(pl.col('target').is_null())
        .sample(n=int(input_df.height * negatives_per_one_positive), seed=42)
        .with_columns([pl.lit(0).alias('target').cast(pl.Int64)])
    )
    user_item_df = (
        pl.concat([
            prepared_input_df.select(key_fields+['CustomerID', 'ProductID', 'target']),
            negative_candidates_df.select(key_fields+['CustomerID', 'ProductID', 'target'])
        ])
        .sort(by=['CustomerID', 'dt'])
    )
    user_item_df = user_item_df.with_columns([pl.col('dt').cast(pl.Date)])
    print(f"Num negatives {user_item_df.to_pandas()['target'].value_counts(normalize=True).to_dict().get(0)}")
    return user_item_df
```