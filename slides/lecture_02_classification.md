# Logistic regression

* [Jupyter Notebook](../jupyter_notebooks/vol_00_pre_requirements_02_machine_learning_classification.ipynb)
* [Naive bayes classifier](../jupyter_notebooks/vol_04_deep_dive_00_probability_hw_2_naive_bayes_solved.ipynb)

## LogLoss

Log Loss это синоним Binary Cross-Entropy (BCE).

Штрафует ли logloss корректно предсказанные метки классов? И если да, то зачем?

$$L(y, F(x)) = -y \log(p) - (1-y)\log(1-p)$$

где $p = \frac{1}{1 + e^{-F(x)}}$

- Антиградиент: $r_{im} = y_i - p_i$

Для предсказания классов нужен порог, ошибки классификации (precision, recall) - не гладкие.

Модель выдает не метку, а вероятность - при пороге 0.5 правильный прогноз в 0.51 может наказываться за слабую уверенность.

# Логистическая регрессия: простой кейс

**Данные:** класс 0 ($x < 4$ кг), класс 1 ($x > 4$ кг)

**Модель:**

$$P(y = 1 | x) = \sigma(wx + b) = \frac{1}{1 + e^{-(wx+b)}}$$

**Параметры:** $w$ и $b$

**Граница разделения:** сигмоид = 0.5 при $wx + b = 0$

$$x = -\frac{b}{w} = 4 \quad \Rightarrow \quad b = -4w$$

**Направление:** для $x < 4 \Rightarrow y = 0$ нужно $w > 0$

**Ответ:**

$$b = -4w, \quad w > 0$$

Например: $w = 1, b = -4$ или $w = 10, b = -40$


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

Compute Precision: Measures how many of the **predicted positives** were actually correct.

$$\text{Precision} = \frac{TP}{TP + FP}$$

Compute Recall: Measures how many of the **actual positives** were correctly identified.

$$\text{Recall} = \frac{TP}{TP + FN}$$

✅ High recall → The model catches most of the positives (few false negatives).

**When to Prioritize Recall?**

* Medical Diagnosis (Cancer, COVID-19, etc.) → Missing a sick patient is dangerous.
* Fraud Detection → Better to flag potential fraud than let it go unnoticed.
* Security Systems (Intrusion Detection) → Better to have false alarms than miss real threats.

**Trade-off:** High recall can increase **false positives** (low precision).

Recall, also known as **Sensitivity** or **True Positive Rate (TPR)**, is a key metric in classification models, especially when missing positive cases is costly (e.g., medical diagnosis, fraud detection).

Sensitivity and specificity measure opposite aspects of model performance:

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Sensitivity (Recall, TPR)** | $\frac{TP}{TP + FN}$ | **Ability to detect positives** |
| **Specificity (TNR)** | $\frac{TN}{TN + FP}$ | **Ability to detect negatives** |

* **High Sensitivity (Recall)** → Good at finding **positives** (e.g., cancer screening).
* **High Specificity** → Good at ruling out **negatives** (e.g., spam detection).

💡 **Trade-off**: Increasing recall often lowers specificity, and vice versa.

Compute Accuracy

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

- Measures the **overall correctness** of the model across all predictions.

⚠️ **Limitation:** Accuracy can be misleading on **imbalanced datasets**. A model predicting all negatives on a 95% negative dataset achieves 95% accuracy but fails to detect any positives.

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

Доля ложноположительных ответов (FPR) остается низкой, даже если многие отрицательные объекты классифицированы неверно (так как знаменатель — общее количество отрицательных объектов — огромен).

**1. Нечувствительность к дисбалансу классов**
ROC-AUC основана на true positive rate (TPR) и false positive rate (FPR). Когда негативный класс доминирует, даже большое количество ложноположительных прогнозов приводит к малому FPR, из-за чего модель выглядит лучше, чем есть на самом деле.

**2. Хороший ранжир ≠ хорошие предсказания**
ROC-AUC измеряет, насколько хорошо модель ранжирует положительные примеры выше отрицательных, но не то, насколько точно она предсказывает положительный класс при используемом пороге. Модель может иметь высокий ROC-AUC, но при этом показывать очень плохие precision или recall для миноритарного класса.

Вопрос: Если возвести предсказания алгоритма в квадрат, что станет с метрикой ROC AUC?

Ответ: Для вероятностей в диапазоне [0,1] возведение в квадрат просто сжимает их к 0, но монотонность сохраняет, поэтому порядок не меняется -> ROC AUC останется неизменным, так как метрика оценивает правильность ранжирования алгоритма - которое не изменится при возведении в квадрат. 

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

# Кейс - lazy model

"Ленивая" модель** - предсказывает только мажоритарный класс (всегда 1 или всегда 0).

Проблема: как обнаружить "ленивую" модель? Диагностика классификации: тест инвертирования классов

Основные метрики

$$\text{Precision} = \frac{TP}{TP + FP} \quad \text{Recall} = \frac{TP}{TP + FN}$$

$$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

Пример несбалансированного датасета

- Класс 1: 30,000 примеров (83.3%)
- Класс 0: 6,000 примеров (16.7%)
- **Итого:** 36,000 примеров

---

###  Сценарий 1: "Ленивая" модель

Поведение: модель всегда предсказывает класс 1

| | Pred 0 | Pred 1 |
|---------|--------|--------|
| **True 0** | 0 | 6,000 |
| **True 1** | 0 | 30,000 |

**Метрики:**
- TP = 30,000, FP = 6,000, FN = 0, TN = 0
- Recall = $\frac{30,000}{30,000 + 0} = 1.0$ ✓
- Precision = $\frac{30,000}{30,000 + 6,000} = 0.833$ ✓

**⚠️ Проблема:** Метрики выглядят хорошо, но модель бесполезна!

### Тест: инвертирование классов (0 ↔ 1)

После инверсии модель всё ещё предсказывает "старый класс 1" = **новый класс 0**

| | Pred 0 (новый) | Pred 1 (новый) |
|---------|--------|--------|
| **True 0** | 30,000 | 0 |
| **True 1** | 6,000 | 0 |

**Новые метрики:**
- TP = 0, FP = 0, FN = 6,000
- Recall = $\frac{0}{0 + 6,000} = 0$ ❌
- Precision = undefined (0/0) ❌

**Вывод:** Метрики упали в ноль → модель ленивая!

---

## Сценарий 2: Хорошо обученная модель

Исходная confusion matrix

| | Pred 0 | Pred 1 |
|---------|--------|--------|
| **True 0** | 5,500 | 500 |
| **True 1** | 600 | 29,400 |

**Метрики:**
- Recall = $\frac{29,400}{29,400 + 600} = 0.98$
- Precision = $\frac{29,400}{29,400 + 500} = 0.983$

После инвертирования классов

| | Pred 0 | Pred 1 |
|---------|--------|--------|
| **True 0** | 29,400 | 600 |
| **True 1** | 500 | 5,500 |

**Новые метрики**:
- Recall = $\frac{5,500}{5,500 + 500} = 0.917$
- Precision = $\frac{5,500}{5,500 + 600} = 0.901$

**Вывод:** Метрики остаются высокими → модель действительно обучена!


| Признак | Ленивая модель | Обученная модель |
|---------|----------------|------------------|
| **Recall на мажоритарном классе** | ≈ 1.0 | < 1.0 (есть ошибки) |
| **Precision** | ≈ доля мажоритарного класса | > доля класса |
| **После инвертирования** | Метрики → 0 | Метрики остаются высокими |
| **Confusion matrix** | Одна строка/столбец = 0 | Все ячейки заполнены |

---

## Практические рекомендации

### 1. Всегда проверяй Confusion Matrix

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
```

**Красные флаги:**
- Строка или столбец полностью нулевые
- Диагональ имеет один нулевой элемент

### 2. Используй дополнительные метрики

```python
from sklearn.metrics import balanced_accuracy_score, f1_score

# Учитывает дисбаланс классов
balanced_acc = balanced_accuracy_score(y_true, y_pred)

# Гармоническое среднее Precision и Recall
f1 = f1_score(y_true, y_pred)
```

### 3. Тест инвертирования

```python
# Инвертируй классы
y_true_inv = 1 - y_true
y_pred_inv = 1 - y_pred

# Пересчитай метрики
recall_inv = recall_score(y_true_inv, y_pred_inv)

# Если упали в ноль → ленивая модель
if recall_inv < 0.1:
    print("⚠️ Модель предсказывает только один класс!")
```

### 4. Борьба с дисбалансом

Взвешивание классов:
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(class_weight='balanced')
```

Ресемплинг:
```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Oversampling миноритарного класса
smote = SMOTE(sampling_strategy=0.5)
X_res, y_res = smote.fit_resample(X, y)

# Undersampling мажоритарного класса
rus = RandomUnderSampler(sampling_strategy=0.8)
X_res, y_res = rus.fit_resample(X, y)
```

---

## Быстрая диагностическая таблица

| Метрики | Интерпретация | Действие |
|---------|---------------|----------|
| Recall=1.0, Precision=доля_класса | 🚨 Ленивая модель | Проверить CM, изменить веса |
| Recall≈1.0, Precision>>доля_класса | ✅ Хорошая модель на мажоритарном | Проверить минор. класс |
| Recall<0.9, Precision>0.9 | ⚠️ Консервативная модель | Порог вероятности? |
| Обе метрики ~0.5 | 🤷 Рандомная модель | Переобучить с нуля |

---

## Математическое объяснение

Почему Precision = доля мажоритарного класса?

Если модель всегда предсказывает класс 1:

$$\text{Precision} = \frac{TP}{TP + FP} = \frac{n_1}{n_1 + n_0} = \frac{n_1}{n_{total}}$$

где $n_1$ - количество примеров класса 1

**Пример:**

$$\frac{30,000}{36,000} = 0.833 = 83.3\%$$

При инвертировании классов

$$\begin{aligned}
TP_{new} &= TN_{old} \\
FP_{new} &= FN_{old} \\
FN_{new} &= FP_{old} \\
TN_{new} &= TP_{old}
\end{aligned}$$

Для ленивой модели: $TN_{old} = 0$ → $TP_{new} = 0$ → метрики падают в 0

---

## Заключение

**Ключевая идея:** На несбалансированных данных высокие метрики могут быть обманчивы.

**Проверочный тест:** Инвертируй классы и пересчитай метрики
- Упали в ноль → ленивая модель
- Остались высокими → модель обучена

**Всегда смотри:**
1. Confusion Matrix
2. Метрики для ОБОИХ классов
3. Balanced Accuracy / F1-score
4. Метрики на валидации после class_weight или resampling

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

ROC-AUC can be deceptive on imbalanced datasets because it does not reflect how the model performs on the minority (often the most important) class.

Key reasons:

* Insensitive to class imbalance. ROC-AUC is based on true positive rate (TPR) and false positive rate (FPR). When the negative class dominates, even a large number of false positives can result in a small FPR, making the model look better than it actually is.

* Good ranking ≠ good predictions ROC-AUC measures how well the model ranks positives above negatives, not how well it predicts the positive class at a usable threshold. A model can have a high ROC-AUC while producing very poor precision or recall for the minority class.

* Ignores real operating point. In practice, you care about performance at a specific decision threshold (e.g., alert rate, cost constraint). ROC-AUC averages performance over all thresholds, many of which are irrelevant in highly imbalanced settings.

* Can hide poor minority-class performance. A classifier that almost always predicts the majority class can still achieve a deceptively high ROC-AUC if it slightly separates the classes.

Better alternatives for imbalanced problems

* Precision–Recall AUC (PR-AUC)
* Recall, Precision, F1-score at a chosen threshold
* Cost-sensitive metrics aligned with business impact
