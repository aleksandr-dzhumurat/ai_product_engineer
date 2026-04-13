# Мультиколлинеарность в линейной регрессии

Мультиколлинеарность — это когда два или более признака сильно линейно зависимы между собой.

- Формально: для признаков $X_1, X_2, ..., X_n$

    если $X_i \approx a \cdot X_j + b$, это уже проблема.

Пример

- $X_1$ = рост в см
- $X_2$ = рост в метрах
- Очевидно, они полностью линейно зависимы.

Почему это проблема

1. Коэффициенты становятся нестабильными
    - Малые изменения в данных → большие колебания $\beta$
    - Трудно интерпретировать признаки
2. Стандартные ошибки коэффициентов растут
    - p-values становятся неинформативными
    - Трудно оценить значимость признаков
3. Понижается точность прогнозов вне обучающей выборки
    - Модель переобучается на зависимости между признаками

Влияние на математику: линейная регрессия решает:

$$\hat{\beta} = (X^TX)^{-1} X^Ty$$

- Если $X^TX$ почти сингулярна из-за коллинеарности → $(X^TX)^{-1}$ плохо вычисляется
- Результат → огромные или нестабильные коэффициенты

---

Проверка на мультиколлинеарность

VIF (Variance Inflation Factor)

$$\text{VIF}_i = \frac{1}{1 - R_i^2}$$

- $R_i^2$ — коэффициент детерминации регрессии $X_i$ на остальные признаки
- VIF > 5–10 → проблема

Корреляционная матрица

- Если $|\text{corr}(X_i, X_j)| > 0.8$ → признак потенциально коллинеарный

---

Как бороться

Удаление / объединение признаков

- Удаляем один из сильно коррелирующих признаков
- Преобразуем их (например, PCA или среднее)

---

Регуляризация

- Ridge (L2)
- "Сжимает" коэффициенты, стабилизирует их
- Уменьшает влияние мультиколлинеарности

- Lasso (L1)
- Может полностью обнулять один из зависимых признаков

---

Преобразования

PCA / SVD
- Преобразуем коррелированные признаки в независимые компоненты
- Используется, если важна предсказательная способность, а не интерпретация

# Metrics

* [mape-vs-smape](https://towardsdatascience.com/choosing-the-correct-error-metric-mape-vs-smape-5328dec53fac)
* [ml-meets-economic](https://nicolas.kruchten.com/content/2016/01/ml-meets-economics/)
* [Yandex ML handbook](https://education.yandex.ru/handbook/ml/article/metriki-klassifikacii-i-regressii)

Философия sklearn: Sklearn включает методы, которые:

* Математически корректны
* Хорошо изучены в литературе
* Имеют гарантии сходимости
* Работают в общем случае

MAPE не проходит эти критерии:

* Нет аналитического решения
* Нет гарантий сходимости
* Не работает с нулевыми/отрицательными значениями
* Смещенные оценки

Альтернативы:

Можно обучить линейную регрессию, минимизируя MAPE через градиентный спуск

## MAE:

`from sklearn.linear_model import QuantileRegressor`

Плюсы MAE:

* Устойчива к выбросам
* Дифференцируема везде кроме нуля
* Нет деления на y_true
* Несмещенные оценки

## HuberLoss

`from sklearn.linear_model import HuberRegressor`

* Комбинирует MSE и MAE
* Устойчива к выбросам
* Гладкая и дифференцируемая

## Когда использовать RMSE

Случай 1: Большие ошибки критичны. Пример: медицинская диагностика

Передозировка на 20 мг опаснее, чем четыре раза по ±5 мг → нужен RMSE.

```python
y_true = [100, 100, 100, 100]  # мг
y_pred_A = [105, 95, 105, 95]  # несколько мелких ошибок
y_pred_B = [100, 100, 100, 80]  # одна большая ошибка

MAE_A = 5.0   MAE_B = 5.0   # одинаково
RMSE_A = 5.0  RMSE_B = 10.0 # B хуже!
```

Прогноз отстатков

Затраты асимметричны: переоценка дороже из-за дополнительных складских расходов.

```python
# Недооценка на 1000 единиц
y_true = 10000
y_pred = 9000
# Упущенная прибыль: 1000 * цена

# Переоценка на 1000 единиц
y_true = 10000  
y_pred = 11000
# Затраты на хранение: 1000 * (цена + складирование)
```

Используйте MAE, но с асимметричной функцией потерь, RMSE сильнее штрафует большие ошибки (из-за квадрата), но в нашем случае затраты растут линейно с размером ошибки, а не квадратично

```python
def asymmetric_loss(y_true, y_pred, overestimation_weight=1.5):
    """
    overestimation_weight > 1.0 если переоценка дороже
    """
    errors = y_true - y_pred
    loss = np.where(errors < 0,  # переоценка (pred > true)
                    np.abs(errors) * overestimation_weight,
                    np.abs(errors))  # недооценка
    return np.mean(loss)
```

Проблемы MAPE:

* Деление на ноль: не определена при y_true = 0
* Асимметрия: сильнее штрафует недооценку, чем переоценку

Ошибка -10% при y=100 → предсказание 90 → MAPE = 10%
Ошибка +10% при y=100 → предсказание 110 → MAPE = 9.1%


* Нет градиента в точках y_pred = y_true (из-за модуля)
* Смещенные предсказания: может приводить к систематической недооценке

Посмотри на распределение ошибок
* Есть выбросы? → MAE
* Нет выбросов, но большие ошибки критичны? → RMSE
* Не уверен? → Huber Loss (золотая середина)
* Попробуй оба и сравни на валидации!

# Выбор ML-метрик для прогнозирования спроса

Цепочка метрик

```
Бизнес-метрика → Таргет → ML-метрика → Loss
```

Выбор таргета

Идеальный вариант: прогнозировать прирост прибыли (Δ Profit), т.к. именно его мы хотим максимизировать.

Проблемы прямого прогноза прибыли
- Прибыль очень волатильна
- Зависит не только от скидок, но и от стоимости закупки

Практичное решение: прогнозировать прирост продаж (Δ Sales), затем через статическую формулу получить прирост прибыли:

$$
\Delta \text{Profit} = \text{Profit}_t - \text{Profit}_{t-1}
$$

$$
\Delta \text{Profit} = (\text{Sales}_{t-1} + \Delta \text{Sales}) \times (\text{price} - \text{discount} - \text{cost}) - \text{Sales}_{t-1} \times (\text{price} - \text{cost})
$$

---

## Ключевые вопросы при выборе ML-метрики

## 1. Линейность функции ошибки
Вопрос: Насколько страшны большие ошибки прогноза?

- Увеличение ошибки с 0 до 100 и с 100 до 200 съедает одинаковое количество прибыли?

Влияние на выбор метрики:
- Если потери растут нелинейно → используйте RMSE (квадратичная функция штрафует большие ошибки сильнее)
- Если потери растут линейно → используйте MAE (линейная функция)

Для прогноза спроса: потери обычно растут линейно → выбираем MAE или схожую метрику

---

## 2. Симметричность ошибки
Вопрос: Одинакова ли стоимость ошибки в сторону недопрогноза и перепрогноза?

- Ошибка в −100 и +100 заказов — это одинаково плохо для бизнеса?

## Типы ошибок

| Тип ошибки | Последствия |
|------------|-------------|
| Недопрогноз (pred < true) | Потеря прибыли из-за дефицита товара на полке |
| Перепрогноз (pred > true) | Избыток товара → затраты на хранение, утилизацию |

---

## Зависимость от категории товара

Обычные товары (непортящиеся)
Правило: Лучше, если товара на полке больше, чем хотят купить

- Пустые полки — самый большой страх ритейлера
- Предпочтителен небольшой перепрогноз

Скоропортящиеся товары
Правило: Лучше недопрогноз, чем перепрогноз

Причины:
- Существенные потери из-за утилизации испорченных продуктов
- Негативный имидж: клиент видит гнилой банан/помидор на полке

Вывод: ML-метрика зависит от категории товара!

---

## Практические метрики для бизнеса

Требования менеджеров
- Метрики должны быть симметричными
- Интервал от 0 до 100%
- Легко интерпретируемыми

Рекомендуемые метрики

MAPE (Mean Absolute Percentage Error)
$$
\text{MAPE} = \frac{\sum |y - y_{\text{true}}|}{\sum y_{\text{true}}}
$$

Показывает среднюю величину ошибки в процентах.

Bias (Смещение)

$$
\text{Bias} = \frac{\sum (y - y_{\text{true}})}{\sum y_{\text{true}}}
$$

Показывает направление ошибки (недо- или перепрогноз).

---

## Целевые значения по категориям

Обычные товары
- Минимизировать MAPE
- При Bias от +1% до +5% (небольшой перепрогноз допустим)

Скоропортящиеся товары
- Минимизировать MAPE
- При Bias от −1% до −5% (небольшой недопрогноз предпочтителен)

---

Важное уточнение: WAPE

Не забываем: метрики нужно измерять для изменения прибыли, а не продаж!

WAPE (Weighted Average Percent Error)

$$
\text{WAPE} = \frac{\sum |(\Delta \text{Profit}) - (\Delta \text{Profit}_{\text{true}})|}{\sum \Delta \text{Profit}_{\text{true}}}
$$

Взвешенная метрика, учитывающая финансовую значимость ошибок.

---

## $R^2$ (Coefficient of Determination)

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$


R^2 Interpretation:
- R^2 = 0: Model no better than mean
- R^2 = 1: Perfect fit
- R^2 < 0: Model worse than mean (overfitting)

Multicollinearity Detection - VIF (Variance Inflation Factor):

$$VIF_j = \frac{1}{1 - R_j^2}$$

Where $R_j^2$ is $R^2$ from regressing $X_j$ on all other predictors.

| VIF | Interpretation |
|-----|---------------|
| 1 | No correlation |
| 1-5 | Moderate |
| 5-10 | High |
| >10 | Severe |

Problems Caused:
- Inflated standard errors
- Unstable coefficient estimates
- Difficult interpretation

Solutions:
- Remove redundant variables
- Use PCA
- Apply Ridge/ElasticNet regularization

Advantages:
- Normalized (0-1 range... usually)
- Indicates proportion of variance explained

Disadvantages:
- Always increases with more features (use Adjusted R²)
- Can be negative on test set

## Adjusted $R^2$

$$R_{adj}^2 = 1 - \frac{(1-R^2)(n-1)}{n-p-1}$$

Where:
- n = sample size
- p = number of predictors

Penalizes adding features that don't improve fit significantly

---

## Итоговые рекомендации

Используйте метрики, нормированные от 0 до 100% — их легко объяснить бизнесу

Учитывайте стоимость ошибки в большую и меньшую стороны (асимметрию)

Адаптируйте целевые значения Bias в зависимости от категории товара

Измеряйте влияние на прибыль (WAPE), а не только на продажи

# RMSE vs MAE: развенчание мифов

## Миф #1: "При скошенных распределениях используйте MAE"

Неправильное понимание
> "При скошенных распределениях таргета с длинным хвостом лучше использовать MAE, так как MAE меньше штрафует за большие ошибки"

Правда
Распределение таргета не имеет значения. Важно только распределение ошибок $(y_{\text{true}} - y_{\text{pred}})$ модели

- Ошибка прогноза не обязательно связана с формой распределения таргета
- У модели на лог-нормальном таргете могут быть ошибки, распределенные нормально
- По этой же причине не нужно сходу использовать RMSLE на скошенных распределениях

---

Что на самом деле прогнозируют MAE и RMSE?

| Метрика | Прогнозирует | Когда использовать |
|---------|--------------|-------------------|
| MAE | Медиану | Разная цена ошибки в ± стороны |
| RMSE | Среднее | Нужно $E(\cdot)$ для расчетов |

## Пример использования RMSE
Прогнозируем среднее, когда оно используется для расчета математического ожидания:

$$
E(\text{выручка}) = \text{цена} \times E(\text{спрос})
$$

Применение: задачи оптимизации цен, промо → используют RMSE

*Также используют Poisson/Tweedie loss, но это отдельная тема*

## Пример использования MAE (квантильной регрессии)
Прогнозируем медиану или любой квантиль при разной цене ошибки в большую/меньшую стороны.

Пример: прогноз спроса для закупки товаров
- Обычно лучше спрогнозировать чуть больше факта → закупить чуть больше товаров
- Можно использовать квантильный loss и прогнозировать, например, 60-й квантиль

---

## Проблемы абсолютных метрик

Абсолютные метрики (RMSE, MAE, RMSLE) хороши простотой, но имеют недостатки:

1. Сложно интерпретировать для бизнеса
   - "У модели MAE = 307"
   - "И что? Хороша ли модель?"

2. Проблемы при разнородных объектах
   - Товар А: 1000 продаж/неделю
   - Товар Б: 5 продаж/неделю
   - Усредненный RMSE — нечто странное

Решение: относительные метрики!

---

## MAPE (Mean Absolute Percentage Error)

$$
\text{MAPE} = \text{mean}\left(\frac{|y_{\text{true}} - y_{\text{pred}}|}{y_{\text{true}}}\right)
$$

## Типы усреднения

| Тип | Формула | Аналог |
|-----|---------|--------|
| Macro average | Метрика по каждому объекту → усредняем | MAPE |
| Micro average | Суммируем числители и знаменатели отдельно → делим | MAE |

$$
\text{Micro MAPE} = \frac{\text{MAE}}{\text{mean}(y_{\text{true}})}
$$

## Проблемы MAPE

Может быть > 1 (> 100%)

Не симметрична: по-разному штрафует недо- и перепрогноз

Плохо работает при маленьких $y_{\text{true}}$

Решение для малых значений:
$$
\text{MAPE}_{\text{offset}} = \text{MEAN}\left(\frac{|y_{\text{true}} - y_{\text{pred}}|}{y_{\text{true}} + \text{offset}}\right)
$$

---

## MAPE: проблема асимметричности

Для задач с $y \geq 0$ (продажи, цены):

Недопрогноз

- Минимальный прогноз = 0
- Максимум: $\frac{|y_{\text{true}} - 0|}{y_{\text{true}}} = 100\%$

Перепрогноз

- Может уходить в бесконечность
- Пример: $\frac{|1 - 8|}{1} = 700\%$

Вывод: MAPE больше штрафует за перепрогноз → ML-модели оптимально недопрогнозировать

---

## SMAPE (Symmetric MAPE) — не так симметричен!

$$
\text{SMAPE} = \frac{|y_{\text{true}} - y_{\text{pred}}|}{(y_{\text{true}} + y_{\text{pred}}) / 2}
$$

Забавный факт: Symmetric MAPE не симметричный!

---

## MALE: аппроксимация MAPE для обучения

Проблема: у MAPE неприятная производная для оптимизации

Решение: аппроксимировать через MALE (Mean Absolute Logarithmic Error)

Математическое обоснование

1. $\ln(1 + f(x)) \approx f(x)$ при $f(x) \to 0$

2. $\ln\left(1 + \frac{|y_{\text{true}} - y_{\text{pred}}|}{y_{\text{pred}}}\right) \to \frac{|y_{\text{true}} - y_{\text{pred}}|}{y_{\text{pred}}} = \text{MAPE}$

3. $\ln(y_{\text{pred}} + |y_{\text{true}} - y_{\text{pred}}|) - \ln(y_{\text{true}}) \to \text{MAPE}$

4. При $\frac{|y_{\text{true}} - y_{\text{pred}}|}{y_{\text{pred}}} \to 0$ выполняется $|y_{\text{true}} - y_{\text{pred}}| \to 0$

5. $\ln(y_{\text{pred}} + |y_{\text{true}} - y_{\text{pred}}|) - \ln(y_{\text{true}}) \to \ln(y_{\text{pred}}) - \ln(y_{\text{true}})$

6. Взяв модуль: MALE $\to$ MAPE при небольших разницах

Хороший рецепт
- Loss = MALE (для обучения)
- Метрика = MAPE (для валидации/теста)

---

## MALE и RMSLE (Logarithmic)

Метрики на логарифмах таргета: $\ln(1 + y)$

Когда используют
- У таргета большой разброс значений
- Пример: продажи товаров (большинство 0-2 шт, редкие >1000 шт)
- Помогает не переобучаться под товары с большими продажами

Дополнительный эффект: логарифмирование вносит нелинейность по фичам:

Для линейной регрессии на $\ln(1 + y)$:
$$
y = \exp(a_0 + a_1 x_1 + a_2 x_2) - 1
$$
Итоговый прогноз нелинеен по фичам из-за экспоненты.

---

## Когда логарифмирование оправдано

Используйте, если:

- Значения таргета $\geq 0$
- Есть теоретическое обоснование логарифма таргета
- Пример: оценка функции спроса → [лог-линейные и лог-лог модели](https://studme.org/198436/ekonomika/logarifmicheskie_lineynye_modeli)

Будьте аккуратны, потому что:

1. Теряется связь с бизнес-задачей
   - Бизнес интересует прогноз $y$, а не $\ln(1+y)$

2. Неконтролируемое влияние фичи
   - Из-за экспоненты небольшое изменение фичи может сильно менять прогноз

3. Распределение ≠ повод
   - Лог-нормальное распределение таргета — не повод использовать MALE и RMSLE

---

## Итоговые рекомендации

Выбирайте метрику на основе распределения ошибок, а не таргета

RMSE для задач оптимизации (нужно среднее для $E(\cdot)$)

MAE/квантильная регрессия при асимметричной цене ошибок

Используйте относительные метрики (MAPE) для презентации бизнесу

MALE как loss, MAPE как метрика на валидации

Логарифмирование — только при теоретическом обосновании


# Refs

- [Дьяконов — AUC ROC (площадь под кривой ошибок)](https://alexanderdyakonov.wordpress.com/2017/07/28/auc-roc-%D0%BF%D0%BB%D0%BE%D1%89%D0%B0%D0%B4%D1%8C-%D0%BF%D0%BE%D0%B4-%D0%BA%D1%80%D0%B8%D0%B2%D0%BE%D0%B9-%D0%BE%D1%88%D0%B8%D0%B1%D0%BE%D0%BA/comment-page-1/)
- [Understanding ROC-AUC](https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5)
- [From Confusion Matrix to Money: How an ML Metric Drove 10% Revenue Growth](https://www.linkedin.com/posts/mikhail-borodastov_from-confusion-matrix-to-money-how-an-share-7444707228915412992-sYkG)
- [PR-AUC vs ROC-AUC](https://www.linkedin.com/feed/update/activity:7250079790420897792)
- [Cross-Validation Techniques](https://www.linkedin.com/feed/update/activity:7082846284994224128/)
- [On Sampled Metrics for Item Recommendations](https://dl.acm.org/doi/pdf/10.1145/3394486.3403226)
- [Mean Average Precision (MAP) for Recommender Systems](http://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html)
- [ROC Curve Drawbacks](https://stats.stackexchange.com/questions/193138/roc-curve-drawbacks)
- [Understanding ROC-AUC: Pros and Cons. Why Brier Score Is a Good Supplement](https://medium.com/@penggongting/understanding-roc-auc-pros-and-cons-why-is-bier-score-a-great-supplement-c7a0c976b679)
- [Evaluation Metrics, ROC Curves, and Imbalanced Datasets](https://aman.ai/primers/ai/evaluation-metrics/#area-under-the-roc-curve-auroc)
- [Imbalanced datasets tricks](https://datascience.stackexchange.com/questions/134389/is-class-imbalance-really-a-problem-in-machine-learning)
- [Disadvantages of Using the Area Under the ROC Curve](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4356897/pdf/330_2014_Article_3487.pdf)
- [Indifference Curve](https://nicolas.kruchten.com/content/2016/01/ml-meets-economics/)
- [Метрики качества ранжирования](https://habr.com/ru/company/econtenta/blog/303458/)
- [Scikit-learn Metrics and Scoring](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Choosing Metrics for Recommender System Evaluation](https://wiki.epfl.ch/edicpublic/documents/Candidacy%20exam/Evaluation.pdf)
- [Statistical Evaluation Metrics](http://iust-projects.ir/post/minidm01/)
- [F1-score vs Accuracy vs ROC-AUC vs PR-AUC](https://neptune.ai/blog/f1-score-accuracy-roc-auc-pr-auc)
- [Evaluation of Ranked Retrieval Results](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html)
- [Evaluation Metrics for Item Recommendation under Sampling](https://arxiv.org/pdf/1912.02263.pdf)
- [Choosing the Right Metric Is a Huge Issue](https://towardsdatascience.com/choosing-the-right-metric-is-a-huge-issue-99ccbe73de61)
- [MRR vs MAP vs NDCG](https://medium.com/swlh/rank-aware-recsys-evaluation-metrics-5191bba16832)
- [Understanding NDCG](https://medium.com/@readsumant/understanding-ndcg-as-a-metric-for-your-recomendation-system-5cd012fb3397)
- [Discounted Cumulative Gain (DCG)](https://en.wikipedia.org/wiki/Discounted_cumulative_gain)
- [F1 vs AUC](https://neptune.ai/blog/f1-score-accuracy-roc-auc-pr-auc/)
- [What Is F1-score? (GetYourGuide)](https://medium.com/tech-getyourguide/whats-a-good-f1-score-b460caf27c90)
- [Ranking Metrics Explained](https://www.linkedin.com/posts/shirin-khosravi-jam_give-me-3-minutes-and-let-me-explain-the-activity-7315630687129157632-tI6X)
- [ROC-AUC Criticism](https://www.linkedin.com/posts/activity-7312050694616719360-OFS1/)
- [Roc_AUC](https://www.linkedin.com/posts/hoang-van-hao_machinelearning-deeplearning-mlengineering-activity-7400323682717528064-6etR)
