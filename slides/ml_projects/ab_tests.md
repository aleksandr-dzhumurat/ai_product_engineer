# Experiment analysis (AB tests & casual inference)

Малые  значения p-value свидетельствуют против нулевой гипотезы

# Основные материалы

Важные моменты

**Ratio метрики** - когда в числителей и знаменателе реализации случайной величины (пример - AOV, когда сумму чеков заказов делим на количество заказов, в этом случае в знаменателе рандом т.к. нормальная ситуация когда больше одного заказа на пользователя)нельзя оценивать с помощью t-test т.к. сессии зависимы (часть сессий принадлежат одному и тому же пользователю), см.лекцию от Яндекса: [Ratio метрики в Yandex](https://www.youtube.com/watch?v=vIdwgJFz5Mk&t=787s). (ещё про [ratio метрики](https://www.youtube.com/watch?v=ObzlKVCiBqI)). **Проблема критерия Стьюдента** с точки зрения ratio-метрик в том, что он не учитывает зависимость данных, а потому **неправильно считает дисперсию.** Как бороться:

- линеаризация (переходим к поюзерной метрике): [обзор](https://www.linkedin.com/feed/update/activity:7170029070368268288?trk=viral_share)
- Bootstrap
- delta-method
- поюзерное усреднение (самый плохой вариант в силу парадокса Симпсона, пример в [статье от Яндекса](https://academy.yandex.ru/journal/kak-provodyat-ab-testy))


```python
import numpy as np
from scipy import stats

# мы провели АВ-тест и собрали данные по ratio-метрике (числители и знаменатели) в контрольной и тестовой группах:

control_numerators, control_denominators
test_numerators, test_denominators

# рассчитаем дисперсию для ratio-метрики в контроле и тесте:

def get_ratio_metric_variance(numerators, denominators):
    n = len(numerators)

    numerators_mean = np.mean(numerators)
    denominators_mean = np.mean(denominators)

    ddof = 1 # delta degrees of freedom
    numerators_variance = np.var(numerators, ddof) 
    denominators_variance = np.var(denominators, ddof)

    covariance = np.cov(numerators, denominators, ddof)[0][1]

    ratio_metric_variance = (numerators_variance/numerators_mean**2 + denominators_variance/denominators_mean**2 - 2*covariance/(numerators_mean**2))*(numerators_mean**2)/(denominators_mean*denominators_mean*n)

    return ratio_metric_variance

control_variance = get_ratio_metric_variance(control_numerators, control_denominators)

test_variance = get_ratio_metric_variance(test_numerators, test_denominators)

control_mean = control_numerators.sum() / control_denominators.sum()
test_mean = test_numerators.sum() / test_denominators.sum()

mean_difference = test_mean - control_mean
mean_difference_relative = mean_difference * 100.0 / control_mean

variance = control_variance + test_variance

z = mean_difference / np.sqrt(variance)
p_value = stats.norm.sf(abs(z))*2

# построим доверительный интервал (ci) для разницы средних в абсолютном и относительном выражении

ci_lower = mean_difference - 1.96 * np.sqrt(variance) 
ci_upper = mean_difference + 1.96 * np.sqrt(variance)

ci_lower_relative = round(ci_lower * 100.0 / control_mean, 5)
ci_upper_relative = round(ci_upper * 100.0 / control_mean, 5)
```

[Family Wise Errow Rates](https://en.wikipedia.org/wiki/Family-wise_error_rate) - нужно применять когда гипотезы зависимы (например строятся на одних и тех же данных)

p-value в случае истинности нулевой гипотезы имеет равномерное распределение

- Пруф
    
    Причина этого действительно заключается в определении альфа как вероятности ошибки первого рода. Мы хотим, чтобы вероятность отклонения истинной нулевой гипотезы была равной альфа; мы отклоняем, когда наблюдаемое p-значение < α, и единственный способ, когда это происходит для любого значения альфа, - это когда p-значение происходит из равномерного распределения. Вся суть использования правильного распределения (нормального, t, f, хи-квадрат и т. д.) заключается в том, чтобы преобразовать статистику теста в равномерное p-значение. Если нулевая гипотеза ложна, то распределение p-значения будет (надеюсь) более сосредоточенным к 0.
    
    P-значение является ключевым понятием в статистическом тестировании гипотез и используется для оценки доказательств против нулевой гипотезы. Нулевая гипотеза обычно утверждает, что эффекта нет или различий между группами нет, и исследователи используют статистические тесты, чтобы определить, есть ли достаточно доказательств для отклонения нулевой гипотезы.
    
    При условии истинности нулевой гипотезы и при правильной постановке статистического теста распределение p-значений следует равномерному распределению в пределах от 0 до 1. Это означает, что при нулевой гипотезе любое p-значение в диапазоне от 0 до 1 равновероятно.
    
    Чтобы понять, почему так происходит, рассмотрим, что p-значение рассчитывается на основе статистики теста, которая является мерой того, насколько наблюдаемые данные отклоняются от того, что ожидается при нулевой гипотезе. Когда нулевая гипотеза верна, статистика теста следует определенному распределению (например, нормальному распределению, t-распределению). Затем p-значение рассчитывается как вероятность получения статистики теста такой же экстремальной, как наблюдаемая, при данном распределении.
    
    При нулевой гипотезе все возможные значения статистики теста равновероятны, и, следовательно, все соответствующие p-значения также равновероятны. Это равномерное распределение p-значений между 0 и 1 является фундаментальным понятием в статистическом выводе. Оно предоставляет основу для определения статистической значимости и помогает исследователям принимать решения о том, следует ли отклонить нулевую гипотезу на основе наблюдаемых данных.
    
    Важно отметить, что это свойство равномерного распределения p-значений при нулевой гипотезе предполагает, что статистический тест корректен, данные соответствуют основным предположениям теста, и в анализе нет других искажений или ошибок. Нарушения этих предположений могут влиять на распределение p-значений и на действительность статистического вывода.
    

Мощность - это вероятность обнаружения эффекта, когда этот эффект действительно присутствует и может быть обнаружен (80 процентов мощности означают, что вы наблюдаете изменение метрики восемь из десяти случаев, когда действительно существует реальный эффект, при мощности 80 процентов, если реальный эффект больше минимально обнаружимого эффекта лечения, у вас есть 80 процентов вероятности его наблюдения)

Peeking problem - Искушение проверить ваши результаты во время выполнения теста, чтобы увидеть, как он себя ведет. Проверяя каждый день можно увидеть достижение статзначимости в силу случайности. Эта проблема наблюдается при подходе “Fixed Horizon Hypothesis Testing”. В других подходах такой проблемы нет: Sequential Hypothesis Testing, Multi-armed Bandits, Bayesian Methods

# Другие источники

* [поста на LinkedIn ratio metrics](https://www.linkedin.com/pulse/ratio-%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8-%D0%B2-ab-%D1%82%D0%B5%D1%81%D1%82%D0%B0%D1%85-polina-egubova-6tedc)
* [Yandex practicum: основы статистики и  АБ-тестов](https://practicum.yandex.ru/statistics-basic/?from=catalog) 



# Как проводить А/Б тесты

**Автор:** Arman Zhylkaidarov  
**Источник:** Курс "Основы статистики и А/Б тестирования" от Яндекс Практикум

---

## Содержание

0. Алгоритм проведения А/Б теста
1. Как сформулировать гипотезу
2. Как определить параметры выборки
3. Как подобрать статистический критерий
4. Как посчитать размер выборки
5. Как проверить тест на валидность
6. Как интерпретировать результаты

---

## 0. Алгоритм проведения А/Б теста

### 1. Гипотеза
   - Формулируем гипотезу
   - Выбираем метрики

### 2. Дизайн эксперимента
   - Выбираем параметры выборки
   - Рассчитываем MDE
   - Считаем размер выборки и длительность эксперимента

### 3. Запуск и сбор данных
   - Ставим задачу на разработку
   - Запускаем тест, проверяем его на корректность и мониторим на критические падения
   - Собираем данные (**Не останавливаем до сбора выборки. Peeking problem!**)

### 4. Анализ
   - Проверяем валидность эксперимента
   - Рассчитываем результаты и принимаем решение о раскатке фичи

---

## 1. Как сформулировать гипотезу

### Формула хорошей гипотезы

> *Если [изменение в продукте], то [изменение в метрике], потому что [решение пользовательской проблемы].*

### Пример плохой гипотезы

❌ Если добавить чат, то мы увеличим контакты.

### Пример хорошей гипотезы

✅ Если добавить чат, то конверсия в контакт увеличится на 100%, так как не всем пользователям удобно звонить продавцу машины.

---

## 2. Как определить параметры выборки

### 1. Определяем целевую аудиторию эксперимента

Примеры:
- Все пользователи iOS/Android
- Все пользователи сайта
- Все продавцы с 1+ объявлением
- Все владельцы авто в гараже

### 2. Определяем условия попадания в эксперимент

Примеры:
- Когда пользователь зашел в листинг
- Когда пользователь зашел в карточку объявления
- Когда пользователь дошел до превью созданного объявления

### 3. Определяем способ разбиения по группам

#### a. Юнит разбиения и юнит эксперимента
- Если юниты разбиения и эксперимента не соответствуют друг другу → **метрика отношения**

#### b. Соотношение между группами
- **50/50** — стандартное
- **80/20** — когда изменение слишком рисковое

#### c. Количество тестируемых групп
- **А/Б** — стандартный тест
- **А/А/Б** — когда нужна дополнительная проверка на отсутствие внешних факторов
- **А/Б/С** — когда новых вариантов изменений несколько

> ⚠️ **Разбиение должно происходить параллельно и случайным образом.**

---

## 3. Как подобрать статистический критерий

| Типы метрик | Примеры | Статистические критерии |
|-------------|---------|-------------------------|
| **Количественные** | Денежные метрики (ARPU, ARPPU), Количество действий (просмотры, товары в корзине), Технические (время загрузки) | t-test (Стьюдента), Бакетный тест, Манна-Уитни |
| **Конверсионные** | Конверсия в просмотр, конверсия в заказ | Z-test |
| **Метрики отношения** | Клики на просмотр, Просмотры за сессию, AOV | Дельта метод, Бутстрэп, Линеаризация |

---

### Критерий Стьюдента (t-test)

#### Условия применения

- Независимость наблюдений
- Нормальность распределения выборочных средних

#### Алгоритм применения

**1. Формулируем $H_0$, $H_1$**

**2. Выбираем уровень значимости $\alpha$**

**3. Считаем разность выборочных средних $\bar{X}_1$ и $\bar{X}_2$**

**4. Считаем ESE (стандартная ошибка среднего)**

$$
ESE = \sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}
$$

**5. Считаем t-статистику**

Если нулевая гипотеза о **нулевой** разнице между группами:

$$
t = \frac{\bar{x}_1 - \bar{x}_2}{ESE}
$$

Если нулевая гипотеза о **ненулевой** разнице между группами:

$$
t = \frac{(\bar{x}_1 - \bar{x}_2) - (\mu_1 - \mu_2)}{ESE}
$$

**6. Считаем количество степеней свободы**

Когда размеры выборок **не равны**:

$$
df = \frac{(ESE_1^2 + ESE_2^2)^2}{\frac{ESE_1^4}{n_1 - 1} + \frac{ESE_2^4}{n_2 - 1}}
$$

Когда размеры выборок **равны**:

$$
df = \frac{(n-1) \cdot (s_1^2 + s_2^2)^2}{s_1^4 + s_2^4}
$$

Формула стандартной ошибки среднего для одной группы:

$$
ESE_1^2 = \left(\frac{s_1}{\sqrt{n_1}}\right)^2 = \frac{s_1^2}{n_1}
$$

**7. Считаем p-value через калькулятор**

**8. Сравниваем p-value с уровнем значимости**

---

### Бакетный тест

Применяется, если выборочные средние распределены **ненормально**.

#### Алгоритм применения

1. Разбить данные в выборке на **100 бакетов** (минимум 30 наблюдений в каждом)
2. По каждому бакету посчитать среднее значение
3. Посчитать среднее и дисперсию для каждой группы бакетов
4. Применить t-test

---

### Z-test для пропорций

#### Алгоритм применения

**1. Формулируем $H_0$, $H_1$**

**2. Выбираем уровень значимости $\alpha$**

**3. Считаем разность выборочных средних $p_1 - p_2$**

$$
p = \frac{\text{кол-во дошедших до действия}}{\text{кол-во в группе}}
$$

**4. Считаем ESE**

$$
ESE = \sqrt{\frac{\bar{p}_1 \cdot (1 - \bar{p}_1)}{n_1} + \frac{\bar{p}_2 \cdot (1 - \bar{p}_2)}{n_2}}
$$

Формула дисперсии для конверсий (распределение Бернулли):

$$
Var_{Bernoulli} = p \cdot (1 - p)
$$

**5. Считаем z-статистику**

Если нулевая гипотеза о **ненулевой** разнице:

$$
z = \frac{(\bar{p}_1 - \bar{p}_2) - (p_1 - p_2)}{ESE}
$$

Если нулевая гипотеза о **нулевой** разнице:

$$
z = \frac{\bar{p}_1 - \bar{p}_2}{ESE}
$$

**6. Считаем p-value**

**7. Сравниваем p-value с уровнем значимости**

---

### Другие методы (TBD)

- **U-критерий Манна-Уитни** — непараметрический тест
- **Бутстрэп** — для метрик отношения
- **Дельта метод** — для метрик отношения
- **Линеаризация** — для метрик отношения

---

## 4. Как посчитать размер выборки

### Основная формула

$$
n = \frac{(Var_{control} + Var_{test}) \cdot (z_{\alpha/2} + z_\beta)^2}{MDE^2}
$$

Где:
- $(z_{\alpha/2} + z_\beta)^2$ — квантиль стандартного нормального распределения
- $MDE$ — минимальный размер эффекта, который мы хотим зафиксировать

### Упрощённая формула

При $\alpha = 5\%$, мощность $= 80\%$, одинаковая дисперсия:

$$
n = \frac{2 \cdot VAR_{hist} \cdot (-2.802)^2}{MDE^2}
$$

### Алгоритм расчета длительности эксперимента

1. Выбрать уровень значимости, мощность и MDE
2. Составить список потенциальной длительности эксперимента (1 неделя, 2 недели, и т.д.)
3. Для каждой длительности рассчитать:
   - Среднее значение метрики
   - Среднюю дисперсию метрики
   - Среднее кол-во пользователей
   - MDE:

$$
MDE = -(z_{\alpha/2} + z_\beta) \cdot \sqrt{\frac{4 \cdot Var_{hist}}{n_{hist}}}
$$

$$
MDE = -(-2.802) \cdot \sqrt{\frac{4 \cdot VAR_{hist}}{N_{hist}}}
$$

4. Выбрать ту длительность, для которой рассчитанный MDE наиболее близок, но **не превышает** минимальный желаемый эффект

---

## 5. Как проверить тест на валидность

### А/А тест перед проведением теста

1. Берем код распределения пользователей по группам
2. Берем пользователей за период с той же длительностью до начала эксперимента
3. Разбиваем их по группам, считаем p-value для нашей метрики
4. Проводим третий шаг **1000 раз**. Каждый раз записываем p-value
5. Считаем долю **False Positive Rate (FPR)**. Должен быть **< 5%**
6. Рисуем гистограмму распределения p-value

### А/А тест после проведения теста

1. Собираем выборку на предпериоде того же размера
2. Считаем значение метрики и дисперсию
3. Считаем p-value. Должен быть **> 0.05**

### Sample Ratio Mismatch (SRM)

1. Считаем размер контрольной и тестовой группы
2. Считаем Хи-квадрат
3. Считаем p-value. Должен быть **> 0.05**

---

## 6. Как интерпретировать результаты

> ⚠️ При подведении результатов, статистически значимое отклонение надо считать **не только от нуля, но и от MDE**.

### Матрица решений

| Отклонение от 0? | Отклонение от MDE? | Что делать |
|------------------|-------------------|------------|
| ❌ Да, отрицательное | ❌ Да, отрицательное | **Не раскатывать.** Красный тест. |
| ➖ Нет | ❌ Да, отрицательное | **Не раскатывать.** Эффект меньше желаемого. |
| ➖ Нет | ➖ Нет | **Перезапустить** с большей выборкой или отказаться. |
| ✅ Да, положительное | ➖ Нет | **Желательно перезапустить.** Можно рискнуть и раскатить. |
| ✅ Да, положительное | ✅ Да, положительное | **Раскатить фичу!** ✅ |

### Подробное описание случаев

**Случай 1** — Отрицательное значимое отклонение от 0 и от MDE
> Красный тест. Фичу не раскатывать.

**Случай 2** — Нет значимого отклонения от 0, но есть отрицательное от MDE
> Эффект меньше желаемого. Фичу не раскатывать.

**Случай 3** — Нет значимых отклонений ни от 0, ни от MDE
> Неопределённость. Перезапустить тест с большей выборкой или отказаться от фичи.

**Случай 4** — Положительное значимое от 0, но нет значимого от MDE
> Эффект > 0, но неизвестно, достаточен ли он. Желательно перезапустить. Можно рискнуть.

**Случай 5** — Положительное значимое и от 0, и от MDE
> Идеальная ситуация. **Раскатить фичу!**

---

## Краткая шпаргалка

### Формулы

| Что считаем | Формула |
|-------------|---------|
| Размер выборки | $n = \frac{2\sigma^2 (z_{\alpha/2} + z_\beta)^2}{MDE^2}$ |
| t-статистика | $t = \frac{\bar{x}_1 - \bar{x}_2}{ESE}$ |
| z-статистика | $z = \frac{\bar{p}_1 - \bar{p}_2}{ESE}$ |
| ESE (t-test) | $ESE = \sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}$ |
| ESE (z-test) | $ESE = \sqrt{\frac{p_1(1-p_1)}{n_1} + \frac{p_2(1-p_2)}{n_2}}$ |
| Дисперсия Бернулли | $Var = p(1-p)$ |

### Стандартные параметры

| Параметр | Значение |
|----------|----------|
| $\alpha$ (уровень значимости) | 0.05 (5%) |
| $1 - \beta$ (мощность) | 0.80 (80%) |
| $z_{\alpha/2}$ | 1.96 |
| $z_\beta$ | 0.84 |
| $(z_{\alpha/2} + z_\beta)^2$ | 7.84 |

# All about Sample-Size Calculations for A/B Testing: Novel Extensions & Practical Guide

**Authors:** Jing Zhou, Jiannan Lu, Anas Shallah (Apple)

**Published:** CIKM '23, October 21–25, 2023, Birmingham, United Kingdom

---

## Abstract

This paper addresses fundamental gaps in sample size calculation for A/B testing by developing new methods for:
- Correlated data (unit of analysis ≠ unit of randomization)
- Absolute vs. relative treatment effects
- Minimal observed difference (MOD) concept

All methods are accompanied by mathematical proofs, illustrative examples, and simulations.

---

## 1. Introduction

A/B testing has become the go-to methodology in the tech industry due to:
- **Causal conclusions** — gold standard like randomized clinical trials
- **Large-scale traffic** — real-time data accumulation
- **Authenticity** — "listening to your customers"

**Key gap:** While there's growing literature on increasing sensitivity, there's a lack of in-depth studies on power analysis / sample size calculation, particularly for:
- Correlated observations
- Percentage treatment effect ("relative lift")

---

## 2. Existing Work

### 2.1 Sample Size for Independent Data

For continuous outcomes with two-sample t-test:

$$
H_0: \mu_x = \mu_y \quad \text{vs.} \quad H_1: \mu_x \neq \mu_y
$$

**Standard sample size formula (per arm):**

$$
n = \frac{2\sigma^2 \cdot (z_{1-\alpha/2} + z_{1-\beta})^2}{\delta^2}
$$

Where:
- $\sigma$ — standard deviation
- $\alpha$ — type I error rate
- $\beta$ — type II error rate (power = $1 - \beta$)
- $\delta = \mu_y - \mu_x$ — ATE (average treatment effect) / MDE (minimal detectable effect)
- $z_k$ — k-th percentile of standard normal distribution

**Rule of thumb:** With $\alpha = 0.05$ and power $= 0.8$:

$$
n \approx \frac{16\sigma^2}{\delta^2}
$$

**For binary outcomes:**

$$
n = \frac{2p_{pool}(1-p_{pool}) \cdot (z_{1-\alpha/2} + z_{1-\beta})^2}{\delta^2}
$$

Where $p_{pool} = (p_x + p_y)/2$

### 2.2 Analysis Methods for Correlated Data

Correlated data occurs when **unit of randomization** (e.g., user) is less granular than **unit of analysis** (e.g., session).

**Delta Method:** For user $i$ with $N_i$ observations $X_{ij}$:

$$
\bar{X} = \frac{\sum_{i,j} X_{ij}}{\sum_i N_i} = \frac{\bar{S}}{\bar{N}}
$$

Where $S_i = \sum_j X_{ij}$ (sum of observations for user $i$)

**Variance using bivariate Delta method:**

$$
\text{Var}\left(\frac{\bar{S}}{\bar{N}}\right) \approx \frac{1}{k\mu_N^2} \left( \sigma_S^2 - 2\frac{\mu_S}{\mu_N}\sigma_{SN} + \frac{\mu_S^2}{\mu_N^2}\sigma_N^2 \right)
$$

---

## 3. Sample Size for Correlated Data

### 3.1 Motivation

Standard sizing formulas assume i.i.d., resulting in **under-powered experiments** when data is correlated.

### 3.2 Main Result

**Required number of users (randomization units):**

$$
k = \frac{2h \cdot (z_{1-\alpha/2} + z_{1-\beta})^2}{\delta^2}
$$

Where:

$$
h = \frac{1}{\mu_N^2} \left( \sigma_S^2 - 2\frac{\mu_S}{\mu_N}\sigma_{SN} + \frac{\mu_S^2}{\mu_N^2}\sigma_N^2 \right)
$$

**Key insight:** This formula uses $h$ (accounting for clustering) instead of $\sigma^2$ (assuming i.i.d.).

**Important:** $h$ is NOT scale-free — you must pre-set experiment duration and estimate $h$ from historical data of the same duration.

#### Theorem 3.1 (Continuous Outcomes)

Under the assumption that treatment has:
- No effect on $N_i$ (number of sessions per user)
- No effect on $\sigma_i$ (within-user variance)
- Constant effect $\delta$ on $\mu_i$

Then $h$ is constant between treatment and control.

#### Theorem 3.2 (Binary Outcomes)

For binary outcomes where $\sigma_i^2 = p_i(1-p_i)$, the difference in $h$ between treatment and control is:

$$
\delta_h = \delta \frac{(1-\delta)\hat{\mu}_N - 2\hat{\mu}_S}{\hat{\mu}_N^2}
$$

In practice, $\delta$ is typically small, so $\delta_h \approx 0$.

### 3.3 Illustrating Example

**Steps for user-randomized experiment with session conversion rate:**

1. **Determine duration** (e.g., 2 weeks)
2. **Collect session-level data** with user ID, session ID, metric $X_{ij}$
3. **Aggregate to user-level:**
   - $N_i$ = count of sessions per user
   - $S_i = \sum_j X_{ij}$ = sum of metric per user
4. **Estimate components of $h$:**
   - $\hat{\mu}_S$ = average converted sessions per user
   - $\hat{\mu}_N$ = average sessions per user
   - $\hat{\sigma}_S^2$, $\hat{\sigma}_N^2$, $\hat{\sigma}_{SN}$
5. **Calculate $k$** using the formula

**Example calculation:**
- $h = 0.151$
- With 5% ATE, 80% power, 5% type I error
- Required users: $k = 949$ per arm
- Required sessions: $k \cdot \mu_N = 3,986$

**Comparison:** Standard i.i.d. formula would suggest only 1,440 sessions — clearly under-powered!

### 3.4 Simulation Results

| Case | Proposed Method |  | Standard Method (i.i.d.) |  |
|------|-----------------|--|--------------------------|--|
|      | Type I Error | Power | Type I Error | Power |
| I ($\lambda=5$) | 0.049 | 0.809 | 0.114 / 0.051 | 0.758 / 0.681 |
| II ($\lambda=20$) | 0.051 | 0.816 | 0.260 / 0.051 | 0.697 / 0.581 |

Standard method suffers both inflated type I error and reduced power.

---

## 4. Sample Size for Relative Lift

### 4.1 Motivation

Sometimes **relative lift** $\delta_{rel} = (\mu_y - \mu_x)/\mu_x$ is more important than absolute lift.

**Question:** Can we simply substitute $\delta = \delta_{rel} \cdot \mu_x$ into standard formulas?

**Answer:** Not always accurate, especially for large relative lifts.

### 4.2 Relative Lift from Independent Data

**For continuous outcomes:**

$$
n_{rel} = \left( \frac{1}{\mu_x^2} + \frac{\mu_y^2}{\mu_x^4} \right) \cdot \sigma^2 \cdot \frac{(z_{1-\alpha/2} + z_{1-\beta})^2}{\delta_{rel}^2}
$$

**For binary outcomes:**

$$
n_{rel} = \left( \frac{1}{p_x^2} + \frac{p_y^2}{p_x^4} \right) \cdot p_{pool}(1-p_{pool}) \cdot \frac{(z_{1-\alpha/2} + z_{1-\beta})^2}{\delta_{rel}^2}
$$

### 4.3 Comparison with Absolute Lift

The difference factor is: $1 + \mu_y^2/\mu_x^2$ vs. $2$

Since $\mu_y/\mu_x = \delta_{rel} + 1$:

| Relative Lift | Sample Size Difference |
|---------------|------------------------|
| 1% | +1% (negligible) |
| 10% | +10.5% |
| 20% | +22% |

**For negative lifts** (e.g., churn rate decrease):
| Relative Lift | Sample Size Difference |
|---------------|------------------------|
| -10% | -9.5% (requires less) |
| -20% | -18% |

### 4.4 Relative Lift from Correlated Data

Replace $\sigma^2$ with $h$:

$$
k_{rel} = \left( \frac{1}{\mu_x^2} + \frac{\mu_y^2}{\mu_x^4} \right) \cdot h \cdot \frac{(z_{1-\alpha/2} + z_{1-\beta})^2}{\delta_{rel}^2}
$$

---

## 5. Minimal Observed Difference (MOD)

### 5.1 Difference between ATE and MOD

- **$\Delta_{ATE}$** (= MDE) — used at design stage to power the experiment
- **$\Delta_{MOD}$** — minimum observed difference that will be statistically significant ($p < 0.05$)

### 5.2 MOD Determination

$$
|\Delta_{MOD}| \approx \frac{z_{1-\alpha/2}}{z_{1-\alpha/2} + z_{1-\beta}} |\Delta_{ATE}|
$$

**Rules of thumb:**
- With 80% power, 5% type I error: $|\Delta_{MOD}| \approx 0.7 \cdot |\Delta_{ATE}|$
- With 90% power, 5% type I error: $|\Delta_{MOD}| \approx 0.6 \cdot |\Delta_{ATE}|$

**Important:** Ensure $\Delta_{MOD}$ is business meaningful!

### 5.3 Example

- Control conversion: 10%
- Treatment conversion: 12%
- $\Delta_{ATE} = 2\%$
- $\Delta_{MOD} \approx 0.7 \times 2\% = 1.4\%$

Any observed difference ≥ 1.4% would yield $p < 0.05$.

---

## 6. Best Practices

### 6.1 Is Balanced Design Always Best?

**Standard assumption:** 1:1 allocation minimizes total sample size.

**But:** In online experiments, control samples are cheap. If duration is the concern:

**Power formula:**

$$
\text{Power} = 1 - \Phi^{-1}\left( z_{1-\alpha/2} - \delta\sqrt{n_{all} \cdot f(1-f)}/\sigma \right)
$$

Where $f = n_y/n_{all}$ is treatment allocation proportion.

| Allocation | Duration Reduction | Total Sample Increase |
|------------|--------------------|-----------------------|
| 1:1 ($f=0.5$) | baseline | baseline |
| 1:2 ($f=0.33$) | -25% | +12.5% |
| 1:4 ($f=0.2$) | -37.5% | +56.25% |
| 1:9 ($f=0.1$) | -44.4% | +178% |

**Recommendation:** Don't go below $f = 0.2$ (diminishing returns + need reliable treatment variance).

### 6.2 Does Skewed Data Matter?

**Common confusion:** Normal assumption refers to the **metric** (mean/rate), not the raw data.

**Central Limit Theorem:** Sample of 30+ is usually sufficient for the mean to be approximately normal.

**Bottom line:** Regular t-test is usually fine even with skewed data. Non-parametric tests are less powerful and test different hypotheses (median or distribution equality).

**Mitigations for high variance from skewed data:**
- Capping
- Binarization
- Conversion rate instead of mean spent

### 6.3 Total Metrics vs Mean Metrics

**Problem 1:** T-test tests **means**, not sums. Sample sizes are rarely exactly equal between arms.

**Problem 2:** Variance of total increases with sample size: $\text{Var}(\sum X_i) = n\sigma^2$
vs. variance of mean decreases: $\text{Var}(\bar{X}) = \sigma^2/n$

**Recommendation:** Always use **mean metrics**, never total metrics.

### 6.4 Other Pitfalls

**Bad practice:** Using confidence interval width directly to compute sample size.

**Problem:** No power involved! This gives MOD-related estimate, smaller than properly powered sample size.

**Recommendation:** Use standard sample size formulas from Sections 2-4.

---

## 7. Summary of Formulas

### Independent Data

| Outcome | Formula |
|---------|---------|
| Continuous (absolute) | $n = \frac{2\sigma^2 (z_{1-\alpha/2} + z_{1-\beta})^2}{\delta^2}$ |
| Binary (absolute) | $n = \frac{2p_{pool}(1-p_{pool})(z_{1-\alpha/2} + z_{1-\beta})^2}{\delta^2}$ |
| Relative lift | $n_{rel} = \left(\frac{1}{\mu_x^2} + \frac{\mu_y^2}{\mu_x^4}\right) \sigma^2 \frac{(z_{1-\alpha/2} + z_{1-\beta})^2}{\delta_{rel}^2}$ |

### Correlated Data

| Type | Formula |
|------|---------|
| Absolute lift | $k = \frac{2h (z_{1-\alpha/2} + z_{1-\beta})^2}{\delta^2}$ |
| Relative lift | $k_{rel} = \left(\frac{1}{\mu_x^2} + \frac{\mu_y^2}{\mu_x^4}\right) h \frac{(z_{1-\alpha/2} + z_{1-\beta})^2}{\delta_{rel}^2}$ |

Where:

$$
h = \frac{1}{\mu_N^2} \left( \sigma_S^2 - 2\frac{\mu_S}{\mu_N}\sigma_{SN} + \frac{\mu_S^2}{\mu_N^2}\sigma_N^2 \right)
$$

### Quick Reference

| Parameter | 80% Power | 90% Power |
|-----------|-----------|-----------|
| $z_{1-\beta}$ | 0.84 | 1.28 |
| $z_{1-\alpha/2}$ (5%) | 1.96 | 1.96 |
| $(z_{1-\alpha/2} + z_{1-\beta})^2$ | 7.84 | 10.50 |
| MOD/ATE ratio | 0.70 | 0.60 |

---

# Yandex A/B Testing Question

**Source:** LinkedIn discussion

---

## Problem

According to the survey, in City A, the percentage of people watching television is **15%**, while in City B, it is **17%**.

**Question:** Can it be said that fewer people in City A watch television based on this data?

---

## Answer

Based on the given information, **it cannot be conclusively stated** that the percentage of people watching television in City A is lower than in City B.

Additional statistical analysis is required to determine if the difference is statistically significant.

---

## Steps for Statistical Analysis

### 1. Formulate Hypotheses

- **Null Hypothesis ($H_0$):** There is no difference in the percentage of people watching television between City A and City B.
- **Alternative Hypothesis ($H_1$):** There is a significant difference in the percentage of people watching television between City A and City B.

### 2. Define Significance Level

Choose a significance level $\alpha$ (commonly 0.05 or 0.01). This represents the probability of rejecting the null hypothesis when it is true.

### 3. Collect Data

Ensure that the survey data is representative and collected using a sound methodology.

### 4. Perform a Two-Sample Hypothesis Test

Utilize an appropriate statistical test for comparing two independent samples. For proportions, a **Z-test** or **Chi-square test** might be applicable.

### 5. Calculate Test Statistic

Calculate the test statistic based on the chosen statistical test. This value will be used to determine the p-value.

### 6. Determine P-value

The p-value represents the probability of obtaining results as extreme as the observed results if the null hypothesis is true. A lower p-value indicates stronger evidence against the null hypothesis.

### 7. Compare P-value to Significance Level

- If $p\text{-value} \leq \alpha$ → **Reject** the null hypothesis
- If $p\text{-value} > \alpha$ → **Fail to reject** the null hypothesis

### 8. Interpret Results

Provide a conclusion based on the statistical analysis. If the null hypothesis is rejected, it suggests a significant difference in the percentage of people watching television between City A and City B.

### 9. Consider Practical Significance

Assess not only statistical significance but also practical significance. Even if a difference is statistically significant, it may not be practically significant if the effect size is small.

### 10. Report Findings

Communicate the results, including the statistical test used, p-value, and any relevant effect size measures, in a clear and understandable manner.

---

## Z-test for Proportions

The Z-test for proportions is appropriate when you have two independent samples and want to assess whether the difference between the proportions is statistically significant.

### Steps

#### 1. Formulate Hypotheses

- **$H_0$:** The proportion of people watching television is the same in City A and City B ($p_1 = p_2$)
- **$H_1$:** There is a significant difference in the proportion ($p_1 \neq p_2$)

#### 2. Calculate Proportions

Calculate the sample proportions for each city:

$$
p_1 = \frac{\text{Number of people watching TV in City A}}{\text{Total number of respondents in City A}}
$$

$$
p_2 = \frac{\text{Number of people watching TV in City B}}{\text{Total number of respondents in City B}}
$$

#### 3. Calculate the Standard Error of the Difference

Compute the standard error using the **pooled proportion**:

$$
SE = \sqrt{p(1-p)\left(\frac{1}{n_1} + \frac{1}{n_2}\right)}
$$

Where $p$ is the pooled sample proportion:

$$
p = \frac{n_1 p_1 + n_2 p_2}{n_1 + n_2}
$$

And $n_1$, $n_2$ are the sample sizes for City A and City B, respectively.

#### 4. Calculate the Z-statistic

$$
Z = \frac{p_1 - p_2}{SE}
$$

#### 5. Determine the P-value

Use the Z-statistic to find the corresponding p-value from the standard normal distribution table.

#### 6. Compare P-value to Significance Level

If $p\text{-value} \leq \alpha$ (e.g., 0.05), reject the null hypothesis.

#### 7. Interpret Results

Conclude whether there is sufficient evidence to suggest a significant difference in the proportion of people watching television between City A and City B.

---

## Chi-Square Test for Independence

The chi-square test for independence assesses the association between two categorical variables.

### Steps

#### 1. Formulate Hypotheses

- **$H_0$:** There is no association between the city and the habit of watching television.
- **$H_1$:** There is a significant association between the city and the habit of watching television.

#### 2. Create a Contingency Table

Organize the data into a contingency table:

|        | Watching TV | Not Watching TV | Total |
|--------|-------------|-----------------|-------|
| City A | $O_{11}$    | $O_{12}$        | $R_1$ |
| City B | $O_{21}$    | $O_{22}$        | $R_2$ |
| Total  | $C_1$       | $C_2$           | $N$   |

Populate the table with the observed frequencies based on your survey data.

#### 3. Calculate Expected Frequencies

The expected frequency for each cell is calculated as:

$$
E_{ij} = \frac{(\text{Row } i \text{ Total}) \times (\text{Column } j \text{ Total})}{\text{Grand Total}}
$$

Compute the expected frequencies for all cells in the table.

#### 4. Calculate the Chi-Square Statistic

$$
\chi^2 = \sum \frac{(O_{ij} - E_{ij})^2}{E_{ij}}
$$

Where:
- $O_{ij}$ is the observed frequency in cell $(i, j)$
- $E_{ij}$ is the expected frequency in cell $(i, j)$

#### 5. Determine Degrees of Freedom

For a contingency table:

$$
df = (rows - 1) \times (columns - 1)
$$

For a 2×2 contingency table: $df = 1$

#### 6. Consult the Chi-Square Distribution Table

Compare the calculated chi-square statistic with the critical value from the chi-square distribution table at the chosen significance level (e.g., 0.05) and degrees of freedom.

#### 7. Make a Decision

- If $\chi^2_{calculated} > \chi^2_{critical}$ → **Reject** the null hypothesis
- If $\chi^2_{calculated} \leq \chi^2_{critical}$ → **Fail to reject** the null hypothesis

#### 8. Interpret Results

Conclude whether there is sufficient evidence to suggest an association between the city and the habit of watching television.

---

## Summary

| Test | When to Use | Formula |
|------|-------------|---------|
| **Z-test** | Comparing two proportions | $Z = \frac{p_1 - p_2}{SE}$ |
| **Chi-square** | Testing independence in contingency table | $\chi^2 = \sum \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$ |

### Key Points

1. **Sample size matters** — Without knowing the sample sizes, we cannot determine if the 2% difference (15% vs 17%) is statistically significant
2. **Statistical vs. practical significance** — Even a statistically significant difference may not be practically meaningful
3. **Confidence intervals** — Consider computing confidence intervals for each proportion to assess overlap