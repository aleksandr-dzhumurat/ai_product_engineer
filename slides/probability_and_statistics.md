# Bootstrap

3. Bootstrap

Bootstrap — это метод оценки статистик выборки путём **повторной выборки с возвращением** из имеющихся данных.

Есть выборка из N элементов. Сделай 1000 раз: возьми N элементов случайно **с возвращением** → получи 1000 "псевдовыборок" → вычисли нужную статистику на каждой → посмотри на распределение.

```python
import numpy as np

data = np.array([3, 7, 2, 9, 1, 5, 8, 4])
n_bootstrap = 1000
means = []

for _ in range(n_bootstrap):
    sample = np.random.choice(data, size=len(data), replace=True)
    means.append(sample.mean())

# Доверительный интервал
ci = np.percentile(means, [2.5, 97.5])
```

- **Доверительный интервал** для любой метрики (accuracy, F1, AUC) — без предположений о распределении
- **Сравнение моделей**: у модели A accuracy 0.82, у B — 0.84. Значимо ли это? Bootstrap скажет.
- **Оценка дисперсии** любой статистики

---

Bootstrap в ML — Bagging

**Bagging (Bootstrap Aggregating)** — обучаем несколько моделей на разных bootstrap-выборках, усредняем предсказания.

```
Исходные данные (N строк)
  ↓
Выборка 1 (N строк, с повторами) → модель 1 → предсказание 1
Выборка 2 (N строк, с повторами) → модель 2 → предсказание 2
...
Выборка K (N строк, с повторами) → модель K → предсказание K
  ↓
Усреднение (или голосование) → финальный ответ
```

**Random Forest = Bagging + случайные признаки** — классический пример.

**Зачем это снижает дисперсию:**
- Разные выборки → модели делают разные ошибки
- Усреднение ошибок, которые не коррелированы → дисперсия падает в K раз

---

Out-of-Bag (OOB) оценка

При выборке с возвращением ~36.8% объектов не попадают в выборку. Их можно использовать как валидационную выборку бесплатно — это OOB score в Random Forest.

## Why ~36.8% of the Dataset Ends Up OOB

**Step 1: one specific object**

Each bootstrap sample draws $n$ objects with replacement from a dataset of size $n$.
The probability that one specific object $x_i$ is never picked across all $n$ draws:

$$P_{\text{OOB}} = \left(1 - \frac{1}{n}\right)^n \xrightarrow{n \to \infty} e^{-1} \approx 0.368$$

This is a statement about a **single object** — not the whole dataset yet.

---

**Step 2: extending to all $n$ objects**

Every object has the **same** $P_{\text{OOB}}$. By linearity of expectation, the expected number of OOB objects across the entire dataset:

$$E[\text{OOB count}] = \sum_{i=1}^{n} P_{\text{OOB}}(x_i) = n \cdot \left(1 - \frac{1}{n}\right)^n \approx \frac{n}{e}$$

---

**Step 3: the expected fraction**

Dividing by $n$, because we want the fraction (proportion), not the count.

$$\frac{E[\text{OOB count}]}{n} = \left(1 - \frac{1}{n}\right)^n \approx 0.368$$

Since every object has the same $\approx 0.368$ probability of being left out, the expected **fraction** of the dataset that ends up OOB is also $\approx 0.368$.

The logic is the same as: if every coin has a 50% chance of heads, then on average 50% of $n$ coins will land heads — regardless of $n$.

# Практические задачи с теорией вероятностей


Будемм применять теорию к задачам по system design. Во всех этих задачах теория вероятностей помогает:

1. **Оценить риски** (SLA, fraud, data loss)
2. **Подобрать пороги** (rate limiting, connection pools)
3. **Спроектировать retry/fallback** (exponential backoff)
4. **Понять trade-offs** (false positives vs false negatives)
5. **Capacity planning** (M/M/c, Little's Law)

Без этого ты либо over-provision (тратишь деньги), либо under-provision (теряешь availability).

## 1. SLA и availability в distributed systems

**Задача:** У вас микросервисная архитектура с 5 сервисами в цепочке. Каждый сервис имеет uptime 99.9%. Какой будет итоговый SLA?

**Теория:** Независимые вероятности, умножение вероятностей.

$$P(\text{система работает}) = (0.999)^5 = 0.995 \approx 99.5\%$$

**Практический вывод:** 
- Цепочка из N сервисов → SLA деградирует
- Для 99.99% итогового SLA каждый сервис должен иметь 99.998%
- Нужны retries, circuit breakers, fallbacks

---

## 2. Cache hit rate и память

**Задача:** У вас есть 10GB RAM под cache, запросы идут по Zipf distribution (80% запросов к 20% ключей). Сколько данных кешировать?

**Теория:** 
- Распределение частот запросов
- Оптимизация hit rate vs память

**Практический подход:**
```
Если кешируем топ-20% ключей → hit rate ≈ 80%
Если кешируем топ-50% ключей → hit rate ≈ 95%
```

Дальше закон убывающей отдачи: 80% памяти даст только +5% hit rate.

---

## 3. Database connection pool sizing

**Задача:** Backend делает в среднем 5 DB запросов на request. Request rate = 200 rps. Средняя latency DB query = 10ms. Сколько нужно connections в pool?

**Теория:** Little's Law + вероятностные пики.

$$L = \lambda \cdot W$$

- \lambda = 200 \times 5 = 1000 queries/sec
- W = 0.01 sec
- L (среднее) = 10 connections

**Но:** нужен буфер под p95/p99 пики → реально 20-30 connections.

**Если pool = 10:** при малейшем всплеске будет connection exhaustion.

---

## 4. Fraud detection: сколько ложных срабатываний?

**Задача:** Система детектит мошенничество с точностью 99% (1% false positive). В день 1 миллион транзакций, из них 0.1% реально фродовые.

**Теория:** Bayes' theorem, precision/recall.

```
Истинно фродовых: 1,000,000 \times 0.001 = 1,000
Ложных срабатываний: 999,000 \times 0.01 = 9,990
```

**Вывод:** На каждую реальную атаку — 10 ложных алертов! Нужно улучшать precision.

Как считать по формуле байеса:

$$P(A|B) = \frac{P(B|A)P(A)}{P(B)}$$

| Параметр | Значение |
|---|---|
| Чувствительность *(P(алерт \| фрод))* | $P(A \mid F) = 0,99$ |
| False positive *(P(алерт \| честная))* | $P(A \mid \neg F) = 0,01$ |
| Доля фрода *(prior)* | $P(F) = 0,001$ |

**Вопрос:** система подняла алерт — какова вероятность реального фрода?

$$P(F \mid A) = ?$$

---

Формула Байеса

$$P(F \mid A) = \frac{P(A \mid F) \cdot P(F)}{P(A)}$$

где полная вероятность алерта:

$$P(A) = P(A \mid F) \cdot P(F) \;+\; P(A \mid \neg F) \cdot P(\neg F)$$

---

Подстановка

$$P(A) = 0,99 \times 0,001 + 0,01 \times 0,999 = 0,00099 + 0,00999 = 0,01098$$

$$\boxed{P(F \mid A) = \frac{0,99 \times 0,001}{0,01098} = \frac{0,00099}{0,01098} \approx 9\%}$$

---

Частотная интерпретация (1 млн транзакций)

$$\underbrace{1\,000\,000}_{\text{всего}} \times 0{,}001 = \underbrace{1\,000}_{\text{реальный фрод}} \quad \text{и} \quad \underbrace{999\,000}_{\text{честные}}$$

$$\text{Алертов на фрод: } 1\,000 \times 0{,}99 = 990$$

$$\text{Ложных алертов: } 999\,000 \times 0{,}01 = 9\,990$$

$$P(F \mid A) = \frac{990}{990 + 9\,990} = \frac{990}{10\,980} \approx 9\%$$

---

Парадокс базовой частоты

> Из **10 980** алертов в день лишь **990** — реальный фрод.
> То есть **~9 из 10 заблокированных транзакций** принадлежат честным клиентам.

Даже при точности $99\%$ — низкая база $P(F) = 0,1\%$ «захлёстывает» систему ложными срабатываниями.

$$\text{Precision} = \frac{TP}{TP + FP} = \frac{990}{10\,980} \approx 9\%$$

**Вывод:** чем реже событие, тем важнее не только *sensitivity*, но и *specificity*.

---

## 5. Retry policy и exponential backoff

**Задача:** API падает с вероятностью 5% на каждый request. Если делать retry, какова вероятность успеха?

**Теория:** Геометрическое распределение.

```
P(успех с 1 попытки) = 0.95
P(успех с 2 попыток) = 1 - (0.05)^2 = 0.9975
P(успех с 3 попыток) = 1 - (0.05)^3 = 0.999875
```

**Практический вывод:**
- 1 retry → с 95% до 99.75%
- 2 retries → 99.9875%
- Но нужен exponential backoff, иначе thunder herd

---

## 6. Load balancer: probability of hot shard

**Задача:** У вас 10 backend instance, приходит burst из 100 requests за 1ms (быстрее, чем они успевают разобраться). При random routing какова вероятность, что какой-то instance получит ≥15 requests?

**Теория:** Binomial distribution → Poisson approximation.

$$\lambda = \frac{100}{10} = 10, \quad P(X \ge 15) \approx e^{-10} \sum_{k=15}^{100} \frac{10^k}{k!}$$

Или через нормальную аппроксимацию: \mu = 10, \sigma = \sqrt{10} \approx 3.16.

$$P(X \ge 15) = P(Z \ge \frac{15-10}{3.16}) \approx P(Z \ge 1.58) \approx 0.057$$

**Вывод:** ≈5-6% вероятность, что instance получит перегрузку → нужен consistent hashing или least-connections routing.

---

## 7. Distributed consensus: вероятность split brain

**Задача:** Raft кластер из 5 нод. Вероятность network partition между любыми двумя нодами = 1%. Какова вероятность split brain (кластер разбился на 2 группы по 2 и 3 ноды)?

**Теория:** Комбинаторика + теория графов.

Упрощённая оценка: если 2 ноды изолированы от 3 остальных, нужно ≥2 рёбер оборваться.

**Практический вывод:** При low partition rate (1%) split brain очень редок, но при network flapping (10-20%) становится реальной проблемой → нужен quorum monitoring.

---

## 8. Rate limiting: сколько legitimate users заблокируем?

**Задача:** Rate limit = 100 rps на IP. Нормальный пользователь делает в среднем 2 rps с дисперсией (bursts). 1% пользователей — боты (200+ rps). Сколько false positives?

**Теория:** Распределение burst'ов + threshold analysis.

Если трафик пользователя ~ Poisson(2), вероятность burst ≥100 за секунду ничтожна.

Но если есть legitimate use case (например, batch upload), нужен token bucket вместо fixed window.

**Практический вывод:** Fixed rate limit блокирует легитимных пользователей → нужен leaky bucket / sliding window.

---

## 9. Backup retention: вероятность потери данных

**Задача:** Daily backup с вероятностью сбоя 0.1%. Храним 30 последних бэкапов. Какова вероятность, что ВСЕ 30 бэкапов битые?

**Теория:** Независимые события.

$$P(\text{все битые}) = (0.001)^{30} = 10^{-90}$$

Практически невозможно. Но если сбои коррелированы (например, bug в backup script), вероятность резко растёт.

**Практический вывод:** 
- Независимые сбои → можно хранить меньше копий
- Коррелированные сбои (software bug) → нужна диверсификация (разные tools, offsite)

---

## 10. Password brute force: когда блокировать?

**Задача:** Пароль из 6 цифр (10⁶ вариантов). Атакующий пробует 1000 паролей/сек. После скольких попыток блокировать аккаунт?

**Теория:** 
- Вероятность угадать за N попыток: $P = \frac{N}{10^6}$
- После 100 попыток: $P = 0,01\%$
- После 10 000 попыток: $P = 1\%$

**Практический вывод:**
```
Block after 3-5 attempts   → false positives (пользователь забыл пароль)
Block after 100 attempts   → reasonable (0.01% шанс угадать)
Block after 10,000 attempts → слишком поздно
```

Нужен CAPTCHA после 3-5 попыток, hard block после 10-20.

---


## 11 Расчёт количества instance

Это классический вопрос из теории массового обслуживания (ТМО), и важно показать, что ты понимаешь разницу между "в лоб" расчётом и корректной моделью.

Расчёт lower bound

**Дано:**
* входной поток: \lambda = 1000 rps
* производительность одного instance: \mu = 100 rps

**Минимально необходимое число instance:**

$$c_{min} = \frac{\lambda}{\mu} = \frac{1000}{100} = 10$$

⚠️ **Это абсолютный минимум, при котором:**
* система работает на 100% загрузке
* любая флуктуация \Rightarrow бесконечные очереди
* latency \to \infty

Как расчитать "c запасом"
**Модель:**
* вход: пуассоновский поток \to **M**
* время обслуживания: экспоненциальное \to **M**
* `c` параллельных серверов \to **M/M/c**

**Параметры:**
* \lambda = 1000 rps
* \mu = 100 rps
* c = ?
* коэффициент загрузки:

$$\rho = \frac{\lambda}{c \mu}$$

**Условие устойчивости:**

$$\rho < 1$$

В реальных системах обычно целятся в:
* **\rho \approx 0.6–0.7** — низкие latency
* **\rho \approx 0.7–0.8** — компромисс
* **\rho > 0.8** — очереди растут очень быстро

Возьмём \rho = 0.7:

$$c = \frac{\lambda}{\rho \mu} = \frac{1000}{0.7 \cdot 100} \approx 14.3$$

Итог - *"Минимально нужно **10 instance**, но это система с загрузкой 100%, очередь будет расти бесконечно. Корректно моделировать это как **M/M/c**. Обычно целимся в загрузку **60–70%**, тогда потребуется около **15 instance**. Более точно число выбирается через **Erlang C** под SLA по latency."*
