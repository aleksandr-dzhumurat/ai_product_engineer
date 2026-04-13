# Bootstrap

Bootstrap

Bootstrap — это метод оценки статистик выборки путём повторной выборки с возвращением из имеющихся данных.

Есть выборка из N элементов. Сделай 1000 раз: возьми N элементов случайно с возвращением → получи 1000 "псевдовыборок" → вычисли нужную статистику на каждой → посмотри на распределение.

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

- Доверительный интервал для любой метрики (accuracy, F1, AUC) — без предположений о распределении
- Сравнение моделей: у модели A accuracy 0.82, у B — 0.84. Значимо ли это? Bootstrap скажет.
- Оценка дисперсии любой статистики

---

Bootstrap в ML — Bagging

Bagging (Bootstrap Aggregating) — обучаем несколько моделей на разных bootstrap-выборках, усредняем предсказания.

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

Random Forest = Bagging + случайные признаки — классический пример.

### Зачем это снижает дисперсию:
- Разные выборки → модели делают разные ошибки
- Усреднение ошибок, которые не коррелированы → дисперсия падает в K раз

---

Out-of-Bag (OOB) оценка

При выборке с возвращением ~36.8% объектов не попадают в выборку. Их можно использовать как валидационную выборку бесплатно — это OOB score в Random Forest.

## Why ~36.8% of the Dataset Ends Up OOB

### Step 1: one specific object

Each bootstrap sample draws $n$ objects with replacement from a dataset of size $n$.
The probability that one specific object $x_i$ is never picked across all $n$ draws:

$$P_{\text{OOB}} = \left(1 - \frac{1}{n}\right)^n \xrightarrow{n \to \infty} e^{-1} \approx 0.368$$

This is a statement about a single object — not the whole dataset yet.

---

### Step 2: extending to all $n$ objects

Every object has the same $P_{\text{OOB}}$. By linearity of expectation, the expected number of OOB objects across the entire dataset:

$$E[\text{OOB count}] = \sum_{i=1}^{n} P_{\text{OOB}}(x_i) = n \cdot \left(1 - \frac{1}{n}\right)^n \approx \frac{n}{e}$$

---

### Step 3: the expected fraction

Dividing by $n$, because we want the fraction (proportion), not the count.

$$\frac{E[\text{OOB count}]}{n} = \left(1 - \frac{1}{n}\right)^n \approx 0.368$$

Since every object has the same $\approx 0.368$ probability of being left out, the expected fraction of the dataset that ends up OOB is also $\approx 0.368$.

The logic is the same as: if every coin has a 50% chance of heads, then on average 50% of $n$ coins will land heads — regardless of $n$.

# Практические задачи с теорией вероятностей


Будемм применять теорию к задачам по system design. Во всех этих задачах теория вероятностей помогает:

1. Оценить риски (SLA, fraud, data loss)
2. Подобрать пороги (rate limiting, connection pools)
3. Спроектировать retry/fallback (exponential backoff)
4. Понять trade-offs (false positives vs false negatives)
5. Capacity planning (M/M/c, Little's Law)

Без этого ты либо over-provision (тратишь деньги), либо under-provision (теряешь availability).

## 1. SLA и availability в distributed systems

Задача: У вас микросервисная архитектура с 5 сервисами в цепочке. Каждый сервис имеет uptime 99.9%. Какой будет итоговый SLA?

Теория: Независимые вероятности, умножение вероятностей.

$$P(\text{система работает}) = (0.999)^5 = 0.995 \approx 99.5\%$$

### Практический вывод:
- Цепочка из N сервисов → SLA деградирует
- Для 99.99% итогового SLA каждый сервис должен иметь 99.998%
- Нужны retries, circuit breakers, fallbacks

---

## 2. Cache hit rate и память

Задача: У вас есть 10GB RAM под cache, запросы идут по Zipf distribution (80% запросов к 20% ключей). Сколько данных кешировать?

### Теория:
- Распределение частот запросов
- Оптимизация hit rate vs память

### Практический подход:
```
Если кешируем топ-20% ключей → hit rate ≈ 80%
Если кешируем топ-50% ключей → hit rate ≈ 95%
```

Дальше закон убывающей отдачи: 80% памяти даст только +5% hit rate.

---

## 3. Database connection pool sizing

Задача: Backend делает в среднем 5 DB запросов на request. Request rate = 200 rps. Средняя latency DB query = 10ms. Сколько нужно connections в pool?

Теория: Little's Law + вероятностные пики.

$$L = \lambda \cdot W$$

- \lambda = 200 \times 5 = 1000 queries/sec
- W = 0.01 sec
- L (среднее) = 10 connections

Но: нужен буфер под p95/p99 пики → реально 20-30 connections.

Если pool = 10: при малейшем всплеске будет connection exhaustion.

---

## 4. Fraud detection: сколько ложных срабатываний?

Задача: Система детектит мошенничество с точностью 99% (1% false positive). В день 1 миллион транзакций, из них 0.1% реально фродовые.

Теория: Bayes' theorem, precision/recall.

```
Истинно фродовых: 1,000,000 \times 0.001 = 1,000
Ложных срабатываний: 999,000 \times 0.01 = 9,990
```

Вывод: На каждую реальную атаку — 10 ложных алертов! Нужно улучшать precision.

Как считать по формуле байеса:

$$P(A|B) = \frac{P(B|A)P(A)}{P(B)}$$

| Параметр | Значение |
|---|---|
| Чувствительность *(P(алерт \| фрод))* | $P(A \mid F) = 0,99$ |
| False positive *(P(алерт \| честная))* | $P(A \mid \neg F) = 0,01$ |
| Доля фрода *(prior)* | $P(F) = 0,001$ |

Вопрос: система подняла алерт — какова вероятность реального фрода?

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

> Из 10 980 алертов в день лишь 990 — реальный фрод.
> То есть ~9 из 10 заблокированных транзакций принадлежат честным клиентам.

Даже при точности $99\%$ — низкая база $P(F) = 0,1\%$ «захлёстывает» систему ложными срабатываниями.

$$\text{Precision} = \frac{TP}{TP + FP} = \frac{990}{10\,980} \approx 9\%$$

Вывод: чем реже событие, тем важнее не только *sensitivity*, но и *specificity*.

---

## 5. Retry policy и exponential backoff

Задача: API падает с вероятностью 5% на каждый request. Если делать retry, какова вероятность успеха?

Теория: Геометрическое распределение.

```
P(успех с 1 попытки) = 0.95
P(успех с 2 попыток) = 1 - (0.05)^2 = 0.9975
P(успех с 3 попыток) = 1 - (0.05)^3 = 0.999875
```

### Практический вывод:
- 1 retry → с 95% до 99.75%
- 2 retries → 99.9875%
- Но нужен exponential backoff, иначе thunder herd

---

## 6. Load balancer: probability of hot shard

Задача: У вас 10 backend instance, приходит burst из 100 requests за 1ms (быстрее, чем они успевают разобраться). При random routing какова вероятность, что какой-то instance получит ≥15 requests?

Теория: Binomial distribution → Poisson approximation.

$$\lambda = \frac{100}{10} = 10, \quad P(X \ge 15) \approx e^{-10} \sum_{k=15}^{100} \frac{10^k}{k!}$$

Или через нормальную аппроксимацию: \mu = 10, \sigma = \sqrt{10} \approx 3.16.

$$P(X \ge 15) = P(Z \ge \frac{15-10}{3.16}) \approx P(Z \ge 1.58) \approx 0.057$$

Вывод: ≈5-6% вероятность, что instance получит перегрузку → нужен consistent hashing или least-connections routing.

---

## 7. Distributed consensus: вероятность split brain

Задача: Raft кластер из 5 нод. Вероятность network partition между любыми двумя нодами = 1%. Какова вероятность split brain (кластер разбился на 2 группы по 2 и 3 ноды)?

Теория: Комбинаторика + теория графов.

Упрощённая оценка: если 2 ноды изолированы от 3 остальных, нужно ≥2 рёбер оборваться.

Практический вывод: При low partition rate (1%) split brain очень редок, но при network flapping (10-20%) становится реальной проблемой → нужен quorum monitoring.

---

## 8. Rate limiting: сколько legitimate users заблокируем?

Задача: Rate limit = 100 rps на IP. Нормальный пользователь делает в среднем 2 rps с дисперсией (bursts). 1% пользователей — боты (200+ rps). Сколько false positives?

Теория: Распределение burst'ов + threshold analysis.

Если трафик пользователя ~ Poisson(2), вероятность burst ≥100 за секунду ничтожна.

Но если есть legitimate use case (например, batch upload), нужен token bucket вместо fixed window.

Практический вывод: Fixed rate limit блокирует легитимных пользователей → нужен leaky bucket / sliding window.

---

## 9. Backup retention: вероятность потери данных

Задача: Daily backup с вероятностью сбоя 0.1%. Храним 30 последних бэкапов. Какова вероятность, что ВСЕ 30 бэкапов битые?

Теория: Независимые события.

$$P(\text{все битые}) = (0.001)^{30} = 10^{-90}$$

Практически невозможно. Но если сбои коррелированы (например, bug в backup script), вероятность резко растёт.

### Практический вывод:
- Независимые сбои → можно хранить меньше копий
- Коррелированные сбои (software bug) → нужна диверсификация (разные tools, offsite)

---

## 10. Password brute force: когда блокировать?

Задача: Пароль из 6 цифр (10⁶ вариантов). Атакующий пробует 1000 паролей/сек. После скольких попыток блокировать аккаунт?

### Теория:
- Вероятность угадать за N попыток: $P = \frac{N}{10^6}$
- После 100 попыток: $P = 0,01\%$
- После 10 000 попыток: $P = 1\%$

### Практический вывод:
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

### Дано:
* входной поток: \lambda = 1000 rps
* производительность одного instance: \mu = 100 rps

### Минимально необходимое число instance:

$$c_{min} = \frac{\lambda}{\mu} = \frac{1000}{100} = 10$$

⚠️ Это абсолютный минимум, при котором:
* система работает на 100% загрузке
* любая флуктуация \Rightarrow бесконечные очереди
* latency \to \infty

Как расчитать "c запасом"
### Модель:
* вход: пуассоновский поток \to M
* время обслуживания: экспоненциальное \to M
* `c` параллельных серверов \to M/M/c

### Параметры:
* \lambda = 1000 rps
* \mu = 100 rps
* c = ?
* коэффициент загрузки:

$$\rho = \frac{\lambda}{c \mu}$$

### Условие устойчивости:

$$\rho < 1$$

В реальных системах обычно целятся в:
* \rho \approx 0.6–0.7 — низкие latency
* \rho \approx 0.7–0.8 — компромисс
* \rho > 0.8 — очереди растут очень быстро

Возьмём \rho = 0.7:

$$c = \frac{\lambda}{\rho \mu} = \frac{1000}{0.7 \cdot 100} \approx 14.3$$

Итог - *"Минимально нужно 10 instance, но это система с загрузкой 100%, очередь будет расти бесконечно. Корректно моделировать это как M/M/c. Обычно целимся в загрузку 60–70%, тогда потребуется около 15 instance. Более точно число выбирается через Erlang C под SLA по latency."*

# Statistics

* [Statistics 101](https://www.linkedin.com/feed/update/activity:7081002429294469120/#share-modal)
* [Central limit theorem](https://www.linkedin.com/feed/update/activity:7089091830310477824?trk=feed_main-feed-card_comment-cta) + [in russian](http://www.mathtask.ru/0034-the-laws-of-large-numbers-and-limit-theorems.php)
* [Maximum Likelihood habr](https://habr.com/ru/companies/otus/articles/585610/)
* [Conversion rates as probabilities](https://www.linkedin.com/feed/update/activity:7073658182073499648)
* [MarkovChain](https://pub.towardsai.net/a-beginners-guide-to-markov-chains-conditional-probability-and-independence-b35887a9032)
* [Classification loss: LogLoss, Hinge etc](https://medium.com/@anushruthikae/understanding-classification-loss-functions-7cc13fd6ac97)
* [introduction-to-reinforcement-learning-markov-decision-process](https://towardsdatascience.com/introduction-to-reinforcement-learning-markov-decision-process-44c533ebf8da)
* [Deploying and pressure testing a Markov Chains model](https://medium.com/tech-getyourguide/deploying-and-pressure-testing-a-markov-chains-model-7e18916f70f8)
* [All You Need Is Logs: Improving Code Completion by Learning from Anonymous IDE Usage](https://arxiv.org/abs/2205.10692)
* [Statistical Learning Theory: Models, Concepts, and Results](https://arxiv.org/abs/0810.4752)
* [Stat tests cheatsheet](https://www.linkedin.com/posts/michael-maltsev_found-a-super-useful-cheat-sheet-in-the-archives-activity-7313943754413891584-yGYU)


# AB-tests

* [colab](https://colab.research.google.com/drive/1uFVf6h98Z9IsrHxkefHBV4-vEt_KjVZG?usp=sharing)
* [Курс по AB-тестам](https://www.youtube.com/watch?v=D81kNptqPiw&list=PLCf-cQCe1FRx6vgs5NHWKzOL5RSyWiiuW)
* [[Youtube] Прикладная статистика](https://www.youtube.com/@user-bg8cd4fn7d))
* [MyTracker AB-tests](https://tracker.my.com/blog/204/5-lajfhakov-dlya-uskoreniya-a-b-testirovaniya-ot-analitikov-mytracker)
* [[Criteo] why-your-ab-test-needs-confidence-intervals](https://medium.com/criteo-engineering/why-your-ab-test-needs-confidence-intervals-bec9fe18db41)
* [[Criteo] ab-test-decisions-reducing-type-1-errors-and-using-elasticity](https://medium.com/criteo-engineering/ab-test-decisions-reducing-type-1-errors-and-using-elasticity-716bc286f24a)
* [[Criteo] how-to-compare-two-treatments](https://medium.com/criteo-engineering/how-to-compare-two-treatments-ade0753fe39f)
* [commonly-used-statistical-tests-in-data-science](https://nathanrosidi.medium.com/commonly-used-statistical-tests-in-data-science-93787568eb36)
* [[Exprf] Когда останавливать тест: MDE](https://medium.com/statistics-experiments/%D0%BA%D0%BE%D0%B3%D0%B4%D0%B0-%D0%BE%D1%81%D1%82%D0%B0%D0%BD%D0%B0%D0%B2%D0%BB%D0%B8%D0%B2%D0%B0%D1%82%D1%8C-a-b-%D1%82%D0%B5%D1%81%D1%82-%D1%87%D0%B0%D1%81%D1%82%D1%8C-1-mde-7d39b668b488)
* [[exprf] Ускорение сходимости: CUPED](https://medium.com/statistics-experiments/cuped-%D0%B8%D0%BB%D0%B8-%D1%83%D0%B2%D0%B5%D0%BB%D0%B8%D1%87%D0%B5%D0%BD%D0%B8%D0%B5-%D1%87%D1%83%D0%B2%D1%81%D1%82%D0%B2%D0%B8%D1%82%D0%B5%D0%BB%D1%8C%D0%BD%D0%BE%D1%81%D1%82%D0%B8-%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8-de7183fc964c)
* [Ускорение AB-тестов](https://youtu.be/Z9gVndEr_70?si=bDmoVg7yORHldV1E)
* [rank transformation in booking](https://booking.ai/increasing-sensitivity-of-experiments-with-the-rank-transformation-draft-c01aff70b255)
* [metric sensitivity](https://medium.com/data-science-at-microsoft/increase-your-chances-to-capture-feature-impact-with-a-b-tests-8c2f16839b8b)
* [choosing-the-right-statistical-tests-for-effective-a-b-testing](https://medium.com/@thauri.dattadeen/choosing-the-right-statistical-tests-for-effective-a-b-testing-dd3dd6e3d5bc)
* [[SbermarketTech] AB intro](https://habr.com/ru/companies/sbermarket/articles/774608/)
* [[SbermarketTech] AB: linearization](https://habr.com/ru/companies/sbermarket/articles/768826)
* [[Ozon] AB-tests meetup](https://www.youtube.com/watch?v=ly-pqx1P34k) + [Slides](https://speakerdeck.com/ozontech/b-tiestirovaniia-iz-shiesti?slide=1)
* [Python implementation](https://github.com/shadowymoses/habr-articles/blob/main/six_reasons_why_your_ab_tests_are_invalid.ipynb)
* [Стрим с объяснениями](https://www.youtube.com/watch?v=dPqpBAXVw7g)
* [Habr article](https://habr.com/ru/companies/ozontech/articles/712306/)
* [Как устроено A/B-тестирование в Авито](https://habr.com/ru/companies/avito/articles/454164/)
* [Tinkoff: AB tests meetup](https://meetup.tinkoff.ru/event/tinkoff-product-analytics-meetup-ab-testy-online)
* [power-and-sample-size-calculations](https://www.statsmodels.org/dev/stats.html#power-and-sample-size-calculations)
* [vk-practitioners-guide-to-statistical-tests](https://vkteam.medium.com/practitioners-guide-to-statistical-tests-ed2d580ef04f)
* [mde](https://medium.com/statistics-experiments/%D0%BA%D0%BE%D0%B3%D0%B4%D0%B0-%D0%BE%D1%81%D1%82%D0%B0%D0%BD%D0%B0%D0%B2%D0%BB%D0%B8%D0%B2%D0%B0%D1%82%D1%8C-a-b-%D1%82%D0%B5%D1%81%D1%82-%D1%87%D0%B0%D1%81%D1%82%D1%8C-1-mde-7d39b668b488)
* [AB number of observations](https://www.linkedin.com/feed/update/activity:7206265862725517312)
* [AB platform](https://habr.com/ru/companies/sbermarket/articles/816841/)
* [10 Advanced ML and Statistics concepts](https://www.linkedin.com/feed/update/activity:7029234053052526592/)
* [standard deviation vs standard error](https://www.linkedin.com/feed/update/urn:li:activity:7322569929361108992)
* [Linkedin statistics](https://www.linkedin.com/feed/update/urn:li:activity:7092496133322596352/)
* [Bayesian A/B testing](https://www.linkedin.com/feed/update/activity:7206265862725517312/)
* [the-trade-offs-of-large-scale-machine-learning](https://medium.com/criteo-engineering/the-trade-offs-of-large-scale-machine-learning-71ad0cf7469f)
* [probability theory book](https://www.linkedin.com/posts/michael-erlihson-phd-8208616_probability-theory-a-short-intro-activity-7417533246982414336-cSdS)
* [economentrics book](https://www.linkedin.com/posts/michael-erlihson-phd-8208616_basic-econometrics-ugcPost-7434732011476533249-tj7W)
* [GetYourGuide: 15 data science principles we live by](https://medium.com/tech-getyourguide/15-data-science-principles-we-live-by-d5b9eca92fd2)
* [H3: Uber’s Hexagonal Hierarchical Spatial Index](https://www.uber.com/en-CZ/blog/h3/)
* [your-ml-setup-is-not-unique-you-dont-need-more-data-scientists](https://blog.metarank.ai/your-ml-setup-is-not-unique-you-dont-need-more-data-scientists-f42f33c0379a)
* [bloom Filter](https://towardsdatascience.com/how-to-store-and-query-100-million-items-using-just-77mb-with-python-bloom-filters-6b3e8549f032)
* [Bloom filter](https://medium.com/javarevisited/interview-how-to-check-whether-a-username-exists-among-one-billion-users-ffa0d0522998)
* [Lamoda: ML-прайсинг , персонализация](https://habr.com/ru/companies/lamoda/articles/849398/)
* [Увеличиваем выручку с помощью математики: как учитывать бизнес-контекст в оптимизационных задачах](https://habr.com/ru/companies/ecom_tech/articles/851296/)
* [[Youtube] Разбор задачек по аналитике](https://www.youtube.com/live/0Kngp3PCp8g?si=ntjnqYHi7wirPUih)

* [Практический гайд от VK](https://vkteam.medium.com/practitioners-guide-to-statistical-tests-ed2d580ef04f) + [python demo](https://github.com/marnikitta/stattests)
* [Как тестировать группы разного размера +](https://medium.com/statistics-experiments/%D0%B4%D0%B8%D1%81%D0%B1%D0%B0%D0%BB%D0%B0%D0%BD%D1%81-%D0%B2-a-b-%D1%82%D0%B5%D1%81%D1%82%D0%B0%D1%85-%D0%B5%D1%81%D1%82%D1%8C-%D0%BB%D0%B8-%D1%80%D0%B0%D0%B7%D0%BD%D0%B8%D1%86%D0%B0-%D0%BC%D0%B5%D0%B6%D0%B4%D1%83-99-1-%D0%B8-50-50-%D0%B2-%D1%8D%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D0%B0%D1%85-11c8f4fe7eb4)
* [Б лог Expf](https://medium.com/statistics-experiments)
* [Методы анализа AB-тестов: выбираем правильный](https://vk.com/@we_use_django-metody-analiza-ab-testov-kak-vybrat-pravilnyi-metod-dlya-kaz)
* [Dealing With Ratio Metrics in A/B Testing at the Presence of Intra-User Correlation and Segments](https://arxiv.org/pdf/1911.03553.pdf)
* [ratio metrics](https://www.linkedin.com/posts/%D0%B0%D1%80%D1%81%D0%B5%D0%BD%D0%B8%D0%B9-%D1%84%D0%B8%D0%BB%D0%B8%D0%BD-798a12259_ab-activity-7272584955073904640-A5JL)
* [Switchback](https://habr.com/ru/company/citymobil/blog/560426/)
* [Switchback in DoorDash](https://medium.com/@DoorDash/switchback-tests-and-randomized-experimentation-under-network-effects-at-doordash-f1d938ab7c2a)
* [bootstrapping-confidence-intervals-the-basics-](https://towardsdatascience.com/bootstrapping-confidence-intervals-the-basics-b4f28156a8da)
* [Dif-n-Diff explained](https://medium.com/bukalapak-data/difference-in-differences-8c925e691fff) + MixTape [Diff-n-Diff](https://mixtape.scunning.com/09-difference_in_differences) + [TheEffectBook](https://www.theeffectbook.net/ch-DifferenceinDifference.html) + [Diff-n-diff linkedin](https://www.linkedin.com/feed/update/activity:7254748476205400064)
* [Syntetic Control Group](https://medium.datadriveninvestor.com/beyond-ab-testing-experimentation-without-a-b-testing-switchbacks-and-synthetic-control-group-b59268ac4c86)
* [validating-the-causal-impact-of-the-synthetic-control-method](https://www.notion.so/Self-intro-8f3dcb4f239a48ea80f98b7aec58b0f3?pvs=21)
* [Syntetic control](https://youtu.be/j5DoJV5S2Ao?si=1LE0kmdyEvFLaZPU)
* [AB-tests terminology](https://www.coveo.com/blog/ab-test-terminology/)
* [experimenting-with-machine-learning-to-target-in-app-messaging](https://engineering.atspotify.com/2023/06/experimenting-with-machine-learning-to-target-in-app-messaging/?utm_medium=social&utm_source=linkedIn&utm_campaign=exp%20in%20app%20msg&utm_content=evergreen)
* [Методы оценки размера выборки в AB-тестах](https://www.youtube.com/watch?v=lJY6eMh10iE)
* [MDE (Youtube search)](https://www.youtube.com/results?search_query=AB+%D1%82%D0%B5%D1%81%D1%82%D1%8B+MDE)
* [A First Course in Causal Inference](https://arxiv.org/abs/2305.18793)
* [Github repositories for casuality](https://www.linkedin.com/feed/update/activity:7089884530261540864?trk=feed_main-feed-card_comment-cta)
* [Demystifying casuality](https://www.linkedin.com/feed/update/activity:7081259950890450944) (experiments in production)
* [alexdeng.github.io/causal/](https://alexdeng.github.io/causal/)
* [Spotify Casual Inference](https://engineering.atspotify.com/2024/03/risk-aware-product-decisions-in-a-b-tests-with-multiple-metrics/)
* [Casual Inference](https://towardsdatascience.com/using-causal-ml-instead-of-a-b-testing-eeb1067d7fc0)
* [Casual inference](https://medium.com/towards-data-science/how-to-use-causal-inference-when-a-b-testing-is-not-possible-c87c1252724a)
* [Анализ рекомендаций без АБ-тестов](https://www.youtube.com/watch?v=MMAGtkb7ZHk) (casual impact)
* [Uber casual ML](https://github.com/uber/causalml) tool
* [Google casual impact library](https://pub.towardsai.net/causal-inference-python-implementation-fa94c76cd5af)
* [Python casuality Handbook](https://matheusfacure.github.io/python-causality-handbook/landing-page.html)
* [Regression for casual impact](https://medium.com/towards-data-science/linear-regressions-for-causal-conclusions-34c6317c5a11)
* [Python casual inference](https://medium.com/@arun.subram456/causal-inference-regression-discontinuity-design-338f0f0b5f31)
* [Casual inference](https://towardsdatascience.com/causal-machine-learning-for-customer-retention-a-practical-guide-with-python-6bd959b25741)
* [using-causal-inference-to-measure-business-impact-after-program-launch](https://medium.com/data-science-at-microsoft/using-causal-inference-to-measure-business-impact-after-program-launch-d8361bfeee71)
* [Pinterest: Web performance degradation casual impact](https://medium.com/pinterest-engineering/web-performance-regression-detection-part-2-of-3-9e0b9d35a11f)
* [casual impact](https://www.linkedin.com/feed/update/activity:7263157169326379008)
* [using-causal-ml-instead-of-a-b-testing](https://towardsdatascience.com/using-causal-ml-instead-of-a-b-testing-eeb1067d7fc0)
* [causal-machine-learning](https://www.notion.so/English-phone-Interview-preparing-1817678191da47329a9871c901bfb347?pvs=21)
* [practitioners-guide-to-statistical-tests](https://medium.com/@vktech/practitioners-guide-to-statistical-tests-ed2d580ef04f)
* [how-to-set-the-minimum-detectable-effect-in-ab-tests](https://towardsdatascience.com/how-to-set-the-minimum-detectable-effect-in-ab-tests-fe07f8002d6d)
* [AB sample size calculations](https://www.linkedin.com/feed/update/activity:7105969148236611584)
* [AB sample size](https://www.linkedin.com/feed/update/activity:7183040757958615040)
* [Netflix Sequential AB testing](https://netflixtechblog.com/sequential-a-b-testing-keeps-the-world-streaming-netflix-part-1-continuous-data-cba6c7ed49df)
* [Netflix sequential testing](https://dvc.ai/blog/dvc-ray)
* [Uber sequential AB-testing](https://www.youtube.com/watch?v=4rWOx5fOJbg)
* [a-guide-on-estimating-long-term-effects-in-a-b-tests](https://medium.com/towards-data-science/a-guide-on-estimating-long-term-effects-in-a-b-tests-9a3790501047)
* [counterfactual testing](https://medium.com/data-shopify/how-to-use-quasi-experiments-and-counterfactuals-to-build-great-products-487193794da)
* [AB-tests without AB tests](https://koch-kir.medium.com/causal-inference-from-observational-data-%D0%B8%D0%BB%D0%B8-%D0%BA%D0%B0%D0%BA-%D0%BF%D1%80%D0%BE%D0%B2%D0%B5%D1%81%D1%82%D0%B8-%D0%B0-%D0%B2-%D1%82%D0%B5%D1%81%D1%82-%D0%B1%D0%B5%D0%B7-%D0%B0-%D0%B2-%D1%82%D0%B5%D1%81%D1%82%D0%B0-afb84f2579f2)
* [Flo experimental](https://medium.com/flo-health/how-flo-conducts-experiments-5ee35fc3327f) + [second part](https://medium.com/flo-health/experimentation-2-0-navigating-the-upgrades-at-flo-health-376c609f6a85) + [hypotesys testing framework](https://gopractice.ru/data/how_to_increase_the_number_of_successful_experiments/)
* [Bayes AB-tests](https://towardsdatascience.com/why-you-should-try-the-bayesian-approach-of-a-b-testing-38b8079ea33a)
* [Bayes AB-tests Python](https://towardsdatascience.com/bayesian-a-b-testing-with-python-the-easy-guide-d638f89e0b8a)
* [Youtube: Bayesian AB-tests](https://youtu.be/1fnXvWwtFss?si=lWgNcF58vCRNvJUC)
* [WISE: SkySkanner Bayes AB-tests system](https://medium.com/@SkyscannerEng/wise-skyscanners-bayesian-ab-experimentation-library-and-decision-engine-6841d1643482)
* [Peeking problem](https://gopractice.ru/data/how-not-to-analyze-abtests/)
* [Peeking monitoring in Netflix](https://www.linkedin.com/feed/update/activity:7229381300728524800)
* [p-value dynamics](https://www.linkedin.com/pulse/%D0%B4%D0%B8%D0%BD%D0%B0%D0%BC%D0%B8%D0%BA%D0%B0-p-value-%D0%B2%D0%B0%D0%B6%D0%BD%D0%B5%D0%B5-%D0%B5%D0%B3%D0%BE-%D0%B8%D1%82%D0%BE%D0%B3%D0%BE%D0%B2%D0%BE%D0%B3%D0%BE-%D0%B7%D0%BD%D0%B0%D1%87%D0%B5%D0%BD%D0%B8%D1%8F-polina-egubova-jpelc?trk=feed_main-feed-card_feed-article-content)
* [Detect cannibalization with bootstrap](https://habr.com/ru/post/451488/)
* [Retail offline AB tests](https://habr.com/company/ods/blog/416101/)
* [Samokat offline ab-tests](https://habr.com/ru/companies/samokat_tech/articles/821777)
* [Retail offline at X5](https://habr.com/ru/company/X5RetailGroup/blog/466349/)
* [AOV in AB-tests: Statistical Significance for Non-Binomial Metrics – Revenue per User, AOV](http://blog.analytics-toolkit.com/2017/statistical-significance-non-binomial-metrics-revenue-time-site-pages-session-aov-rpu/)
* [Multiple experiments: theory and practise](https://habr.com/ru/company/yandex/blog/476826/)
* [AA tests guide](https://www.linkedin.com/feed/update/ugcPost:7188082288037953536)
* [Increase statistic power](https://towardsdatascience.com/5-ways-to-increase-statistical-power-377c00dd0214)
* [Booking Statistic Power](https://medium.com/booking-com-data-science/raising-the-bar-by-lowering-the-bound-3b12d3bd43a3)
* [SkyScanner blog peeking problem](https://medium.com/@SkyscannerEng/the-fourth-ghost-of-experimentation-peeking-b33890dcd3de)
* [Metric linearization](https://medium.com/@nikolai.neustroev/empowering-ratio-metrics-with-linearization-d694b6e56d7f)
* [Meta network effect](https://medium.com/@AnalyticsAtMeta/how-meta-tests-products-with-strong-network-effects-96003a056c2c)
* [AB materials compilation](https://www.linkedin.com/feed/update/activity:7183720380124020736?trk=feed_main-feed-card_comment-cta) (TO DO: analyze)
* [CUPED: Бабушкин](https://youtu.be/HpinAY5QfCo?si=Wm2chyM02jVmiht8)
* [AB tests speedup](https://www.linkedin.com/feed/update/activity:7210899458773995520)
* [Lamoda tech analytics party](https://youtu.be/esJyU03LwAs)
* [AB tests design](https://youtu.be/RVDlJOW4Vns)
* [forget-statistical-tests-a-b-testing-is-all-about-simulations](https://towardsdatascience.com/forget-statistical-tests-a-b-testing-is-all-about-simulations-33efa2241ae2)

# AB-platform

* [Как устроено AB-тестирование в Avito](https://habr.com/ru/companies/avito/articles/454164/)
* [AB-platform в Ozon](https://habr.com/ru/companies/ozontech/articles/689052/)
* [AB-platform HH](https://habr.com/ru/company/hh/blog/321386/)
* [Sbermarket AB platform](https://www.youtube.com/watch?v=YoTTuiVDeMo)
* [Experiments in mobile app](https://gopractice.ru/ab_testing_mobile_apps/)
* [AB-platform](https://www.youtube.com/watch?v=-xjd32x8QN4)
* [Teads AB-platform](https://medium.com/teads-engineering/production-a-b-test-analysis-framework-at-teads-19450f0c9bf)
* [AB tests digest](https://www.linkedin.com/feed/update/activity:7262358598905696256)