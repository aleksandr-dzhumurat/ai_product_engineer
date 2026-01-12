# Балансировка бизнес-целей в ML: User Clicks vs High-Commission Orders



[Проблема](https://www.linkedin.com/posts/hoang-van-hao_machinelearning-mlengineer-mlsystemdesign-activity-7399437308841750528-N8AI)

**Product:** максимизировать User Clicks  
**Sales:** максимизировать High-Commission Orders

**Типичное (неправильное) решение:** объединить цели в одну Loss Function

---

## ❌ Почему это плохо

Наивный подход

$$\text{Loss} = \alpha \cdot \text{LogLoss(Click)} + \beta \cdot \text{MSE(Profit)}$$

**Проблемы:**

1. Бизнес-логика зашита в веса модели
2. Изменение приоритетов = переобучение модели
3. Невозможно быстро реагировать на события (Black Friday)

**Пример:** VP Sales просит на 6 часов поднять маржу на 10%

Нужно:
- Переобучить модель с новыми α/β
- Ревалидировать метрики
- Canary deploy

⏱️ **3 дня** на **3-часовую** задачу

---

## ✅ Правильное решение: Decoupled Objective Protocol

Принцип разделения

**Relevance** (что хочет пользователь) — физика → модель  
**Priority** (что хочет компания) — бизнес → serving layer

### Архитектура

1. **Model A (Brain):** обучается только на P(Click) или P(Conversion)
   - Не знает ничего про деньги
   - Стабильная, не требует частого переобучения

2. **Signal (Context):** Commission_Rate из feature store

3. **Fusion:** объединение на этапе ранжирования

$$\text{Final\_Score} = w_1 \cdot \text{Model\_Prediction} + w_2 \cdot \text{Commission\_Normalized}$$

---

## Преимущества

| Задача | Наивный подход | Decoupled подход |
|--------|----------------|------------------|
| Изменить баланс | Переобучить модель (3 дня) | Изменить config (минуты) |
| A/B тест приоритетов | Несколько моделей | Один config параметр |
| Rollback | Откат модели | Откат config |


---

## Ключевой инсайт

> "Мы не используем ML для обучения trade-off.  
> Мы используем ML для обучения вероятностей,  
> а trade-off определяем динамически в runtime."

---

## Правило для продакшена

Не запекайте бизнес-логику в веса модели.

**Coupling = inflexibility**

Модель должна решать:
- ✅ Что пользователь хочет кликнуть
- ✅ Какова вероятность конверсии

Serving layer должен решать:
- ✅ Как балансировать цели компании
- ✅ Как реагировать на рыночные события

# Metrics

[Evaluating Recommendation Systems](https://tzin.bgu.ac.il/~shanigu/Publications/EvaluationMetrics.17.pdf)

Code

```python
def mean_precision_at_k(y_true, y_score, group, k=3):
    df = pd.DataFrame({'group_id': group, 'y_score': y_score, 'y_true': y_true})
    df['rank'] = df.groupby("group_id")["y_score"].rank(ascending=False)
    return df[df['rank'] <= k].groupby("group_id").y_true.sum().mean()


def mean_reciprocal_rank(y_true, y_score, group):
    df = pd.DataFrame({'group_id': group, 'y_score': y_score, 'y_true': y_true})
    df['rank'] = df.groupby("group_id")["y_score"].rank(ascending=False)
    return (1 / df.query("y_true==1")['rank']).mean()


def mean_rank(y_true, y_score, group):
    df = pd.DataFrame({'group_id': group, 'y_score': y_score, 'y_true': y_true})
    df['rank'] = df.groupby("group_id")["y_score"].rank(ascending=False)
    return (df.query("y_true==1")['rank']).mean()
```