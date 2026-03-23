# Visualizations

## Data Visualization Explained

Data visualization helps interpret complex data patterns. Below are some common types of charts and their best use cases.

---

## Bar Chart

- What is it? A bar chart represents categorical data with rectangular bars, where the height (or length) is proportional to the value.

- Use Cases:
  - Comparing different categories (e.g., sales by region, product popularity).
  - Displaying trends over time (if categories are time-based).

- Variations:
  - Stacked Bar Chart (Shows parts of a whole)
  - Grouped Bar Chart (Compares multiple datasets)

---

## Line Chart

- What is it? A line chart connects data points with lines, typically used for tracking changes over time.

- Use Cases:
  - Trends over time (e.g., stock prices, website traffic).
  - Continuous data comparisons (e.g., temperature changes).

- Variations:
  - Multi-line Chart (Compares multiple variables)
  - Area Chart (Like a line chart but filled below the line)

---

## Scatter Plot

- What is it? A scatter plot visualizes relationships between two numerical variables using dots.

- Use Cases:
  - Identifying correlations (e.g., height vs weight).
  - Outlier detection (e.g., fraudulent transactions).
  - Clustering patterns (e.g., customer segmentation).

---

## Heat Map

- What is it? A heat map represents values using colors, making it easy to spot patterns.

- Use Cases:
  - Correlation matrices (e.g., feature relationships in ML).
  - Website click tracking (e.g., user engagement).
  - Performance monitoring (e.g., CPU usage over time).

---

## Pie Chart & Donut Chart

- What are they?
  - Pie Chart shows proportions as slices of a circle.
  - Donut Chart is a modified pie chart with a hollow center, making it more readable.

- Use Cases:
  - Market share distribution.
  - Budget allocation percentages.
  - Simple proportions (avoid for too many categories).

---

## Tree Map (Rectangular)

- What is it? A tree map uses nested rectangles to represent hierarchical data, where area size reflects value.

- Use Cases:
  - File system storage analysis (e.g., folder sizes).
  - Market capitalization by company size.
  - Hierarchical aggregations (e.g., sales per region & sub-region).

---

## Word Cloud

- What is it? A word cloud displays words in different sizes based on frequency.

- Use Cases:
  - Text analysis (e.g., customer reviews, social media trends).
  - Identifying key topics in a document.
  - Keyword analysis for SEO.

---

## Summary

- Bar & Line Charts: Best for trends & comparisons.
- Scatter Plots: Best for correlation & clustering.
- Heat Maps: Best for intensity & patterns.
- Pie & Donut Charts: Best for proportions (but avoid overuse).
- Tree Maps: Best for hierarchical data representation.
- Word Clouds: Best for textual data insights.


# Experimental Design

На секции по АБ тестам есть контрольные вопросы, с помощью которых интервьюер может быстро определить твой практический опыт в АБ.

Правила: сначала пробуем ответить сами, после — можно смотреть вариант решения.

---

## Вопрос 1

**Какой статистический критерий ты бы применил для проверки на стат. значимости изменения среднего чека?**

**Ответ:**

Метрика среднего чека рассчитывается как общая выручка с продаж делённая на количество заказов.

Средний чек — это **метрика отношения (ratio-метрика)**, которая требует отдельного внимания, т. к. числитель и знаменатель коррелируют.

Поэтому если ты ответил **t-test** — его применять как раз нельзя. Сначала нужно использовать **delta-метод или линеаризацию**, после чего можно будет использовать t-test как к непрерывной метрике, либо **bootstrap**.

> На понимание и умение определять метрики отношения любят ловить на собесах.

---

## Вопрос 2

**Мы — Telegram. Хотим добавить новую фичу — видеозвонки. Как бы ты задизайнил такой эксперимент?**

**Ответ:**

Самое главное, что хотят услышать — это наличие **сетевого эффекта** и как мы можем с ним работать.

Часть пользователей, которым видеозвонки будут доступны, смогут звонить тем, у кого этой функции нет. И это явно найдёт отражение на метриках.

Умение определять наличие сетевого эффекта очень важно. В противном случае можно не только испортить результаты эксперимента, но и наделать делов в продукте.

---

## Вопрос 3

**Как бы ты подбирал MDE для эксперимента?**

**Ответ:**

Если твой ответ это:
- ❌ «Его должен сказать продакт»
- ❌ «Пальцем в небо, как чувствую»
- ❌ «Прикидываю по формуле»

— то это не то, что хотят услышать.

Ожидается:
- оценка **unit-экономики** АБ эксперимента
- оценка величин **прокраса метрики** на исторических экспериментах
- определение иных **стратегий по работе с MDE**

---

## Вопрос 4

**Мы — Самокат (сервис по доставке продуктов на дом). Провели АБ эксперимент в небольшом городке РФ. Зафиксировали невероятный положительный эффект на метрики. Руководство готово внедрять изменение по всей стране. Раскатываем?**

**Ответ:**

**Конечно же нет!**

Ключевая проблема — **нерепрезентативность** эксперимента. Мы не можем по небольшому городу судить и обобщать результаты на Москву и Петербург.

Наличие эффекта — это хорошо, но это лишь подтверждает потенциал идеи. Следующий шаг — проводить эксперименты на более репрезентативную аудиторию.

Этим часто грешат компании:
- провели тест и получили эффект в одной стране/городе → катим на все без проверки
- протестировали с положительным эффектом на Android и нет времени тестировать на iOS → катим на iOS без теста

---

## Вопрос 5

**Запустили АБ эксперимент. Целевая метрика — конверсия из регистрации в покупку. После запуска метрика в контроле 3 дня оставалась неизменной, а в тесте — стат. значимо упала с 10 до 6%. Мы теряем деньги. Останавливаем эксперимент или ждём полной выборки?**

**Ответ:**

Если твой ответ:
- ❌ «Останавливаем, мы же теряем деньги»
- ❌ «Ждём выборку, как же методология»

— то не совсем.

В эксперименте есть много всего, что могло пойти не так:
- баг, который сломал пользовательский путь
- проблема с логированием
- проблема с репрезентативностью
- эффект сопротивления новому

**В зависимости от причины будут понятны и дальнейшие действия.**
