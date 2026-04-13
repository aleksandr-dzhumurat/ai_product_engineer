# AI Product engineer

Всем привет! Вы в репозитории курса AI product engineer.

Who am I? Мой [linkedin](https://www.linkedin.com/in/aleksandr-dzhumurat/) и [tg: Машинист продакшна](https://t.me/locomotive_production_driver), где я пишу о продуктовом ML

Курс включает

* базовый ML - разбираем основые основные алгоримы (c формулами!) и ML библитеки (от scikit-learn до sentence-transformers)
* воркшопы по ML тулам: MLFlow, Streamlit, Langfuse, hyperopt.
* сервисы и их взаимодействие: Docker, S3 (minIO), FastAPI, PineCone, Telegram как пользовательский интерфейc, Ollama

Ресурсы
* Курс полностью доступен в Google Colab (ноутбуки адаптированы для запуска локально)
* Ссылка на данные для курса - скопируйие директорию с данными к себе [google drive](https://drive.google.com/drive/folders/1FMLKfNZZyFgzOhWjOiyeN3XvCsjT5-ET)
* [Вопросы к экзамену](slides/ml_breadth_questions.md) - тут по факту список вопросов для собесов
* Справочник: [Stuart Russell, Peter Norvig: Artificial Intelligence: A Modern Approach](https://people.engr.tamu.edu/guni/csce625/slides/AI.pdf)
* [Введение в ML от Константина Воронцова](http://www.machinelearning.ru/wiki/images/f/fc/Voron-ML-Intro-slides.pdf)
* [ML intro от Высшей школы экономики](https://yadi.sk/i/RajIebEkmqgzw)


Приятного просмотра!

## Содержание курса

| Тема | Ссылка |
|------|--------|
| Введение в ML | [🔗](#введение-в-ml) |
| ML в проде | [🔗](#ml-в-проде) |
| Введение в NLP | [🔗](#nlp) |
| ML lifecycle: от моделей к проектам | [🔗](#ml-от-моделей-к-проектам) |
| Лекция 05: Трекинг экспериментов. MLFlow. | [🔗](#лекция-05-трекинг-экспериментов-mlflow) |
| Лекция 06 vol 01: Поиск. ElasticSearch | [🔗](#лекция-06-vol-01-поиск-elasticsearch) |
| Лекция 07: AI агенты | [🔗](#лекция-07-ai-агенты) |
| Лекция 08 vol 01: Рекомендательные системы | [🔗](#лекция-08-vol-01-введение-в-рекомендательные-системы-content-based) |
| Лекция 09: Telegram бот + AI agent | [🔗](#лекция-09-telegram-бот--ai-agent) |

# Структура курса

## Введение в ML

### Лекция 01 vol 1: введение в ML. Изучаем линейную регрессию

| Материалы | Видео |
|-----------|-------|
| **Линейная регрессия:**<br>• [Сracking linear regression](slides/lecture_02_vol_01_linear_regression.md)<br>• [Notebook: linear regression](https://github.com/aleksandr-dzhumurat/ai_product_engineer/blob/main/jupyter_notebooks/vol_00_pre_requirements_01_machine_learning_intro.ipynb) | [![Лекция 01 vol 1: введение в ML. Изучаем линейную регрессию](http://img.youtube.com/vi/sCIegfIcl10/0.jpg)](http://www.youtube.com/watch?v=sCIegfIcl10 "Лекция 01 vol 1: введение в ML. Изучаем линейную регрессию") |
| **Классификация:**<br>• [Cracking classification](slides/lecture_02_vol_02_classification.md)<br>• [Notebook: classification](https://github.com/aleksandr-dzhumurat/ai_product_engineer/blob/main/jupyter_notebooks/vol_00_pre_requirements_02_machine_learning_classification.ipynb) | |
| **Gradient descent:**<br>• [Notebook: linear regression SGD deep dive](https://github.com/aleksandr-dzhumurat/ai_product_engineer/blob/main/jupyter_notebooks/vol_00_pre_requirements_04_machine_learning_linear_regression_sgd_deep_dive.ipynb) | |
| **Организация кода в ML проектах:**<br>• [Подготовка окружения](slides/lecture_00_prepare_env.md) | [![Лекция 04 vol 1: Организация кода в ML проектах](http://img.youtube.com/vi/yFGYz8XAw30/0.jpg)](http://www.youtube.com/watch?v=yFGYz8XAw30 "Лекция 04 vol 1: Организация кода в ML проектах") |

**Дополнительные материалы** - если нужно освежить базовые знания:

* [Notebook: Основы вероятности](https://github.com/aleksandr-dzhumurat/ai_product_engineer/blob/main/jupyter_notebooks/vol_04_deep_dive_00_probability.ipynb)
    * [применение для system design](slides/probability_statistics_ab_tests.md)
* [Notebook: Validation, generalization, overfitting](https://github.com/aleksandr-dzhumurat/ai_product_engineer/blob/main/jupyter_notebooks/vol_00_pre_requirements_05_machine_learning_validation_generalization_overfitting.ipynb)
* [Notebook: Naive Bayes classifier](https://github.com/aleksandr-dzhumurat/ai_product_engineer/blob/main/jupyter_notebooks/vol_04_deep_dive_00_probability_hw_2_naive_bayes.ipynb)
* [Notebook: unsupervised algorithms](https://github.com/aleksandr-dzhumurat/ai_product_engineer/blob/main/jupyter_notebooks/vol_00_pre_requirements_05_unsupervised_intro.ipynb)
* [Notebook: clustering algorithms implementation](https://github.com/aleksandr-dzhumurat/ai_product_engineer/blob/main/jupyter_notebooks/vol_04_deep_dive_10_unsupervised_learning_implementation.ipynb)
* [Notebook: Trees, gradient boosting](https://github.com/aleksandr-dzhumurat/ai_product_engineer/blob/main/jupyter_notebooks/vol_04_deep_dive_09_trees_boosting.ipynb)

#### Домашка

* [Notebook: machine_learning_intro_hw_1.ipynb](https://github.com/aleksandr-dzhumurat/ai_product_engineer/blob/main/jupyter_notebooks/vol_00_pre_requirements_01_machine_learning_intro_hw_1.ipynb)
* [Notebook: machine_learning_intro_hw_2.ipynb](https://github.com/aleksandr-dzhumurat/ai_product_engineer/blob/main/jupyter_notebooks/vol_00_pre_requirements_01_machine_learning_intro_hw_2.ipynb)
* [Notebook: unsupervised_hw.ipynb](https://github.com/aleksandr-dzhumurat/ai_product_engineer/blob/main/jupyter_notebooks/vol_00_pre_requirements_05_unsupervised_hw.ipynb)

### Лекция 01 vol 2: CRISP-DM

| Описание | Видео |
|----------|-------|
| Базовая лекция про этапы ML проекта - пригодится при разработке курсового проекта.<br>[Notebook: feature engineering](https://github.com/aleksandr-dzhumurat/ai_product_engineer/blob/main/jupyter_notebooks/vol_00_pre_requirements_06_feature_engineering.ipynb) | [![Лекция 01 vol 2: CRISP-DM](http://img.youtube.com/vi/7_ua8tWjQtA/0.jpg)](http://www.youtube.com/watch?v=7_ua8tWjQtA "Лекция 01 vol 2: CRISP-DM") |

## ML в проде

Как проводить демки - создание http сервисов и другие хитрости.

### Лекция 02 vol 1: Вывод модели в продакшн: чеклист

| Материалы | Видео |
|-----------|-------|
| [Notebook: machine_learning_production_docker.ipynb](https://github.com/aleksandr-dzhumurat/ai_product_engineer/blob/main/jupyter_notebooks/vol_01_ml_products_02_machine_learning_production_docker.ipynb)<br><br>Рассказ про system design, логирование, мониторинг. | [![Лекция 02 vol 1: Вывод модели в продакшн: чеклист](http://img.youtube.com/vi/xXCzeXK3y80/0.jpg)](http://www.youtube.com/watch?v=xXCzeXK3y80 "Лекция 02 vol 1: Вывод модели в продакшн: чеклист") |

### Лекция 02 vol 2: Вывод модели в продакшн: упаковка Docker. Streamlit

| Материалы | Видео |
|-----------|-------|
| • Разбираем как работает [train.py](https://github.com/aleksandr-dzhumurat/ai_product_engineer/blob/main/src/train.py)| [![Лекция 02 vol 2: Вывод модели в продакшн: упаковка Docker. Streamlit](http://img.youtube.com/vi/przsL26slSA/0.jpg)](http://www.youtube.com/watch?v=przsL26slSA "Лекция 02 vol 2: Вывод модели в продакшн: упаковка Docker. Streamlit") |

## NLP

Обработка текстов

### Лекция 03 vol 1 Векторизация текста Bag of Words

| Материалы | Видео |
|-----------|-------|
| [Notebook: nlp_problems.ipynb](https://github.com/aleksandr-dzhumurat/ai_product_engineer/blob/main/jupyter_notebooks/vol_01_ml_products_03_nlp_problems.ipynb) | [![Лекция 03 vol 1 Векторизация текста Bag of Words](http://img.youtube.com/vi/h0XiVQ-OvOI/0.jpg)](http://www.youtube.com/watch?v=h0XiVQ-OvOI "Лекция 03 vol 1 Векторизация текста Bag of Words") |

Домашка
* [Notebook: nlp_problems_hw.ipynb](https://github.com/aleksandr-dzhumurat/ai_product_engineer/blob/main/jupyter_notebooks/vol_01_ml_products_03_nlp_problems_hw.ipynb)
* [LabelStudio для разметки](slides/ml_projects/label_studio.md)

### Лекция 03 vol 2. Векторизация текста Word2Vec Transformers

| Материалы | Видео |
|-----------|-------|
| • [Notebook: ollama embeddings](https://github.com/aleksandr-dzhumurat/ai_product_engineer/blob/main/jupyter_notebooks/vol_02_ml_products_03_knowledgebase_rag.ipynb)<br>• [Notebook: basic rag](https://github.com/aleksandr-dzhumurat/ai_product_engineer/blob/main/jupyter_notebooks/vol_02_ml_products_02_search_rag.ipynb) | [![Лекция 03 vol 2. Векторизация текста Word2Vec Transformers](http://img.youtube.com/vi/csqW3HF_3p8/0.jpg)](http://www.youtube.com/watch?v=csqW3HF_3p8 "Лекция 03 vol 2. Векторизация текста Word2Vec Transformers") |

## ML: от моделей к проектам

Как провести этап моделирования

### Лекция 04: Стадии CRISP-DM

создаем систему модерации контента

| Материалы | Видео |
|-----------|-------|
| [Notebook: ML_project_flow.ipynb](https://github.com/aleksandr-dzhumurat/ai_product_engineer/blob/main/jupyter_notebooks/vol_01_ml_products_01_ML_project_flow.ipynb) | [![Лекция 04 vol 2: Стадии CRISP-DM, создаем систему модерации контента](http://img.youtube.com/vi/NZrgApPYkpk/0.jpg)](http://www.youtube.com/watch?v=NZrgApPYkpk "Лекция 04 vol 2: Стадии CRISP-DM, создаем систему модерации контента") |

## Лекция 05: Трекинг экспериментов. MLFlow.

| Материалы | Видео |
|-----------|-------|
| • Разбираем код логирования экспериментов [parameters_tuning.py](https://github.com/aleksandr-dzhumurat/ai_product_engineer/blob/main/src/parameters_tuning.py)<br>• [Notebook: mflow_powered_classifier.ipynb](https://github.com/aleksandr-dzhumurat/ai_product_engineer/blob/main/jupyter_notebooks/vol_01_ml_products_02_mflow_powered_classifier.ipynb)| [![Лекция 05: Трекинг экспериментов. MLFlow. MlOps.](http://img.youtube.com/vi/Zeo6fqrTc1A/0.jpg)](http://www.youtube.com/watch?v=Zeo6fqrTc1A "Лекция 05: Трекинг экспериментов. MLFlow.") |


## Лекция 06 vol 01: Поиск. ElasticSearch

| Материалы | Видео |
|-----------|-------|
| [Notebook: ml_products_01_search.ipynb](https://github.com/aleksandr-dzhumurat/ai_product_engineer/blob/main/jupyter_notebooks/vol_02_ml_products_01_search.ipynb) | [![Лекция 06 vol 01: Поиск. ElasticSearch](http://img.youtube.com/vi/aD6q_KAq6LU/0.jpg)](http://www.youtube.com/watch?v=aD6q_KAq6LU "Лекция 06 vol 01: Поиск. ElasticSearch") |

## Лекция 07: AI агенты

| Материалы | Видео |
|-----------|-------|
| [Building AI travel agent using Google ADK and Langfuse](https://medium.com/@alexandrdzhumurat/building-a-production-ready-ai-travel-agent-using-google-adk-and-langfuse-fc08f8ac1b3c) | [![Лекция 07: AI агенты](http://img.youtube.com/vi/5RabCJMPJE8/0.jpg)](http://www.youtube.com/watch?v=5RabCJMPJE8 "Лекция 07: AI агенты") |

## Лекция 08: Введение в рекомендательные системы

| Материалы | Видео |
|-----------|-------|
| • [Notebook: recommendation_system.ipynb](https://github.com/aleksandr-dzhumurat/ai_product_engineer/blob/main/jupyter_notebooks/vol_03_sys_design_01_recommendation_system.ipynb)<br>• [Recsys, Search, Ranking reading list](slides/lecture_05_recommender_system.md)| [![Лекция 08 vol 01: Введение в рекомендательные системы: content-based](http://img.youtube.com/vi/QQaCfwuR8gE/0.jpg)](http://www.youtube.com/watch?v=QQaCfwuR8gE "Лекция 08 vol 01: Введение в рекомендательные системы: content-based") |

## Лекция 09: Telegram бот + AI agent

| Материалы | Видео |
|-----------|-------|
| [Google ADK framework](https://github.com/aleksandr-dzhumurat/ai_product_engineer/tree/main/dockerfiles/agent) | [![Лекция 09: Telegram бот + AI agent](http://img.youtube.com/vi/CJAptUEGojA/0.jpg)](http://www.youtube.com/watch?v=CJAptUEGojA "Лекция 09: Telegram бот + AI agent") |


# Подробнее о курсе

![ml_mindmap](img/ml_mindmap.png)


Курс построен таким образом чтобы дать максимально широкое понимание темы “Запуск ML продукта” с глубоким пониманием каждого отдельного этапа

- Business understanding
- Exploratory data analysis
- Experiment planning
- MVP and prepare service for deploy

# Программа курса

Будет шесть занятий (с лабораторными), в каждой из которых разберём одну тему из области ML и один прикладной инструмент

Темы

- [Многорукие бандиты](https://youtu.be/3jurSlIe2Q8?si=BKfPlgMis6G77ZTg) как пример realtime ML, изучаем FastAPI.
- [Обучение без учителя](https://youtu.be/TT5Kd1Zmwpo?si=WDn0QKIH3yLhNH8m): кластеризация, снижение размерности, изучаем Streamlit

Чего не будет в курсе

- глубокого погпужения в нейросети не будет
- devops часть трогать не будем

# Курсовой проект

Для курсового проекта нужно выбрать и реализовать в составе команды ML проект. Идеал - команда из трех специалистов

- Аналитик
- Data Scientist
- ML engineer

В качестве источника данных выбираем [Delivery Hero Recommendation Dataset](https://dl.acm.org/doi/pdf/10.1145/3604915.3610242)

**Важно:** данныe [доступны в Google Drive](https://github.com/deliveryhero/dh-reco-dataset)

Темы курсовых проектов представлены ниже (нужно выбрать одну тему на команду)

Курсовой проект должен состоять из трех частей

- EDA (jupyter notebook) - делает Data аналитик
- Модель - делает Data Scientist на основании
- Интерфейс (Streamlit, либо React) - делает ML инженер

## Рекомендательная система

- [Прогноз корзины](https://youtu.be/872uZTqY85k?si=pGfQaKTrsM9XflZE)
- Рекомендация нового ресторана для пользователя
- [Товары-заменители](https://youtu.be/tbTekebpK6E?si=iYBMOsgBryCcsZUd)
- Блок “С этим товаром покупают”
- [Алгоритм DPP](https://github.com/laming-chen/fast-map-dpp/blob/master/dpp.py): добавить в рекомендательный сервис механизмы разнообразия

## Контентные сервисы

- [Категоризация товаров](https://www.youtube.com/watch?v=38P2RIkHolQ&t=1240s): прогноз кухни для продукта
- Составление рациона на день на основе предпочтений пользователя
- Автокомплит поисковой строки
- кросс-рекомендации ресторанов между городами

## Оптимизация маркетплейса

- Модель ценности заказа для курьера (какой заказ выбрать по расстоянию и цене)
- [Прогнозирование спроса в районе](https://youtu.be/YnsJ4l0Z3o8?si=_jp3JkGEgZjUrvUS)
- [Cимуляция города](https://youtu.be/F_bN3CuRPU8?si=TKbazVflyi7ED_R4)
- Поиск локации локации для ресторана
