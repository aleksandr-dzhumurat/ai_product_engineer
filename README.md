# AI Product engineer

Всем привет! Вы в репозитории курса AI product engineer.

Who am I?

* [linkedin](https://www.linkedin.com/in/aleksandr-dzhumurat/)
* [tg: Машинист продакшна](https://t.me/locomotive_production_driver)

Курс включает

* базовый ML - разбираем основые основные алгоримы (c формулами!) и ML библитеки (от scikit-learn до sentence-transformers)
* воркшопы по ML тулам: MLFlow, Streamlit, Langfuse, hyperopt.
* сервисы и их взаимодействие: Docker, S3 (minIO), FastAPI, PineCone, Telegram как пользовательский интерфейc, Ollama

Ресурсы
* Курс полностью доступен в google collab
* Если хотите запускать локально - требуется подготовить [локальное окружение](slides/lecture_0_prepare_env.md)
* Ссылка на данные для курса - её надо скопировать к себе [google drive](https://drive.google.com/drive/folders/1FMLKfNZZyFgzOhWjOiyeN3XvCsjT5-ET)
* [Вопросы к экзамену](slides/ml_breadth_questions.md) - тут по факту список вопросов для собесов
* [Stuart Russell, Peter Norvig: Artificial Intelligence: A Modern Approach](https://people.engr.tamu.edu/guni/csce625/slides/AI.pdf)

TG канал где я [пишу о продуктовом ML](https://t.me/locomotive_production_driver)

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
| **Основные материалы:**<br>• [ML intro: linear regression](jupyter_notebooks/vol_00_pre_requirements_01_machine_learning_intro.ipynb)<br>• [cracking linear regression](slides/cracking_linear_regression.md)<br><br> | [![Лекция 01 vol 1: введение в ML. Изучаем линейную регрессию](http://img.youtube.com/vi/sCIegfIcl10/0.jpg)](http://www.youtube.com/watch?v=sCIegfIcl10 "Лекция 01 vol 1: введение в ML. Изучаем линейную регрессию") |



**Дополнительные материалы** - если нужно освежить базовые знания:

* [Основы вероятности](jupyter_notebooks/vol_04_deep_dive_00_probability.ipynb)
    * [применение для system design](slides/probability_and_statistics.md)
* [Validation, generalization, overfitting](jupyter_notebooks/vol_00_pre_requirements_05_machine_learning_validation_generalization_overfitting.ipynb)
* [Classification](jupyter_notebooks/vol_00_pre_requirements_02_machine_learning_classification.ipynb)
* [cracking classification interview](slides/lecture_02_classification.md)
* [Naive Bayes classifier](jupyter_notebooks/vol_04_deep_dive_00_probability_hw_2_naive_bayes.ipynb)
* [feature engineering](jupyter_notebooks/vol_00_pre_requirements_06_feature_engineering.ipynb)
* [unsupervised algorithms](jupyter_notebooks/vol_00_pre_requirements_05_unsupervised_intro.ipynb)
* [clustering algorithms implementation](jupyter_notebooks/s/vol_04_deep_dive_10_unsupervised_learning_implementation.ipynb)
* [Gradient descent: linear regression](jupyter_notebooks/vol_00_pre_requirements_04_machine_learning_linear_regression_sgd_deep_dive.ipynb)
* [Trees, gradient boosting](jupyter_notebooks/vol_04_deep_dive_09_trees_boosting.ipynb)

#### Домашка

* [machine_learning_intro_hw_1.ipynb](jupyter_notebooks/vol_00_pre_requirements_01_machine_learning_intro_hw_1.ipynb)
* [machine_learning_intro_hw_2.ipynb](jupyter_notebooks/vol_00_pre_requirements_01_machine_learning_intro_hw_2.ipynb)
* [unsupervised_hw.ipynb](jupyter_notebooks/vol_00_pre_requirements_05_unsupervised_hw.ipynb)

### Лекция 01 vol 2: CRISP-DM

| Описание | Видео |
|----------|-------|
| Базовая лекция про этапы ML проекта - пригодится при разработке курсового проекта. | [![Лекция 01 vol 2: CRISP-DM](http://img.youtube.com/vi/7_ua8tWjQtA/0.jpg)](http://www.youtube.com/watch?v=7_ua8tWjQtA "Лекция 01 vol 2: CRISP-DM") |

## ML в проде

Как проводить демки - создание http сервисов и другие хитрости.

### Лекция 02 vol 1: Вывод модели в продакшн: чеклист

| Материалы | Видео |
|-----------|-------|
| [machine_learning_production_docker.ipynb](jupyter_notebooks/vol_01_ml_products_02_machine_learning_production_docker.ipynb)<br><br>Рассказ про system design, логирование, мониторинг. | [![Лекция 02 vol 1: Вывод модели в продакшн: чеклист](http://img.youtube.com/vi/xXCzeXK3y80/0.jpg)](http://www.youtube.com/watch?v=xXCzeXK3y80 "Лекция 02 vol 1: Вывод модели в продакшн: чеклист") |

### Лекция 02 vol 2: Вывод модели в продакшн: упаковка Docker. Streamlit

| Материалы | Видео |
|-----------|-------|
| • Разбираем как работает [train.py](./src/train.py)<br>• [домашка](jupyter_notebooks/vol_01_ml_products_02_machine_learning_production_docker_hw.ipynb)| [![Лекция 02 vol 2: Вывод модели в продакшн: упаковка Docker. Streamlit](http://img.youtube.com/vi/przsL26slSA/0.jpg)](http://www.youtube.com/watch?v=przsL26slSA "Лекция 02 vol 2: Вывод модели в продакшн: упаковка Docker. Streamlit") |

## NLP

Обработка текстов

### Лекция 03 vol 1 Векторизация текста Bag of Words

| Материалы | Видео |
|-----------|-------|
| [nlp_problems.ipynb](jupyter_notebooks/vol_01_ml_products_03_nlp_problems.ipynb) | [![Лекция 03 vol 1 Векторизация текста Bag of Words](http://img.youtube.com/vi/h0XiVQ-OvOI/0.jpg)](http://www.youtube.com/watch?v=h0XiVQ-OvOI "Лекция 03 vol 1 Векторизация текста Bag of Words") |

Домашка
* [nlp_problems_hw.ipynb](jupyter_notebooks/vol_01_ml_products_03_nlp_problems_hw.ipynb)
* [LabelStudio для разметки](slides/lecture_06_mlops_intro.md)

### Лекция 03 vol 2. Векторизация текста Word2Vec Transformers

| Материалы | Видео |
|-----------|-------|
| • [ollama embeddings](jupyter_notebooks/vol_02_ml_products_03_knowledgebase_rag.ipynb)<br>• [basic rag](jupyter_notebooks/vol_02_ml_products_02_search_rag.ipynb) | [![Лекция 03 vol 2. Векторизация текста Word2Vec Transformers](http://img.youtube.com/vi/csqW3HF_3p8/0.jpg)](http://www.youtube.com/watch?v=csqW3HF_3p8 "Лекция 03 vol 2. Векторизация текста Word2Vec Transformers") |

## ML: от моделей к проектам

Как провести этап моделирования

### Лекция 04 vol 1: Организация кода в ML проектах

| Материалы | Видео |
|-----------|-------|
| [Подготовка окружения](slides/lecture_0_prepare_env.md) | [![Лекция 04 vol 1: Организация кода в ML проектах](http://img.youtube.com/vi/yFGYz8XAw30/0.jpg)](http://www.youtube.com/watch?v=yFGYz8XAw30 "Лекция 04 vol 1: Организация кода в ML проектах") |

### Лекция 04 vol 2: Стадии CRISP-DM

создаем систему модерации контента

| Материалы | Видео |
|-----------|-------|
| [ML_project_flow.ipynb](jupyter_notebooks/vol_01_ml_products_01_ML_project_flow.ipynb) | [![Лекция 04 vol 2: Стадии CRISP-DM, создаем систему модерации контента](http://img.youtube.com/vi/NZrgApPYkpk/0.jpg)](http://www.youtube.com/watch?v=NZrgApPYkpk "Лекция 04 vol 2: Стадии CRISP-DM, создаем систему модерации контента") |

## Лекция 05: Трекинг экспериментов. MLFlow.

| Материалы | Видео |
|-----------|-------|
| • Разбираем код логирования экспериментов [parameters_tuning.py](src/parameters_tuning.py)<br>• [mflow_powered_classifier.ipynb](jupyter_notebooks/vol_01_ml_products_02_mflow_powered_classifier.ipynb)| [![Лекция 05: Трекинг экспериментов. MLFlow. MlOps.](http://img.youtube.com/vi/Zeo6fqrTc1A/0.jpg)](http://www.youtube.com/watch?v=Zeo6fqrTc1A "Лекция 05: Трекинг экспериментов. MLFlow.") |


## Лекция 06 vol 01: Поиск. ElasticSearch

| Материалы | Видео |
|-----------|-------|
| [ml_products_01_search.ipynb](jupyter_notebooks/vol_02_ml_products_01_search.ipynb) | [![Лекция 06 vol 01: Поиск. ElasticSearch](http://img.youtube.com/vi/aD6q_KAq6LU/0.jpg)](http://www.youtube.com/watch?v=aD6q_KAq6LU "Лекция 06 vol 01: Поиск. ElasticSearch") |

## Лекция 07: AI агенты

| Материалы | Видео |
|-----------|-------|
| [Building AI travel agent using Google ADK and Langfuse](https://medium.com/@alexandrdzhumurat/building-a-production-ready-ai-travel-agent-using-google-adk-and-langfuse-fc08f8ac1b3c) | [![Лекция 07: AI агенты](http://img.youtube.com/vi/5RabCJMPJE8/0.jpg)](http://www.youtube.com/watch?v=5RabCJMPJE8 "Лекция 07: AI агенты") |

## Лекция 08: Введение в рекомендательные системы

| Материалы | Видео |
|-----------|-------|
| • [recommendation_system.ipynb](jupyter_notebooks/vol_03_sys_design_01_recommendation_system.ipynb)<br>• [Recsys, Search, Ranking reading list](https://cold-scallion-5b8.notion.site/Recsys-Search-Ranking-ccbf1b9863ef4701a483a1585c8b51f1)| [![Лекция 08 vol 01: Введение в рекомендательные системы: content-based](http://img.youtube.com/vi/QQaCfwuR8gE/0.jpg)](http://www.youtube.com/watch?v=QQaCfwuR8gE "Лекция 08 vol 01: Введение в рекомендательные системы: content-based") |

## Лекция 09: Telegram бот + AI agent

| Материалы | Видео |
|-----------|-------|
| [Google ADK framework](dockerfiles/agent) | [![Лекция 09: Telegram бот + AI agent](http://img.youtube.com/vi/CJAptUEGojA/0.jpg)](http://www.youtube.com/watch?v=CJAptUEGojA "Лекция 09: Telegram бот + AI agent") |