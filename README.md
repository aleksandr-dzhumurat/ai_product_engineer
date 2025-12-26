# AI Product engineer

Всем привет! Вы в репозитории курса AI product engineer.

Курс включает

* базовый ML - разбираем основые основные алгоримы (c формулами!)
* воркшопы по ML тулам: MLFlow, Streamlit, 
* сервисы и их взаимодействие: Docker, FastAPI, PineCone, Telegram как пользовательский интерфейc

Курс полностью доступен в google collab
Если хотите запускать локально - требуется подготовить [локальное окружение](slides/lecture_0_prepare_env.md)

Ссылка на данные для курса - её надо скопировать к себе [google drive](https://drive.google.com/drive/folders/1FMLKfNZZyFgzOhWjOiyeN3XvCsjT5-ET)


[Вопросы к экзамену](slides/lecture_0_prepare_env.md) - тут по факту список вопросов для собесов

TG канал где я [пишу о продуктовом ML](https://t.me/locomotive_production_driver)

## Введение в ML

### Лекция 01 vol 1: введение в ML. Изучаем линейную регрессию

[![Лекция 01 vol 1: введение в ML. Изучаем линейную регрессию](http://img.youtube.com/vi/sCIegfIcl10/0.jpg)](http://www.youtube.com/watch?v=sCIegfIcl10 "Лекция 01 vol 1: введение в ML. Изучаем линейную регрессию")

* [ML intro: linear regression](jupyter_notebooks/vol_00_pre_requirements_01_machine_learning_intro.ipynb)
* [cracking linear regression](slides/cracking_linear_regression.md)

Дополнительные метериалы
* [Validation, generalization, overfitting](jupyter_notebooks/vol_00_pre_requirements_05_machine_learning_validation_generalization_overfitting.ipynb)
* [Classification](jupyter_notebooks/vol_00_pre_requirements_02_machine_learning_classification.ipynb)
* [Naive Bayes classifier](jupyter_notebooks/vol_04_deep_dive_00_probability_hw_2_naive_bayes.ipynb)
* [feature engineering](jupyter_notebooks/vol_00_pre_requirements_06_feature_engineering.ipynb)
* [unsupervised algorithms](jupyter_notebooks/vol_00_pre_requirements_05_unsupervised_intro.ipynb)
* [clustering algorithms implementation](jupyter_notebooks/vol_04_deep_dive_06_unsupervised_learning_implementation.ipynb)
* [Gradient descent: linear regression](jupyter_notebooks/vol_00_pre_requirements_04_machine_learning_linear_regression_sgd_deep_dive.ipynb)
* [Trees, gradient boosting](jupyter_notebooks/vol_04_deep_dive_09_trees_boosting.ipynb)


### Лекция 01 vol 2: CRISP-DM

[![Лекция 01 vol 2: CRISP-DM](http://img.youtube.com/vi/7_ua8tWjQtA/0.jpg)](http://www.youtube.com/watch?v=7_ua8tWjQtA "Лекция 01 vol 2: CRISP-DM")

Базовая лекция про этапы ML проекта - пригодится при разработке курсового проекта.

ML в проде

### Лекция 02 vol 1: Вывод модели в продакшн: чеклист

[![Лекция 02 vol 1: Вывод модели в продакшн: чеклист](http://img.youtube.com/vi/xXCzeXK3y80/0.jpg)](http://www.youtube.com/watch?v=xXCzeXK3y80 "Лекция 02 vol 1: Вывод модели в продакшн: чеклист")

Рассказ про system design, логирование, мониторинг.

## Лекция 02 vol 2: Вывод модели в продакшн: упаковка Docker. Streamlit


[![Лекция 02 vol 2: Вывод модели в продакшн: упаковка Docker. Streamlit](http://img.youtube.com/vi/przsL26slSA/0.jpg)](http://www.youtube.com/watch?v=przsL26slSA "Лекция 02 vol 2: Вывод модели в продакшн: упаковка Docker. Streamlit")

Разбираем как работает [train.py](./src/train.py)

## Лекция 03 vol 1 Векторизация текста Bag of Words


[![Лекция 03 vol 1 Векторизация текста Bag of Words](http://img.youtube.com/vi/h0XiVQ-OvOI/0.jpg)](http://www.youtube.com/watch?v=h0XiVQ-OvOI "Лекция 03 vol 1 Векторизация текста Bag of Words")

## Лекция 03 vol 2. Векторизация текста Word2Vec Transformers


[![Лекция 03 vol 2. Векторизация текста Word2Vec Transformers](http://img.youtube.com/vi/csqW3HF_3p8/0.jpg)](http://www.youtube.com/watch?v=csqW3HF_3p8 "Лекция 03 vol 2. Векторизация текста Word2Vec Transformers")

## Лекция 04 vol 1: Организация кода в ML проектах


[![Лекция 04 vol 1: Организация кода в ML проектах](http://img.youtube.com/vi/yFGYz8XAw30/0.jpg)](http://www.youtube.com/watch?v=yFGYz8XAw30 "Лекция 04 vol 1: Организация кода в ML проектах")

## Лекция 04 vol 2: Стадии CRISP-DM, создаем систему модерации контента


[![Лекция 04 vol 2: Стадии CRISP-DM, создаем систему модерации контента](http://img.youtube.com/vi/NZrgApPYkpk/0.jpg)](http://www.youtube.com/watch?v=NZrgApPYkpk "Лекция 04 vol 2: Стадии CRISP-DM, создаем систему модерации контента")

* [jupyter notebook](jupyter_notebooks/vol_01_ml_products_01_ML_project_flow.ipynb)

## Лекция 05: Трекинг экспериментов. MLFlow.

[![Лекция 05: Трекинг экспериментов. MLFlow. MlOps.](http://img.youtube.com/vi/Zeo6fqrTc1A/0.jpg)](http://www.youtube.com/watch?v=Zeo6fqrTc1A "Лекция 05: Трекинг экспериментов. MLFlow.")


## Лекция 06 vol 01: Поиск. ElasticSearch

[![Лекция 06 vol 01: Поиск. ElasticSearch](http://img.youtube.com/vi/aD6q_KAq6LU/0.jpg)](http://www.youtube.com/watch?v=aD6q_KAq6LU "Лекция 06 vol 01: Поиск. ElasticSearch")

## Лекция 07: AI агенты


[![Лекция 07: AI агенты](http://img.youtube.com/vi/5RabCJMPJE8/0.jpg)](http://www.youtube.com/watch?v=5RabCJMPJE8 "Лекция 07: AI агенты")

## Лекция 08 vol 01: Введение в рекомендательные системы: content-based


[![Лекция 08 vol 01: Введение в рекомендательные системы: content-based](http://img.youtube.com/vi/QQaCfwuR8gE/0.jpg)](http://www.youtube.com/watch?v=QQaCfwuR8gE "Лекция 08 vol 01: Введение в рекомендательные системы: content-based")

## Лекция 09: Telegram бот + AI agent


[![Лекция 09: Telegram бот + AI agent](http://img.youtube.com/vi/CJAptUEGojA/0.jpg)](http://www.youtube.com/watch?v=CJAptUEGojA "Лекция 09: Telegram бот + AI agent")