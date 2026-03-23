# Рекомендательные системы

[![Recommender Systems](http://img.youtube.com/vi/fEbwRMnviqA/0.jpg)](http://www.youtube.com/watch?v=fEbwRMnviqA "Recommender Systems")

[Jupyter Notebook](../src/jupyter_notebooks/lecture_05_recommended_system.ipynb)
[Colab notebook](https://drive.google.com/file/d/1z8K06ZiYKPFhOgNkPZX5eCiM7jGvzN-n/view?usp=sharing)

[slides](https://docs.google.com/presentation/d/1_pbTCGMs1Vfnoqe2GroLUnvKFE5uPKI_x87OVdXHH_I/edit?usp=sharing)

Вопросы для самопроверки

* Основы рекомендательных систем: бизнес-применение
* Content-based vs collaborative systems
* -
* -

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

# Reading list

# Recsys in production (recommender systems)

# Common resources

* [Personalized Machine Learning book.pdf](https://cseweb.ucsd.edu/~jmcauley/pml/pml_book.pdf)
* [Recommendations in Lyft](https://eng.lyft.com/the-recommendation-system-at-lyft-67bc9dcc1793)
* [Lyft Engineering: geo embeddings](https://eng.lyft.com/lyft2vec-embeddings-at-lyft-d4231a76d219)
* [recys design](https://www.theinsaneapp.com/2021/03/system-design-and-recommendation-algorithms.html)

# Recsys tricks

* [Position bias](https://eugeneyan.com/writing/position-bias/)
* [Bragin: position bias](https://www.youtube.com/watch?v=5dEzcKTkojQ&ab_channel=ODSAIRu)
* [Propensity score](https://towardsdatascience.com/a-review-of-propensity-score-modelling-approaches-19af9ecd60d9)
* [Propensity modeling](https://youtu.be/NdZaM0_mhVM?si=FKOgpcwvnmV_gg8Q)
* [Beyong propensity](https://www.notion.so/acabce8dc0de4bbdb6d61966432e3e5c?pvs=21)
* [Counterfactual-evaluation](https://eugeneyan.com/writing/counterfactual-evaluation/)

## Cases

* [Beyond Ranking: Optimizing Whole-Page Presentation](https://dl.acm.org/doi/pdf/10.1145/2835776.2835824)
* [Pinterest blog overview](https://www.notion.so/Pinterest-Blog-overview-d5a47fdceb1d442f8196d1a20c892834?pvs=21)
* [Learning to rank introduction](https://medium.com/@yahya12/introduction-to-learning-to-rank-666d39a50a39)
* [Deep Learning Recommender Systems - Cristian Martinez, Ilia Ivanov](https://www.youtube.com/live/LWAQUgJOYm0?si=7Xqs4Nb12fCqkr5t)
* [Innovative recommendations applications using Two tower embeddings in Uber](https://www.uber.com/en-GB/blog/innovative-recommendation-applications-using-two-tower-embeddings/?uclick_id=982ceced-4874-4793-8a00-9bfa528d87cc)
* [Two tower](https://habr.com/ru/companies/wildberries/articles/938938/) from Wildberries
* [Two tower GPU training](https://medium.com/pinterest-engineering/gpu-serving-two-tower-models-for-lightweight-ads-engagement-prediction-5a0ffb442f3b)
* [Entropy Sampling — How to balance “relevance” and “surprise” in recommendation](https://medium.com/@reika.k.fujimura/entropy-sampling-how-to-balance-relevance-and-surprise-in-recommendation-2223417a38ce)
* [generating-item-recommendatations-with-open-source-service-metarank](https://medium.com/metarank/generating-item-recommendatations-with-open-source-personalization-service-metarank-49f0689d8e8d)
* [Metadata item2vec](https://medium.com/swiggy-bytes/item2vec-with-metadata-incorporating-side-information-in-item-embeddings-167fb8d3f404)
* [Twitter recommendations algorithm](https://www.linkedin.com/feed/update/activity:7047936669689233409) + [Linkedin post](https://www.linkedin.com/feed/update/activity:7047964173841879040)
* [SVD vs matrix factorisation](https://www.cse.iitd.ac.in/~mausam/papers/cikm21.pdf)
* [Neo4j graph recommender system](https://medium.com/@susmitpy/recommender-system-using-neo4j-hands-on-part-2-557c36772c7)
* [optimizing-connections-mathematical-optimization-within-graphs](https://towardsdatascience.com/optimizing-connections-mathematical-optimization-within-graphs-7364e082a984)
* [Instagramm recsys](https://engineering.fb.com/2023/08/09/ml-applications/scaling-instagram-explore-recommendations-system/?trk=feed_main-feed-card_feed-article-content)
* [Uber two tower](https://www.uber.com/en-GB/blog/innovative-recommendation-applications-using-two-tower-embeddings/?uclick_id=982ceced-4874-4793-8a00-9bfa528d87cc&utm_source=substack&utm_medium=email)
* [personalized-recommendations-two-tower-models-for-retrieval](https://medium.com/@ManishChablani/personalized-recommendations-two-tower-models-for-retrieval-c934c140089a)
* [Time-Aware Item Weighting for the Next Basket Recommendations](https://arxiv.org/pdf/2307.16297.pdf)
* [counters-in-recommendations-from-exponential-decay-to-position-debiasing](https://roizner.medium.com/counters-in-recommendations-from-exponential-decay-to-position-debiasing-30a6175bba5)
* [[Youtube] Neural nets recommender (Avito)](https://youtu.be/FLiJSh3G5qo?si=U5u-icaStTfFjmZ-)
* [Recommender Systems with Generative Retrieval](https://arxiv.org/pdf/2305.05065.pdf)
* [SPAR: Personalized Content-Based Recommendation via Long Engagement Attention](https://huggingface.co/papers/2402.10555)
* [Linkedin recommender](https://blog.gopenai.com/paper-review-lirank-industrial-large-scale-ranking-models-at-linkedin-60bc5d21a332)
* [Next Basket Prediction](https://www.youtube.com/watch?v=7dCf9s4ZAv8)
* [scaling facebook recommender systems](https://engineering.fb.com/2023/08/09/ml-applications/scaling-instagram-explore-recommendations-system/)
* [foundation-model-for-personalized-recommendation](https://netflixtechblog.medium.com/foundation-model-for-personalized-recommendation-1a0bd8e02d39)
* [A Machine-Learning Item Recommendation System for Video Games](https://arxiv.org/pdf/1806.04900.pdf)
* [Personalized Bundle Recommendation in Online Games](https://arxiv.org/pdf/2104.05307.pdf)
* [Sequential Recommendation in Online Games with Multiple Sequences, Tasks and User Levels](https://arxiv.org/pdf/2102.06950.pdf)
* [4-stage recommender system](https://www.linkedin.com/feed/update/activity:7235938187540332544)
* [re-ranking in WB](https://t.me/wildrecsys/45)
* [RAG in ecom](https://youtu.be/EqUjf5X6IPE?t=3014)

# Bandits & RL

* [RL in recsys, overview](https://scitator.medium.com/rl-in-recsys-an-overview-e02815019a8f)
* [Deep reinforcement learning from human preferences](https://arxiv.org/abs/1706.03741)
* [Yandex Taxi: transformers Next Best Action](https://www.youtube.com/watch?v=GdFqJPYyR3Q&t=393s)
* [Transformers for personal recs](https://youtu.be/uG8vrdcDeJU?si=qHpew9m_1QiOTuKj)
* [RL: next best action](https://youtu.be/gtmQ7A5_TPY?t=551)
* [Paper Review: RecMind: Large Language Model Powered Agent For Recommendation](https://andlukyane.com/blog/paper-review-recmind)
* [MakeMyTrip: contextual bandits](https://tech.makemytrip.com/deep-contextual-bandits-for-model-selection-774eec6e5603)
* [Contextual Bandits and Reinforcement Learning](https://towardsdatascience.com/contextual-bandits-and-reinforcement-learning-6bdfeaece72a)
* [https://www.realworldml.net/the-hands-on-reinforcement-learning-course](https://www.realworldml.net/the-hands-on-reinforcement-learning-course?trk=feed_main-feed-card_feed-article-content)
* [develop-your-first-ai-agent-deep-q-learning](https://medium.com/towards-data-science/develop-your-first-ai-agent-deep-q-learning-375876ee2472)
* [Lyft reinforcement learning platform](https://eng.lyft.com/lyfts-reinforcement-learning-platform-670f77ff46ec)
* [simulating-content-personalization-with-contextual-bandits](https://medium.com/mlearning-ai/simulating-content-personalization-with-contextual-bandits-6f4efb902af)
* [whats-next-using-multi-armed-bandits-in-a-content-feed](https://scott-in-2d.medium.com/whats-next-using-multi-armed-bandits-in-a-content-feed-285d76876d68)
* [Application of Multi-Armed Bandits to Promotion Ranking in MoMo](https://medium.com/@vvviet123/application-of-multi-armed-bandits-to-promotion-ranking-in-momo-eac28dbcf8bb)
* [News feed recommender with various feedback](https://arxiv.org/pdf/2102.04903.pdf)
* [MakeMyTrip: Sequential recommender (Bert4Rec)](https://tech.makemytrip.com/hotel-ranking-personalization-at-makemytrip-using-sequential-recommenders-bert4rec-fd02f994c1cd)
* [Generating diverse travel recommendations](https://medium.com/expedia-group-tech/generating-diverse-travel-recommendations-76688f49c812)
* [Evolution of search ranking at Thumbtack](https://medium.com/thumbtack-engineering/evolution-of-search-ranking-at-thumbtack-f7a69fd0da13)
* [accelerating-ranking-experimentation-at-thumbtack-with-interleaving](https://medium.com/thumbtack-engineering/accelerating-ranking-experimentation-at-thumbtack-with-interleaving-20cbe7837edf)
* [A better clickthrough-rate: how Pinterest upgraded everyones favorite engagement metric](https://medium.com/pinterest-engineering/a-better-clickthrough-rate-how-pinterest-upgraded-everyones-favorite-engagement-metric-27f6fa6cba14)
* [TikTok recommender](https://www.linkedin.com/posts/damienbenveniste_machinelearning-datascience-artificialintelligence-activity-7054840232403169281-JWN7?utm_source=share&utm_medium=member_desktop)
* [Personalized recommendations: Two tower models for retrieval](https://medium.com/@ManishChablani/personalized-recommendations-two-tower-models-for-retrieval-c934c140089a)
* [Yandex: neural net multi tower](https://www.youtube.com/watch?v=0ZU-FtLO4Fw&t=2157s)
* [tthe-rise-of-two-tower-models-in-recommender-systems](https://medium.com/towards-data-science/the-rise-of-two-tower-models-in-recommender-systems-be6217494831)
* [Personalising the Swiggy homepage layout part I](https://bytes.swiggy.com/personalising-the-swiggy-homepage-layout-part-i-1048dba5e703)
* [Uber ETA prediction](https://www.linkedin.com/posts/damienbenveniste_machinelearning-datascience-artificialintelligence-activity-7049763143027195904-U2jR?utm_source=share&utm_medium=member_desktop)
* [Uber Two Tower](https://www.uber.com/en-GB/blog/innovative-recommendation-applications-using-two-tower-embeddings/?uclick_id=982ceced-4874-4793-8a00-9bfa528d87cc)
* [TikTok retrieval](https://medium.com/@valeriybabushkin/article-review-of-deep-retrieval-learning-a-retrievable-structure-for-large-scale-recommendations-57a512e20397)
* [tech_makemytrip](https://tech.makemytrip.com/)
* [SaARec Amazon](https://github.com/microsoft/recommenders/blob/main/examples/00_quick_start/sasrec_amazon.ipynb)
* [Inside our recommender system: Data pipeline execution and monitoring](https://medium.com/tech-getyourguide/inside-our-recommender-system-data-pipeline-execution-and-monitoring-c95f1316cceb)
* [Factorization Machines](https://medium.com/@datadote/factorization-machines-pictures-code-pytorch-9fca1c300838)
* [Music recommendation in VK](https://habr.com/ru/companies/vk/articles/683152/)
* [Multi objective](https://medium.com/@subirverma/multi-objective-ranking-in-large-scale-e-commerce-recommender-systems-9bab88bc00a8)
* [Tabby transformers](https://open.substack.com/pub/alextuzovsky/p/which-architecture-to-choose-for)

# Other algorithms

* [Recommendations at MakeMyTrip](https://tech.makemytrip.com/)
* [Библиотека рекомендаций внутри Яндекса](https://youtu.be/WQZ6a8ryVvU)
* [Practical Lessons from Predicting clicks Facebook](http://quinonero.net/Publications/predicting-clicks-facebook.pdf)
* [BPR for recsys](https://towardsdatascience.com/recommender-system-using-bayesian-personalized-ranking-d30e98bba0b9)
* [Multi-objective ranking](https://medium.com/@ebaytechblog/multi-objective-ranking-for-promoted-auction-items-293bf204574f)
* [multi-relevance-ranking-model-for-similar-item-recommendation](https://medium.com/@ebaytechblog/multi-relevance-ranking-model-for-similar-item-recommendation-a1c834938f0f)
* [introduction to ranking algorithms](https://medium.com/towards-data-science/introduction-to-ranking-algorithms-4e4639d65b8)
* [Hastie](https://hastie.su.domains/ISLP/ISLP_website.pdf)
* [FeedRec: News Feed Recommendation with Various User Feedbacks](https://arxiv.org/abs/2102.04903)

## Seacrh

* [Sparse information Retrieval](https://itnext.io/deep-learning-in-information-retrieval-part-i-introduction-and-sparse-retrieval-12de0423a0b9)
* [information retrieval](https://dl.acm.org/doi/book/10.1145/3674127)
* [how-to-build-a-generative-search-engine-for-your-local-files-using-llama-3](https://towardsdatascience.com/how-to-build-a-generative-search-engine-for-your-local-files-using-llama-3-399551786965)
* [Multi-modal RAG](https://www.deeplearning.ai/short-courses/building-multimodal-search-and-rag)
* [Semantic Search with Transformers and Faiss](https://towardsdatascience.com/how-to-build-a-semantic-search-engine-with-transformers-and-faiss-dcbea307a0e8)
* [TinySearch](https://arxiv.org/pdf/1908.02451.pdf)
* [hybrid search](https://www.innoventsolutions.com/search-technology/mongos-guide-to-effective-hybrid-search/)
* [reciprocal rank fusion](https://www.mongodb.com/docs/atlas/atlas-vector-search/tutorials/reciprocal-rank-fusion/)
* [zincsearch](https://zincsearch-docs.zinc.dev/api/search/search/#response)
* [dagster](https://medium.com/indiciumtech/data-ingestion-with-the-dagster-embedded-elt-library-60d860321d45)
* [zincsearch](https://zincsearch-docs.zinc.dev/)

## Deep Learning in Information Retrieval

* [Part 1: sparse retrieval](https://itnext.io/deep-learning-in-information-retrieval-part-i-introduction-and-sparse-retrieval-12de0423a0b9)
* [Part 2: dense retrieval](https://medium.com/@aikho/deep-learning-in-information-retrieval-part-ii-dense-retrieval-1f9fecb47de9)
* [Part 3: deep learning](https://itnext.io/deep-learning-in-information-retrieval-part-iii-ranking-da511f2dc325)

## Search meetup

* [Yandex.Toloka:Search](https://www.youtube.com/watch?v=GenUVbQj1Qc)
* [Yandex.Market:Coldstart](https://www.youtube.com/watch?v=hKjM_Jv5PIg)
* [Aliexpress:Search](https://www.youtube.com/watch?v=Z8RaotO4fZg)
* [Zen-meetup: multi-armed bandits](https://www.youtube.com/watch?v=VDhwkOi5Yvo)
* [Modern search systems](https://www.linkedin.com/posts/philipp-schmid-a6a2bb196_how-would-you-build-a-modern-search-engine-activity-7362746388637491201-CVVG): embeddings + transformers
* [ozon: query prediction](https://habr.com/en/companies/ozontech/articles/990180/)
* [AutoRu: ML&Search](https://youtu.be/5wv26h2ridM)
* [Search suggestions Citymobil](https://habr.com/ru/company/citymobil/blog/519556/)
* [VK: search](https://youtu.be/oMQjl8NBFuE)
* [UZUM: search](https://habr.com/ru/companies/uzum/articles/753094/)
* [UZUM: hybrid search](https://habr.com/ru/companies/uzum/articles/816773)
* [Hybrid search: eleasticsearch](https://www.elastic.co/what-is/hybrid-search)
* [instacart nulti modal search](https://tech.instacart.com/multi-modal-catalog-attribute-extraction-platform-at-instacart-5ebeb0073dfa)
* [RabotaRU: BERT in production](https://youtu.be/O589t08FfIY)
* [BERTtopic](https://towardsdatascience.com/topics-per-class-using-bertopic-252314f2640)
* [Build a Search Engine for Medium Stories Using Streamlit and Elasticsearch](https://betterprogramming.pub/build-a-search-engine-for-medium-stories-using-streamlit-and-elasticsearch-b6e717819448)
* [Персонализация поиска в Auto ru](https://www.youtube.com/watch?v=5wv26h2ridM)
* [Дедупликация объявлений в Циан](https://www.linkedin.com/posts/vladimir-dimitrov-4460b8242_%D0%B4%D0%B5%D0%B4%D1%83%D0%BF%D0%BB%D0%B8%D0%BA%D0%B0%D1%86%D0%B8%D1%8F-%D0%BE%D0%B1%D1%8A%D1%8F%D0%B2%D0%BB%D0%B5%D0%BD%D0%B8%D0%B8-ugcPost-7325052758502293504-iEWG/)
* [Строим поисковый движок](https://youtu.be/oMQjl8NBFuE) (Дмитрий Емец из VK очень подробно и по шагам)
* [Elasticsearch at SberMegaMarket](https://habr.com/ru/companies/sbermegamarket/articles/688216/)
* [Build search engine python](https://www.deepset.ai/blog/how-to-build-a-semantic-search-engine-in-python)
* [Semantic-search-with-s-bert-is-all-you-need](https://medium.com/mlearning-ai/semantic-search-with-s-bert-is-all-you-need-951bc710e160)
* [Bert Topic](https://medium.com/towards-data-science/bertopic-what-is-so-special-about-v0-16-64d5eb3783d9)
* [building-a-qa-semantic-search-engine-in-3-minutes](https://github.com/hanxiao/bert-as-service/blob/master/README.md#building-a-qa-semantic-search-engine-in-3-minutes) + [building-a-semantic-search-engine-using-open-source-components](https://blog.onebar.io/building-a-semantic-search-engine-using-open-source-components-e15af5ed7885)
* [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over](https://arxiv.org/abs/2004.12832)
* [SBert for semantic search](https://www.linkedin.com/posts/maxbuckley_from-65-hours-to-5-seconds-how-sbert-empowered-activity-7390810937206788096-3nS4)
* [Bert: two encoder approach](https://www.linkedin.com/posts/maxbuckley_one-encoder-or-two-sentence-bert-sbert-activity-7394835704855293953-BBVP)
* [BERT fine-tuning](https://www.cloudskillsboost.google/course_sessions/3623482/video/377867)

GPT

* [ChatGPT as recommender](https://arxiv.org/pdf/2304.10149.pdf)
* [Exploring ChatGPT’s Ability to Rank](https://arxiv.org/pdf/2303.07610.pdf)
* [Open search: hybrid model (tags + embeddings)](https://towardsdatascience.com/text-search-vs-vector-search-better-together-3bd48eb6132a)
* [hybrid search](https://medium.com/towards-data-science/how-to-use-hybrid-search-for-better-llm-rag-retrieval-032f66810ebe)
* [Hybrid Search: SPLADE (Sparse Encoder)](https://medium.com/@sowmiyajaganathan/hybrid-search-splade-sparse-encoder-neural-retrieval-models-d092e5f46913)
* [learn-to-rank-with-opensearch-and-metarank](https://medium.com/metarank/learn-to-rank-with-opensearch-and-metarank-3557fa70f8e8)
* [docs.metarank.ai/introduction/quickstart](https://docs.metarank.ai/introduction/quickstart)
* [efficient-semantic-search-over-unstructured-text-in-neo4j](http://efficient-semantic-search-over-unstructured-text-in-neo4j/)
*    [Weaviate + Distilbert search](https://towardsdatascience.com/a-sub-50ms-neural-search-with-distilbert-and-weaviate-4857ae390154)
* [search-rank-and-recommendations](https://medium.com/mlearning-ai/search-rank-and-recommendations-35cc717772cb)
* [neural embeddings in search](https://medium.com/@masoumzadeh/using-neural-embedding-approaches-in-e-commerce-search-engines-df1c119972ca)
* [Fast search engine](https://github.com/unum-cloud/usearch)
* [single-vector-embeddings-are-so-2022-theres-activity-7309963894104469504-fMmO](https://www.linkedin.com/posts/dannyjameswilliams_single-vector-embeddings-are-so-2022-theres-activity-7309963894104469504-fMmO)
* [netflixs-federated-graph](https://netflixtechblog.com/reverse-searching-netflixs-federated-graph-222ac5d23576)
* [pinterest search relevance](https://medium.com/pinterest-engineering/improving-pinterest-search-relevance-using-large-language-models-4cd938d4e892)
* [fine-tuning-text-embeddings](https://shawhin.medium.com/fine-tuning-text-embeddings-f913b882b11c)
* [emerging travel search](https://blog.emergingtravel.com/data-science-in-travel-tech-search-and-booking/)
* [Redis HNSW](https://www.linkedin.com/pulse/how-hierarchical-navigable-small-world-hnsw-algorithms-can-improve-8k7xc)
* [VectorDB](https://sarthakai.substack.com/p/a-vectordb-doesnt-actually-work-the)
* [Gemini embedding model](https://www.linkedin.com/posts/philipp-schmid-a6a2bb196_how-to-use-the-new-gemini-embedding-model-activity-7354039939996524545-LJck)