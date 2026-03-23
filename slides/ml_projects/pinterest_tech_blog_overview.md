# Search

## Architecture

* [Search serving and ranking](https://medium.com/the-graph/search-serving-and-ranking-at-pinterest-224707599c92) (architecture notes)

* [Search architecture](https://medium.com/pinterest-engineering/building-a-universal-search-system-for-pinterest-e4cb03a898d4)

## Models

* [Search suggestions](https://medium.com/pinterest-engineering/evolving-search-recommendations-on-pinterest-136e26e0468a)

* [Stemming approach](https://medium.com/pinterest-engineering/improving-search-relevance-and-engagement-with-text-attributes-452047853967)

* [User history](https://medium.com/pinterest-engineering/building-a-platform-to-understand-search-queries-7138e923c06a) for better search

* [Breaking feedback loop](https://medium.com/pinterest-engineering/query-rewards-building-a-recommendation-feedback-loop-during-query-selection-70b4d20e5ea0) in search

## Content feature extractors for search

*   [BERT embeddings](https://medium.com/pinterest-engineering/searchsage-learning-search-query-representations-at-pinterest-654f2bb887fc) for search

* [Content classifier](https://medium.com/pinterest-engineering/pin2interest-a-scalable-system-for-content-classification-41a586675ee7) (interests taxonomy)

* [Images in search](https://medium.com/pinterest-engineering/hybrid-search-building-a-textual-and-visual-discovery-experience-at-pinterest-8527ba9728a9) (combine textual and visual information)

# Recommendations

## Graph DNN

* [Taste Graph](https://medium.com/pinterest-engineering/taste-graph-part-1-assigning-interests-to-pins-9158b4c25906): pins + users + interests

* [Graph convolutional network](https://medium.com/pinterest-engineering/pinsage-a-new-graph-convolutional-neural-network-for-web-scale-recommender-systems-88795a107f48)

* [GBDT model](https://medium.com/pinterest-engineering/how-machine-learning-significantly-improves-engagement-abroad-98c6ca937f9f) for ranking

Pixie - candidate selection system

* [Pixie](https://medium.com/pinterest-engineering/introducing-pixie-an-advanced-graph-based-recommendation-system-e7b4229b664b): graph-based recommendations framework (for candidate generation)
* [Lightweight ranking with Pixie](https://medium.com/pinterest-engineering/improving-the-quality-of-recommended-pins-with-lightweight-ranking-8ff5477b20e3)
* [GraphQL in Pixie](https://medium.com/pinterest-engineering/an-update-on-pixie-pinterests-recommendation-system-6f273f737e1b)

## Feature extraction

* [Transformer encoder](https://medium.com/pinterest-engineering/how-pinterest-leverages-realtime-user-actions-in-recommendation-to-boost-homefeed-engagement-volume-165ae2e8cde8)

* [Multilabel embeddings](https://medium.com/pinterest-engineering/pintext-a-multitask-text-embedding-system-in-pinterest-b80ece364555)

* [Images: visual complements](https://medium.com/pinterest-engineering/introducing-complete-the-look-a-scene-based-complementary-recommendation-system-eb891c3fe88)

## Ranking models

* [Two-tower architecture for feed ranking](https://medium.com/pinterest-engineering/pinterest-home-feed-unified-lightweight-scoring-a-two-tower-approach-b3143ac70b55)

* [Hierarchical DNN recommendation model](https://medium.com/pinterest-engineering/hiertcn-deep-learning-models-for-dynamic-recommendations-and-inferring-user-interests-a31e8cd4b71e)

* [Look-alike for advertising personalization](https://medium.com/pinterest-engineering/the-machine-learning-behind-delivering-relevant-ads-8987fc5ba1c0)

* [Position bias in CTR optimization](https://medium.com/pinterest-engineering/a-better-clickthrough-rate-how-pinterest-upgraded-everyones-favorite-engagement-metric-27f6fa6cba14) (ads)

* [Contextual relevance](https://medium.com/pinterest-engineering/contextual-relevance-in-ads-ranking-63c2ff215aa2) (ads)

* [Multilabel action prediction](https://medium.com/pinterest-engineering/multi-task-learning-and-calibration-for-utility-based-home-feed-ranking-64087a7bcbad)

## Architecture

* [Linchpin](https://medium.com/pinterest-engineering/the-little-engine-that-could-linchpin-dsl-for-pinterest-ranking-17699add8e56): Map-Reduce engine

* [Gevent](https://medium.com/@Pinterest_Engineering/how-we-use-gevent-to-go-fast-e30fa9f81334) for faster results

## ML for business value

* [ML for more than one country](https://medium.com/pinterest-engineering/personalizing-pinterests-new-user-experience-abroad-60f8f55177ac)

* [Trends](https://medium.com/pinterest-engineering/pinterest-trends-insights-into-unstructured-data-b4dbb2c8fb63)

* [New content type: guides](https://medium.com/pinterest-engineering/a-look-behind-search-guides-74bff56b3398) in search

* [User boards clustering](https://medium.com/pinterest-engineering/using-machine-learning-to-auto-organize-boards-13a12b22bf5)

* [long-term effect from recomendations](https://medium.com/pinterest-engineering/trapped-in-the-present-how-engagement-bias-in-short-run-experiments-can-blind-you-to-long-run-58b55ad3bda0)

* [Flex Budget](https://medium.com/pinterest-engineering/flexible-daily-budgeting-at-pinterest-91fc310c2e33)

* [Multi task learning](https://medium.com/pinterest-engineering/multi-task-learning-for-related-products-recommendations-at-pinterest-62684f631c12)