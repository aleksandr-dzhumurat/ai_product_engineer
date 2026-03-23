# CatBoost training tips & tricks

## Training speedup

Tips & tricks for [speedup training](https://catboost.ai/en/docs/concepts/speed-up-training)

- [Boosting type](https://catboost.ai/en/docs/concepts/speed-up-training#boosting-type): `Ordered` (for small datasets) or `Plain`  (for other cases). [Docs](https://catboost.ai/en/docs/references/training-parameters/common#boosting_type)
- [Loss_function](https://catboost.ai/en/docs/references/training-parameters/common#loss_function):
    - [Classification](https://catboost.ai/en/docs/concepts/loss-functions-classification)
        - `CrossEntropy` (fastest on big dataset)
        - `LogLoss`
        - Other loss functions don`t have [GPU support](https://catboost.ai/en/docs/concepts/loss-functions-classification#usage-information)
- [Bootstrap type](https://catboost.ai/en/docs/concepts/algorithm-main-stages_bootstrap-options)
    - `Bernoulli` is the fastest
    - some of them supported on GPU
- [max_ctr_complexity](https://catboost.ai/en/docs/references/training-parameters/ctr#max_ctr_complexity) should be set at `1`

## How to train Catboost

For classification task we choose `CrossEntropy` as a fastest loss

```python
train_df = prepare_train_df(source_df, train_fraction, feature_set)
    label_series = train_df['target'].astype(np.int32)

    logging.info('Train started...')
    train_pool = Pool(
        data=train_df[feature_set],
        label=label_series,
        cat_features=cat_features,
        group_id=train_df['order_id'],
    )
    estimator = CatBoost({
        'loss_function': 'CrossEntropy', 'boosting_type': 'Plain',
				'bootstrap_type': 'Bernoulli', 'n_estimators': 500
    })
    estimator.fit(train_pool)
```

In `prepare_df` you can implement some pre-processing. For example some [stratified sampling](https://www.geeksforgeeks.org/stratified-sampling-in-pandas/) (if you want to decrease RAM consumption).

```python
train_fraction = 60  # in percents
logging.info('Before sampling %d', train_df.shape[0])
sample_fraction = train_fraction / 100
train_df = train_df.groupby('city_id', group_keys=False).apply(lambda x: x.sample(frac=sample_fraction))
logging.info('After sampling %d', train_df.shape[0])
```