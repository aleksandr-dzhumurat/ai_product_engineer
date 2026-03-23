# Memory profiling

What if you code interrupted with Out Of Memory killer? Just use line profiler to find and fix memory leak?

At fist you need to create python environment with this [memory_profiler](https://pypi.org/project/memory-profiler/) package

- requirements.txt
    
    ```python
    memory_profiler==0.61.0
    numpy==1.23
    pandas==1.5.2
    catboost==1.1.1
    ```
    

Then create environment and install packages

```bash
pyenv install 3.8.10 && \
pyenv virtualenv 3.8.10 profiling && \
source ~/.pyenv/versions/profiling/bin/activate
```

Then you just need to add `@profile` decorate to you code

- `catboost_train.py`
    
    ```python
    import logging
    import pickle
    
    import numpy as np
    import pandas as pd
    from catboost import CatBoost, Pool
    from memory_profiler import profile  # profiling module
    
    def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
        bytes_in_megabyte = 1024**2
        start_mem = df.memory_usage().sum() / bytes_in_megabyte
        print(f'Memory usage of dataframe is {np.round(start_mem, 2)} MB')
        for col in df.columns:
            col_type = df[col].dtype
            numerics_types = ('int16', 'int32', 'int64', 'float16', 'float32', 'float64')
            if col_type in numerics_types:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
            else:
                df[col] = df[col].astype(str) # 'category'
    
        end_mem = df.memory_usage().sum() / bytes_in_megabyte
        print(f'Memory usage after optimization is: {np.round(end_mem, 2)} MB')
        print(f'Decreased by {np.round(100 * (start_mem - end_mem) / start_mem)}%')
        return df
    
    @profile
    def train():
    
        with open('columns.pkl', 'rb') as f:
            features = pickle.load(f)
    
        cat_features, feature_set = features['cat_features'], features['final_features']
    
        cols_sub_list = feature_set + ['order_id', 'target', 'city_id']
        print('loading csv...')
        train_df = pd.read_csv('data_train_2023_11', compression='gzip', usecols=cols_sub_list, nrows=10**6)
        traind_df = reduce_mem_usage(train_df)
        train_df.sort_values(by='order_id', inplace=True)
    
        label_series = train_df['target'].astype(np.int32)
    
        print('creating Pool...')
        sorted_cat_features = np.sort(np.array(cat_features))
        sorted_feature_set = np.sort(np.array(feature_set))
        train_pool = Pool(data=train_df[sorted_feature_set].values, label=label_series.values, cat_features=np.searchsorted(sorted_feature_set, sorted_cat_features), group_id=train_df['order_id'].values)
        # train_pool = Pool(data=train_df[sorted_feature_set], label=label_series, cat_features=sorted_cat_features, group_id=train_df['order_id'])
        model_params = {'loss_function': 'Logloss', 'boosting_type': 'Plain', 'bootstrap_type': 'Bernoulli', 'n_estimators': 50, 'used_ram_limit': '8gb', 'max_depth': 4}
        print('Train starting...')
        estimator = CatBoost(model_params)
        estimator.fit(train_pool, verbose=False)
    
        print('Predict started...')
        train_df['y_pred'] = estimator.predict(train_pool)
        train_df['y_pred'] = train_df.groupby('order_id')['y_pred'].rank(method='first', ascending=False)
        print('Train finished')
    
    if __name__ == '__main__':
        train()
    ```
    

Then just run `python catboost_train.py` and enjoy result! You can see how much memory requires pandas DataFrame and Catboost Pool. It will help you to find memory leak – for example optimize pandas DataFrame memory usage (you can see some tricks on [Medium](https://towardsdatascience.com/seven-killer-memory-optimization-techniques-every-pandas-user-should-know-64707348ab20))

```bash
Line #    Mem usage    Increment  Occurrences   Line Contents
=============================================================
    43     77.0 MiB     77.0 MiB           1   @profile
    44                                         def train():
    45                                         
    46     77.0 MiB      0.0 MiB           1       with open('columns.pkl', 'rb') as f:
    47     77.1 MiB      0.0 MiB           1           features = pickle.load(f)
    48                                         
    49     77.1 MiB      0.0 MiB           1       cat_features, feature_set = features['cat_features'], features['final_features']
    50                                         
    51     77.1 MiB      0.0 MiB           1       cols_sub_list = feature_set + ['order_id', 'target', 'city_id']
    52     77.1 MiB      0.0 MiB           1       print('loading csv...')
    53   1693.0 MiB   1615.9 MiB           1       train_df = pd.read_csv('data_train_2023_11', compression='gzip', usecols=cols_sub_list, nrows=10**6)
    54   1530.2 MiB   -162.7 MiB           1       traind_df = reduce_mem_usage(train_df)
    55   1761.7 MiB    231.4 MiB           1       train_df.sort_values(by='order_id', inplace=True)
    56                                         
    57   1761.7 MiB      0.0 MiB           1       label_series = train_df['target'].astype(np.int32)
    58                                         
    59   1761.7 MiB      0.0 MiB           1       print('creating Pool...')
    60   1761.7 MiB      0.0 MiB           1       sorted_cat_features = np.sort(np.array(cat_features))
    61   1761.7 MiB      0.0 MiB           1       sorted_feature_set = np.sort(np.array(feature_set))
    62    331.5 MiB  -1430.2 MiB           1       train_pool = Pool(data=train_df[sorted_feature_set].values, label=label_series.values, cat_features=np.searchsorted(sorted_feature_set, sorted_cat_features), group_id=train_df['order_id'].values)
    63                                             # train_pool = Pool(data=train_df[sorted_feature_set], label=label_series, cat_features=sorted_cat_features, group_id=train_df['order_id'])
    64    331.6 MiB      0.0 MiB           1       model_params = {'loss_function': 'Logloss', 'boosting_type': 'Plain', 'bootstrap_type': 'Bernoulli', 'n_estimators': 50, 'used_ram_limit': '8gb', 'max_depth': 4}
    65    331.8 MiB      0.2 MiB           1       print('Train starting...')
    66    332.1 MiB      0.3 MiB           1       estimator = CatBoost(model_params)
    67   1766.7 MiB   1434.6 MiB           1       estimator.fit(train_pool, verbose=False)
    68                                         
    69   1766.7 MiB      0.0 MiB           1       print('Predict started...')
    70   1791.2 MiB     24.6 MiB           1       train_df['y_pred'] = estimator.predict(train_pool)
    71   1900.6 MiB    109.4 MiB           1       train_df['y_pred'] = train_df.groupby('order_id')['y_pred'].rank(method='first', ascending=False)
    72   1900.7 MiB      0.0 MiB           1       print('Train finished')
```