import os

# import polars as pl
import pandas as pd
import numpy as np

import mlflow
import numpy as np
from sklearn.metrics import roc_auc_score
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from catboost import Pool


from bidmachine.utils import get_model, get_valuable_columns


mlflow.set_tracking_uri("http://mlflow_container_ui:8000")
print('MLFlow connected successfully')
experiment_name = "catboost-params"
try:
    mlflow.create_experiment(experiment_name, artifact_location="s3://mlflow")
except mlflow.exceptions.RestException as e:
    print(e)
mlflow.set_experiment(experiment_name)

def run_optimization(train_pool, y_true, num_trials: int):
    def objective(params):
        with mlflow.start_run():
            mlflow.log_params(params)
            model = get_model(model_params=params)
            model.fit(train_pool)
            predicted_scores = model.predict_proba(train_pool)
            preds = predicted_scores[:, 1]
            metric = roc_auc_score(y_true, preds)
            mlflow.log_metric("roc_auc", metric)
        return {'loss': metric, 'status': STATUS_OK}

    search_space = {
        'iterations': hp.quniform('iterations', 100, 500, 50),
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
        'depth': hp.quniform('depth', 4, 10, 1),
        'l2_leaf_reg': hp.uniform('l2_leaf_reg', 1, 10)
    }

    rstate = np.random.default_rng(42)  # for reproducible results
    fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=num_trials,
        trials=Trials(),
        rstate=rstate
    )


if __name__ == '__main__':
    print('Reading datasets...')
    root_data_dir = os.environ['DATA_DIR']
    train_data_dir = os.path.join(root_data_dir, 'bidmachine_task_data')
    train_df = pd.read_csv(os.path.join(train_data_dir, 'test_data.csv'), nrows=1000)
    print(train_df.shape[0])
    columns_subset = get_valuable_columns(train_df)
    cat_candidates = ['request_context_device_type', 'dsp', 'ssp', 'hour']
    features = {
        'cat': [col for col in cat_candidates if col in columns_subset],
        'num': ['price', ]
    }
    features_set = features['cat'] + features['num']
    X = train_df[features_set]
    y = train_df['target']
    train_pool = Pool(data=X, label=y, cat_features=features['cat'])
    if train_pool is not None:
        print('training model...')
        y_true = y
        run_optimization(train_pool, y_true, num_trials=12)
