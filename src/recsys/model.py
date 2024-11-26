
from catboost import CatBoostClassifier, Pool

def get_model(params = None):
    if params is None:
        params = {
            'loss_function': 'CrossEntropy', 'boosting_type': 'Plain',
            'bootstrap_type': 'Bernoulli', 'n_estimators': 150
        }
    model = CatBoostClassifier(**params)
    return model

def get_data(input_df, feature_store):

    X = [feature_store[i] for i in input_df['ProductID'].values]
    y = input_df['target'].values

    data_pool = Pool(data=X, label=y)
    return data_pool, y