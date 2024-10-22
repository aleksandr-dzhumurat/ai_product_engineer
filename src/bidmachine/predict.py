def get_model(model_path=None, mode='ranker'):
    if mode == 'ranker':
        model = CatBoostRanker(**{
            'loss_function': 'YetiRank', 'boosting_type': 'Plain',
            'bootstrap_type': 'Bernoulli', 'n_estimators': 400
        })
    elif mode == 'classifier':
        model = CatBoostClassifier(**{
            'loss_function': 'CrossEntropy', 'boosting_type': 'Plain',
            'bootstrap_type': 'Bernoulli', 'n_estimators': 150
        })
    else:
        model = CatBoostClassifier(**{
            'loss_function': 'CrossEntropy', 'boosting_type': 'Plain',
                'bootstrap_type': 'Bernoulli', 'n_estimators': 150
        })
    if model_path is not None:
        model.load_model(model_path)
    return model

def train_model(train_df, features, feature_importances_file_path, model_path, excluded_columns=None, is_train=True, model=None):
    """
    features = get_features(os.path.join(root_data_dir, 'models', 'features.json'))
    feature_importances_file_path = os.path.join(root_data_dir, 'models', 'feature_importance.csv')
    model_path = os.path.join(root_data_dir, 'models', 'catboost_model.cbm')
    """

    print('trainng model...')
    if model is None:
        model = get_model(model_path=None, mode='classifier')
    train_pool, features = prepare_data(train_df, features, excluded_columns, is_train=is_train)
    model.fit(train_pool)

    feature_importances_df = dump_feature_importances(model, train_pool, features, feature_importances_file_path)

    model.save_model(model_path)
    print(f'model saved to {model_path}')
    return model, feature_importances_df

if __name__ == '__main__':
    pass