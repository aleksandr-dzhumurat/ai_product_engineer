

from catboost import CatBoostClassifier


def get_model(model_params=None):
    if model_params is None:
        model_params = {
            'loss_function': 'CrossEntropy', 'boosting_type': 'Plain',
            'bootstrap_type': 'Bernoulli', 'n_estimators': 150
        }
    model = CatBoostClassifier(**model_params)
    return model

def get_valuable_columns(input_df):
    col_subset = []
    for col in input_df.columns:
        try:
            most_frequent_value_count = input_df[col].value_counts().iloc[0]
            total_rows = len(input_df)
            if most_frequent_value_count / total_rows <= 0.95:
                col_subset.append(col)
        except (TypeError, IndexError):
            # Ignore columns that can't be analyzed (e.g. mixed data types)
            pass
    return col_subset
