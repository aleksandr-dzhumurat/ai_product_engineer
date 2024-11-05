import os

from catboost import Pool, CatBoostClassifier

from IPython.display import clear_output
import pandas as pd

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

if __name__ == '__main__':
    print('Train started')
    root_data_dir = os.environ['DATA_DIR']
    train_data_dir = os.path.join(root_data_dir, 'bidmachine_task_data')
    train_df = pd.read_csv(os.path.join(train_data_dir, 'train_data.csv'), nrows=1000)
    print(train_df.shape[0])
    columns_subset = get_valuable_columns(train_df)
    cat_candidates = ['request_context_device_type', 'dsp', 'ssp', 'hour']
    features = {
        'cat': [col for col in cat_candidates if col in columns_subset],
        'num': ['price', ]
    }
    model = CatBoostClassifier(**{
        'loss_function': 'CrossEntropy', 'boosting_type': 'Plain',
        'bootstrap_type': 'Bernoulli', 'n_estimators': 150
    })
    features_set = features['cat'] + features['num']
    X = train_df[features_set]
    y = train_df['target']
    train_pool = Pool(data=X, label=y, cat_features=features['cat'])
    model.fit(train_pool)
    clear_output()
    print(f'Model trained: {model}')
    model_path = os.path.join(root_data_dir, 'model_dockerized.cb')
    model.save_model(model_path)
    print(f'model saved to {model_path}')