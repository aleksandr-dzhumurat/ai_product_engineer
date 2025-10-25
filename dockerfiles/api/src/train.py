import os

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

root_data_dir = '/srv/data'
log_file_path = os.path.join(root_data_dir, 'pipelines-data/service.log')

from ml_tools.utils import setup_logger


def train_model(X, y):
    print(f'Num rows: {df_source.shape[0]}')
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X, y)
    return clf

def test_classifier(X, random_id):
    # преобразуем фичи, чтобы подошли на вход модели и делаем предсказания
    test_object = X[random_id,:].reshape(1, -1)
    pred = clf.predict(test_object)[0]
    return pred

if __name__ == '__main__':
    print()
    logger = setup_logger(log_file_path)

    logger.info('You have run train script づ｡◕‿‿◕｡)づ')
    logger.info('Training process started')
    df_source = pd.read_csv(os.path.join(root_data_dir, 'client_segmentation.csv'))

    X = df_source[['call_diff','sms_diff','traffic_diff']].values
    y = df_source['customes_class'].values

    num_objects = X.shape[0]
    random_id = np.random.randint(num_objects)

    clf = train_model(X, y)
    print(f"For object id={random_id} redicted class: {test_classifier(X, random_id)}, actual class={y[random_id]}")

    # Your training code here
    # Example:
    # logger.info('Loading training data')
    # logger.info('Model initialized: %s', model)
    # logger.info('Training completed')