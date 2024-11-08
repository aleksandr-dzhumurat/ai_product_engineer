import os
import sys

import pandas as pd
import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

from utils import get_config, train_and_save_model

HPO_EXPERIMENT_NAME = "catboost-params-new"
# mlflow.create_experiment(HPO_EXPERIMENT_NAME, artifact_location="s3://mlflow")
RF_PARAMS = ['max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'random_state']

# mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_tracking_uri("http://mlflow_container_ui:8000")
mlflow.set_experiment(HPO_EXPERIMENT_NAME)


def train_best_model(train_data_path, config, model_path):
    top_n=5
    client = MlflowClient()
    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.test_rmse ASC"]
    )[0]
    run_id = run.info.run_id
    print(run.data.params, run.info.run_id)
    with mlflow.start_run(run_id=run_id):
        train_df = pd.read_csv(train_data_path)
        train_and_save_model(train_df, config, model_path)
        # mlflow.log_artifact(model_path)  ## TODO: fix the bug
        model_uri = f"runs:/{run_id}/model"
        mlflow.register_model(model_uri, name="rf-best-model")
    print('model logged and registered')

if __name__ == '__main__':
    config = get_config(sys.argv[1])
    root_data_dir = '/srv/data/'
    model_path = os.path.join('/srv/data', config['model_file_name'])
    train_data_path = os.path.join(root_data_dir, 'train_dataset.csv')

    # train_model(train_data_path, config, model_path)
    train_best_model(train_data_path, config, model_path)