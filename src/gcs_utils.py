import os

from google.cloud import storage

def download_cs_file(bucket_name, file_name, destination_file_name): 
    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    blob = bucket.blob(file_name)
    blob.download_to_filename(destination_file_name)

    return True

def extract_zip_archive(zip_file_path, root_data_dir):
    import zipfile

    extracted_dir = os.path.join(root_data_dir, 'ml_100k')
    
    os.makedirs(extracted_dir, exist_ok=True)

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_dir)
    print('Data extracted to')

def train_catboost(data):
    import pandas as pd
    from sklearn.model_selection import train_test_split


    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    from catboost import CatBoost, CatBoostClassifier
    from catboost import Pool

    # Set up the CatBoost model
    model = CatBoostClassifier(
        iterations=100,  # Adjust the number of iterations as needed
        learning_rate=0.1,
        loss_function='MultiClass',  # Pairwise ranking loss function
        verbose=100  # Print progress every 100 iterations
    )

    # Define the features and target columns
    features = ['user_id', 'item_id']
    target = 'rating'

    # Create CatBoost pools
    train_pool = Pool(data=train_data[features], label=train_data[target])
    test_pool = Pool(data=test_data[features], label=test_data[target])

    # Train the model
    model.fit(train_pool, eval_set=test_pool)

    from sklearn.metrics import average_precision_score

    # Get predicted ratings for the test data
    test_predictions = model.predict(test_pool)

    # Calculate average precision score
    ap_score = average_precision_score(test_data[target], test_predictions)
    print(f"Average Precision Score on Test Data: {ap_score:.4f}")

    return model

def load_model_to_gcs(bucket_name, source_file_name, destination_file_name):
    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    blob = bucket.blob(destination_file_name)
    blob.upload_from_filename(source_file_name)

    return True


def prepare_data(root_dir, output_file_name: str):
    import pandas as pd

    if not os.path.exists('/srv/data'):
        os.mkdir('/srv/data')
    gcs_bucket = os.environ['GCS_BUCKET']
    gcs_file_path = os.environ['GCS_DATA_PATH']
    result_filename = os.path.join(root_dir, output_file_name)
    print('Data loading to %s' % result_filename)
    if not os.path.exists(result_filename):
        download_cs_file(
            bucket_name=gcs_bucket, file_name=gcs_file_path,
            destination_file_name=result_filename
        )
    if not os.path.exists(os.path.join(root_dir, 'ml_100k')):
        extract_zip_archive(result_filename, root_dir)
        print("Writing to GCS")             
        train_df.to_csv("gs://geo-recommendations-store/train_data.csv.gzip", compression='gzip')
    columns_name=['user_id','item_id','rating','timestamp']
    train_df = pd.read_csv(os.path.join(root_dir, 'ml_100k', 'ml-100k', "u.data") ,sep="\t",names=columns_name)
    model = train_catboost(train_df)

    with open(os.path.join(root_dir, 'model.pkl'), 'wb') as f:
        import pickle

        pickle.dump(model, f)
    print('Model dumped to pickle')
    load_model_to_gcs(gcs_bucket, os.path.join(root_dir, 'model.pkl'), 'model.pkl')
    print("Data saved to GCS")

if __name__ == '__main__':
    mode = os.getenv('mode')
    root_dir = '/srv/data'
    if mode == 'train':
        prepare_data(root_dir, 'ml_100k.zip')