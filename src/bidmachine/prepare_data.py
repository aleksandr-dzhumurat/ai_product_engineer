import os
import zipfile
import json

# import polars as pl

def extract_zip(root_dataset_dir: str):
    print(f'loading from {root_dataset_dir}')
    zip_file_path = os.path.join(root_dataset_dir, 'bidmachine_logs.zip')
    extraction_dir = os.path.join(root_dataset_dir, 'bidmachine_task_data')
    if not os.path.exists(extraction_dir):
        os.makedirs(extraction_dir)
    if len(os.listdir(extraction_dir)) == 0:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extraction_dir)
    print(f'All files have been extracted to {extraction_dir}')
    return extraction_dir

if __name__ == '__main__':
    root_dataset_dir = os.environ['ROOT_DATA_DIR']
    extract_zip(root_dataset_dir)
