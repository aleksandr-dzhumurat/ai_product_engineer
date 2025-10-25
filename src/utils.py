import gzip
import json
import logging
import os
import re
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests
import yaml

run_env = os.getenv('RUN_ENV', 'COLLAB')

from pathlib import Path

from dotenv import load_dotenv


@dataclass
class AuthConfig:
    """Authentication configuration for ZincSearch"""
    ZINCSEARCH_URL: str
    USERNAME: str
    PASSWORD: str


def get_auth(env_path: str | Path) -> AuthConfig:
    """
    Load authentication configuration from environment file.
    
    Args:
        env_path: Path to the .env file
        
    Returns:
        AuthConfig dataclass with ZINCSEARCH_URL, USERNAME, and PASSWORD
    """
    print(f'ENV loaded from {env_path}: {load_dotenv(env_path)}')
    
    return AuthConfig(
        ZINCSEARCH_URL=os.environ.get('ZINCSEARCH_URL'),
        USERNAME=os.environ.get('ZINCSEARCH_USERNAME'),
        PASSWORD=os.environ.get('ZINCSEARCH_PASSWORD')
    )


def get_config(config_path: str | Path) -> dict:
    """
    Load configuration from JSON or YAML file.

    Args:
        config_path: Path to the config file (.json or .yml/.yaml)

    Returns:
        Dictionary containing configuration parameters
    """
    config_path = Path(config_path)

    if config_path.suffix == '.json':
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    elif config_path.suffix in ['.yml', '.yaml']:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported config file format: {config_path.suffix}. Use .json, .yml, or .yaml")

    print(f'Config loaded from {config_path}')
    return config


@dataclass
class Dataset:
    data: np.array
    target: np.array
    DESCR: str

def load_sales(root_data_dir):
  file_path = os.path.join(root_data_dir, 'sales_timeseries_dataset.csv.gz')
  df = pd.read_csv(file_path, compression='gzip')
  feature_cols = [
        'QTy', 'NetSales_Previous_Day',
       'Avg_Sale_Previous_Week', 'Sales_This_Day_Last_Week',
       'Sales_This_Day_Week_Before_Last',
       'Diff_Sales_This_Day_Last_Week_And_Week_Before_Last'
  ]
  target_col = 'NetSales'
  sales = Dataset(data=df[feature_cols].values, target=df[target_col].values, DESCR=', '.join(feature_cols))
  return sales

def request_elastic(api_endpoint, debug=False, method='get', data=None, conf=None):
    url = f"{conf.ZINCSEARCH_URL}/{api_endpoint}"
    if data is None:
        data = {}
    headers = {"Content-Type": "application/json"}
    request_method = getattr(requests, method.lower())
    if debug:
        print(url, json.dumps(data))
    response = request_method(url, headers=headers, data=json.dumps(data), auth=(conf.USERNAME, conf.PASSWORD))
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to request: {response.status_code}, {response.text}")

def load_config():
    conf_dir = os.environ['CONFIG_DIR']
    config_path = os.path.join(conf_dir,'index_config.yml')
    config = {}
    with open(config_path, "r", encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    index_config = config['elastic_index_settings']
    return index_config

def create_index(index_config: dict, debug=False, conf=None):
    response = request_elastic('api/index', debug=debug, data=index_config, method='post', conf=conf)
    return response

def load_document(doc, index_name, debug=False, conf=None):
    response = request_elastic(f'api/{index_name}/_doc', debug=debug, data=doc, method='post', conf=conf)
    return response

def load_bulk_documents(index_name, documents, conf):
    errors = []
    for doc in documents:
        try:
            load_document(doc, index_name, conf=conf)
        except Exception:
            errors.append({'status': 'error', 'doc': doc})
    print(f"Document loaded successfully, num errors: {len(errors)}")

def clean_text(txt):
    cleaned_text = re.sub(r'\n+', ' ', txt)
    cleaned_text = re.sub(r'\t+', ' ', cleaned_text)
    cleaned_text = re.sub(r"<.*?>", "", cleaned_text)
    return cleaned_text

def save_json_gzip(input_path, output_path):
    with open(input_path, 'r', encoding='utf8') as f_in:
        with gzip.open(output_path, 'wt', encoding='UTF-8') as f_out:
            data = json.load(f_in)
            json.dump(data, f_out)
    print(f'Saved to {output_path}')

def read_raw_data(file_path, limit: int = None, fields = None):
    with gzip.open(file_path, 'rt', encoding='UTF-8') as gz_file:
        if limit is None and 'jsonl' not in file_path:
           res = json.load(gz_file)
        else:
            res = []
            for line in gz_file:
                data = json.loads(line.strip())
                if fields is not None:
                    res.append({i: j for i, j in data.items() if i in fields})
                else:
                    res.append(data)
                if limit == len(res):
                    break
        print(f'Dataset num items: {len(res)} from {file_path}')
        return res

def search(index_name, query=None, fields_list=None, category='health', debug=False, limit=10, conf=None):
    if fields_list is None:
        fields_list = ['content']
    if query is not None:
        search_query = {
            "query": {
                "bool": { 
                    "must": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": fields_list,
                                "type": "best_fields"
                            }
                        },
                    ],
                    "should": [],
                    "must_not": [],
                    "filter": [ 
                        { "term":  { "category": category }},
                    ]
                }
            },
            "size": limit
        }
    else:
        search_query = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "match_all": {}
                        }
                    ],
                    "should": [],
                    "must_not": [],
                    "filter": [
                        {"term": {"category": category}}
                    ]
                }
            },
            "size": limit
        }
    response = request_elastic(f'es/{index_name}/_search', debug=debug, data=search_query, method='post', conf=conf)
    return response

def pretty(search_results, include_fields=None):
    """
        print(pretty(res))
    """
    result_docs = []
    if include_fields is None:
        include_fields = ['content']
    for hit in search_results:
        result_docs.append({k: f'{clean_text(v)[:150]}...' for k, v in  hit['_source'].items() if k in include_fields})
    return result_docs


def process_row(i, val):
    val = val.replace('\0', '')
    transformed_val = val
    if i == 1:
        transformed_val = hashlib.md5(val.encode('utf-8')).hexdigest()
    elif i == 4:
        transformed_val = int(datetime.datetime.strptime(val, '%Y-%m-%d %H:%M:%S').timestamp())
    if i in (1,5,6,7,8,9,10,15,17, 19):
        text_replaced = transformed_val.replace('"','')
        transformed_val = f'"{text_replaced}"'
    return transformed_val
    
def csv_reader(file_name, sink_file_name):
    """
    Read a csv file
    """
    with open(file_name, 'r') as file_obj:
        reader = csv.reader((line.replace('\0','') for line in file_obj), delimiter=',')
        with open(sink_file_name, 'w', encoding='utf-8') as sink_file_obj:
            sink_file_obj.write(
                f"{','.join([i for i in next(reader)])}\n"
            )
            for row in reader:
                sink_file_obj.write(
                    f"{','.join(map(str, [process_row(i, j) for i, j in enumerate(row)]))}\n"
                )

def main_function():
    home_dir = '/usr/share/data_store/raw_data'
    # file_name = 'csv_example.csv'
    file_name = 'win_users_events2.csv'
    #file_name = 'null.csv'
    sink_file_name = 'events.csv'

    csv_reader(os.path.join(home_dir, file_name), os.path.join(home_dir, sink_file_name))

def setup_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(filename)-24s :%(lineno)-4d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger