import json
import gzip
import os
import re
from dataclasses import dataclass

import yaml
import requests
import numpy as np
import pandas as pd

run_env = os.getenv('RUN_ENV', 'COLLAB')
if run_env == 'LOCAL':
    from dotenv import load_dotenv
    print('.env loaded: ', load_dotenv(dotenv_path='./../.env'))
ZINCSEARCH_URL = os.getenv("ZINCSEARCH_URL")
USERNAME=os.getenv('ZINCSEARCH_USERNAME')
PASSWORD=os.getenv('ZINCSEARCH_PASSWORD')


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

def request_elastic(api_endpoint, debug=False, method='get', data=None):
    url = f"{ZINCSEARCH_URL}/{api_endpoint}"
    if data is None:
        data = {}
    headers = {"Content-Type": "application/json"}
    request_method = getattr(requests, method.lower())
    if debug:
        print(url, json.dumps(data))
    response = request_method(url, headers=headers, data=json.dumps(data), auth=(USERNAME, PASSWORD))
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

def create_index(index_config: dict, debug=False):
    response = request_elastic('api/index', debug=debug, data=index_config, method='post')
    return response

def load_document(doc, index_name, debug=False):
    response = request_elastic(f'api/{index_name}/_doc', debug=debug, data=doc, method='post')
    return response

def load_bulk_documents(index_name, documents):
    for doc in documents:
        load_document(doc, index_name)
    print("Document loaded successfully.")

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
        if limit is None and not 'jsonl' in file_path:
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

def search(index_name, query=None, fields_list=None, category='health', debug=False, limit=10):
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
    response = request_elastic(f'es/{index_name}/_search', debug=debug, data=search_query, method='post')
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

