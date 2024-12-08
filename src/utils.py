import json
import gzip
import os
import re

import yaml
import requests
from dotenv import load_dotenv

print('.env loaded: ', load_dotenv(dotenv_path='./../.env'))
ZINCSEARCH_URL = os.environ["ZINCSEARCH_URL"]
USERNAME=os.environ['ZINCSEARCH_USERNAME']
PASSWORD=os.environ['ZINCSEARCH_PASSWORD']

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
