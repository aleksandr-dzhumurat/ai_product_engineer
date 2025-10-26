import os
import pickle
from typing import List

import numpy as np
from fastapi import FastAPI, Query
from ml_tools.utils import load_config, pretty, search
from pydantic import BaseModel

app = FastAPI()
index_config = load_config()
index_name = index_config['name']

# Load classifier model
root_data_dir = '/srv/data'
output_bin_path = os.path.join(root_data_dir, 'pipelines-data', 'segmentation_model.pkl')
with open(output_bin_path, 'rb') as f:
    classifier_model = pickle.load(f)
    print(f'Model was loaded from {output_bin_path}')

class SearchRequest(BaseModel):
    text: str
    num: int

class SearchResult(BaseModel):
    content: str
    asin: str

@app.post("/search", response_model=List[SearchResult])
async def search_items(request: SearchRequest):
    # Placeholder search logic based on the request
    index_name = index_config['name']
    search_results = search(index_name, request.text, limit=request.num)['hits']['hits']
    print(search_results[0])
    results = pretty(search_results, include_fields = ['content', 'asin'])
    return results

@app.get("/ping")
async def ping():
    """
    Health check endpoint.
    """
    return {"message": "pong"}


@app.get("/classifier/")
async def classify(
    x1: float = Query(..., description="Feature 1: call_diff"),
    x2: float = Query(..., description="Feature 2: sms_diff"),
    x3: float = Query(..., description="Feature 3: traffic_diff")
):
    features = np.array([[x1, x2, x3]])
    prediction = classifier_model.predict(features)[0]
    probabilities = classifier_model.predict_proba(features)[0]
    proba_dict = {f"class_{i}": float(prob) for i, prob in enumerate(probabilities)}
    result = {
        "prediction": int(prediction),
        "features": {
            "x1_call_diff": x1,
            "x2_sms_diff": x2,
            "x3_traffic_diff": x3
        }
    }
    if proba_dict:
        result["probabilities"] = proba_dict

    return result



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
