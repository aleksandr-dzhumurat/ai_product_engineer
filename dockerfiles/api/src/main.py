from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

from ml_tools.utils import search, load_config, pretty


app = FastAPI()
index_config = load_config()
index_name = index_config['name']

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
    # 'cough'
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
