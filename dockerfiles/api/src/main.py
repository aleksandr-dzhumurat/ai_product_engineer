"""Customer segmentation API and RTB scaffold for the MLOps assignment.

Implement the TODOs: load the model once at startup and add inference to /predict.
"""

import os
import pickle
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import numpy as np
import requests
import yaml
from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field, StringConstraints
from ml_tools.utils import AuthConfig, search

logger = logging.getLogger(__name__)

MODEL_PATH = Path(os.getenv("DATA_DIR", "/srv/data")) / "model_dockerized.cb"
SEGMENTATION_MODEL_PATH = (
    Path(os.getenv("DATA_DIR", "/srv/data")) / "pipelines-data" / "segmentation_model.pkl"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.classifier_model = None
    if SEGMENTATION_MODEL_PATH.is_file():
        try:
            with SEGMENTATION_MODEL_PATH.open("rb") as model_file:
                app.state.classifier_model = pickle.load(model_file)
        except Exception:
            logger.warning("Segmentation model could not be loaded; /classifier/ is unavailable")
    config_path = Path(os.getenv("CONFIG_DIR", "configs")) / "index_config.yml"
    config = yaml.safe_load(config_path.read_text()) if config_path.is_file() else {}
    app.state.search_index = os.getenv("SEARCH_INDEX") or config.get("elastic_index_settings", {}).get("name", "index12")
    app.state.search_auth = AuthConfig(
        ZINCSEARCH_URL=os.getenv("ZINCSEARCH_URL", "http://localhost:4080"),
        USERNAME=os.getenv("ZINCSEARCH_USERNAME"),
        PASSWORD=os.getenv("ZINCSEARCH_PASSWORD"),
    )
    app.state.model = None
    # TODO: load CatBoostClassifier from MODEL_PATH once at startup.
    # Decide how startup should fail if the model file is missing or invalid.
    yield


app = FastAPI(title="ML model API: Docker and MLOps practicals", lifespan=lifespan)


class SearchRequest(BaseModel):
    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
    num: int = Field(default=5, ge=1, le=20)


class SearchResult(BaseModel):
    content: str
    asin: str


@app.post("/search", response_model=list[SearchResult])
def search_reviews(payload: SearchRequest, request: Request):
    try:
        result = search(
            request.app.state.search_index, payload.text, limit=payload.num,
            conf=request.app.state.search_auth, raise_on_error=True,
        )
    except (requests.ConnectionError, requests.Timeout):
        raise HTTPException(status_code=503, detail="Search temporarily unavailable") from None
    except requests.HTTPError as exc:
        status = 503 if exc.response is not None and exc.response.status_code >= 500 else 502
        raise HTTPException(status_code=status, detail="Search engine request failed") from None
    except (requests.RequestException, ValueError):
        raise HTTPException(status_code=502, detail="Invalid search engine response") from None
    try:
        hits = result["hits"]["hits"]
        if not isinstance(hits, list):
            raise ValueError("Expected a list of hits")
        results = []
        for hit in hits[:payload.num]:
            source = hit["_source"]
            content, asin = source["content"], source["asin"]
            if not isinstance(content, str) or not isinstance(asin, str):
                raise ValueError("Invalid document fields")
            results.append(SearchResult(content=content[:1000], asin=asin))
        return results
    except (KeyError, TypeError, ValueError):
        raise HTTPException(status_code=502, detail="Invalid search engine response") from None


class PredictRequest(BaseModel):
    request_context_device_type: str | int
    dsp: str | int
    ssp: str | int
    hour: int = Field(ge=0, le=23)
    price: float = Field(allow_inf_nan=False)


class PredictResponse(BaseModel):
    prediction: int = Field(ge=0, le=1)
    probability: float = Field(ge=0, le=1, description="Probability of target=1")


@app.get("/ping")
def ping():
    """Liveness only: this does not mean the model is ready for inference."""
    return {"message": "pong"}


@app.get("/classifier/")
def classify(
    request: Request,
    x1: float = Query(..., description="Feature 1: call_diff", allow_inf_nan=False),
    x2: float = Query(..., description="Feature 2: sms_diff", allow_inf_nan=False),
    x3: float = Query(..., description="Feature 3: traffic_diff", allow_inf_nan=False),
):
    """Predict customer segmentation using the model from the Docker lesson."""
    model = request.app.state.classifier_model
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Segmentation model is missing. Run make run-train and restart the API.",
        )
    features = np.array([[x1, x2, x3]])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    return {
        "prediction": int(prediction),
        "features": {
            "x1_call_diff": x1,
            "x2_sms_diff": x2,
            "x3_traffic_diff": x3,
        },
        "probabilities": {
            f"class_{i}": float(probability)
            for i, probability in enumerate(probabilities)
        },
    }


@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={501: {"description": "Inference has not been implemented yet"}},
)
def predict(payload: PredictRequest, request: Request):
    """Implement model inference here as part of the practical assignment."""
    # TODO: obtain the loaded model from request.app.state.model.
    # TODO: build a single-row input using model.feature_names_ in training order.
    # Some categorical candidates may have been excluded by get_valuable_columns.
    # Preserve the categorical types and preprocessing used during training.
    # TODO: return the class and the probability of target=1 as Python int/float.
    raise HTTPException(status_code=501, detail="Implement model inference in /predict")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
