"""
Prepare and upload eventally_dataset_cloud.jsonl to the Nebius platform.

Run with:
    python3 src/nebius_batch_llm_inference/upload_dataset.py

Transforms columns:
  link         -> custom_id
  cleared_text -> prompt  (JSON text_messages format)

Prints dataset_id and version_id on success.

Required env vars:
  BATCH_API_KEY   - API key for the inference platform
  BATCH_BASE_URL  - Base URL, e.g. https://api.tokenfactory.nebius.com
"""

import json
import os

import requests
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path)

BASE_URL = os.environ["BATCH_BASE_URL"].rstrip("/")
FOLDER_ID = os.environ["BATCH_FOLDER_ID"]
HEADERS = {
    "Authorization": f"Bearer {os.environ['BATCH_API_KEY']}",
    "Content-Type": "application/json",
}

INPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "eventally_dataset", "eventally_dataset_cloud.jsonl")

SCHEMA = [
    {"name": "custom_id", "type": {"name": "string"}},
    {"name": "prompt",    "type": {"name": "json"}},
]


def build_prompt_messages(text: str) -> list:
    return [
        {
            "role": "user",
            "content": f"Translate this Russian text to English. Return only the translation, no explanation:\n\n{text}",
        }
    ]


def prepare(input_path: str) -> list:
    """To do: update prompt with actual task"""
    rows = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows.append({
                "custom_id": row["link"],
                "prompt": build_prompt_messages(row["cleared_text"]),
            })
    return rows


def upload(rows: list) -> tuple[str, str]:
    response = requests.post(
        f"{BASE_URL}/v1/datasets",
        headers=HEADERS,
        json={
            "name": "eventally_batch",
            "folder": FOLDER_ID,
            "schema": SCHEMA,
            "rows": rows,
        },
    )
    if not response.ok:
        print(f"Upload failed {response.status_code}: {response.text}")
        response.raise_for_status()
    data = response.json()
    return data["id"], data["current_version"]


if __name__ == "__main__":
    print("Preparing dataset...")
    rows = prepare(INPUT_PATH)
    print(f"  {len(rows)} rows prepared")

    print("Uploading...")
    dataset_id, version_id = upload(rows)

    print(f"\nDataset uploaded successfully:")
    print(f"  DATASET={dataset_id}")
    print(f"  VERSION={version_id}")
    print(f"\nRun inference with:")
    print(f" python3 src/nebius_batch_llm_inference/batch_inference.py --dataset {dataset_id} --version {version_id}")
