"""
Batch inference script for EventAlly dataset.

Flow:
  1. Start batch inference on an already-uploaded dataset
  2. Poll until done

Required env vars:
  BATCH_API_KEY   - API key for the inference platform
  BATCH_BASE_URL  - Base URL, e.g. https://api.example.com

Usage:
  python src/nebius_batch_llm_inference/batch_inference.py --dataset DATASET_ID --version VERSION_ID
"""

import argparse
import json
import os
import time

import requests
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_URL = os.environ["BATCH_BASE_URL"].rstrip("/")
HEADERS = {
    "Authorization": f"Bearer {os.environ['BATCH_API_KEY']}",
    "Content-Type": "application/json",
}

# https://tokenfactory.nebius.com/models
# MODEL = "openai/gpt-oss-20b"
MODEL = "google/gemma-2-9b-it-fast"
COMPLETION_WINDOW = "1h"
POLL_INTERVAL = 5  # seconds


# ---------------------------------------------------------------------------
# Step 1: Start batch inference
# ---------------------------------------------------------------------------

def start_batch_inference(dataset_id: str, version_id: str) -> tuple[str, str]:
    src_entry = {
        "id": dataset_id,
        "version": version_id,
        "mapping": {
            "type": "text_messages",
            "messages": {
                "type": "column",
                "name": "prompt",
            },
            "custom_id": {
                "type": "column",
                "name": "custom_id",
            },
            "max_tokens": {
                "type": "text",
                "value": "3500",
            },
        },
    }

    response = requests.post(
        f"{BASE_URL}/v1/operations",
        json={
            "type": "batch_inference",
            "src": [src_entry],
            "dst": [],
            "params": {
                "model": MODEL,
                "completion_window": COMPLETION_WINDOW,
            },
        },
        headers=HEADERS,
    )
    response.raise_for_status()
    data = response.json()
    operation_id = data["id"]
    dst_dataset_id = data["dst"][0]["id"]
    print(f"Batch inference started: operation_id={operation_id}, dst_dataset_id={dst_dataset_id}")
    return operation_id, dst_dataset_id


# ---------------------------------------------------------------------------
# Step 2: Poll until done
# ---------------------------------------------------------------------------

def wait_for_completion(operation_id: str) -> None:
    print("Waiting for operation to complete...")
    while True:
        status_response = requests.get(
            f"{BASE_URL}/v1/operations/{operation_id}",
            headers=HEADERS,
        )
        status_response.raise_for_status()
        status_data = status_response.json()
        status = status_data["status"]
        print(f"  Operation status: {status}")
        if status not in {"queued", "running"}:
            break
        time.sleep(POLL_INTERVAL)

    if status != "completed":
        raise RuntimeError(f"Operation ended with status: {status}\n{json.dumps(status_data, indent=2)}")
    print("Operation completed successfully.")
    usage = status_data.get("usage") or status_data.get("stats") or status_data.get("metrics")
    if usage:
        print(f"  Token usage: {json.dumps(usage, indent=4)}")
    else:
        print(f"  Full response (look for token fields): {json.dumps(status_data, indent=2)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run batch translation inference on an uploaded dataset.")
    parser.add_argument("--dataset", required=True, help="Source dataset ID")
    parser.add_argument("--version", required=True, help="Dataset version ID")
    args = parser.parse_args()

    print(f"=== Starting batch inference for dataset={args.dataset} ===")
    operation_id, dst_dataset_id = start_batch_inference(args.dataset, args.version)

    print("\n=== Waiting for completion ===")
    wait_for_completion(operation_id)

    print(f"\nDone. Destination dataset ID: {dst_dataset_id}")
    print(f"python3 src/nebius_batch_llm_inference/count_tokens.py $(pwd)/data/{dst_dataset_id}.jsonl")


if __name__ == "__main__":
    main()
