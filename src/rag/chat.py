"""Interactive RAG chat CLI.

Usage:
    DATA_DIR="$(pwd)" python src/rag/chat.py
    DOTENV_FILE="$(pwd)/.env" DATA_DIR="$(pwd)" python src/rag/chat.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from agent.graph import COLLECTION_NAME, VLLM_MODEL, AgentState, _get_db_client, graph
from dotenv import load_dotenv

DOTENV_FILE = os.environ.get("DOTENV_FILE", ".env")

print(f'load_dotenv({DOTENV_FILE}): {load_dotenv(DOTENV_FILE)}')

NEBIUS_MODELS_JSON = Path(__file__).resolve().parent.parent / "nebius_batch_llm_inference" / "nebius_models.json"


def _register_langfuse_model_price(model_name: str) -> None:
    """Register custom model pricing in Langfuse via its API."""
    if not NEBIUS_MODELS_JSON.exists():
        print(f"  Pricing file not found: {NEBIUS_MODELS_JSON}")
        return

    pricing_data = json.loads(NEBIUS_MODELS_JSON.read_text())
    model_pricing = pricing_data.get(model_name)
    if not model_pricing or model_pricing.get("input_per_1m") is None:
        print(f"  No pricing found for model: {model_name}")
        return

    input_per_token = model_pricing["input_per_1m"] / 1_000_000
    output_per_token = model_pricing["output_per_1m"] / 1_000_000

    import requests

    langfuse_host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
    resp = requests.post(
        f"{langfuse_host.rstrip('/')}/api/public/models",
        auth=(os.environ["LANGFUSE_PUBLIC_KEY"], os.environ["LANGFUSE_SECRET_KEY"]),
        json={
            "modelName": model_name,
            "matchPattern": model_name,
            "inputPrice": input_per_token,
            "outputPrice": output_per_token,
            "unit": "TOKENS",
        },
        timeout=10,
    )
    if not resp.ok:
        print(f"  Langfuse model registration failed: {resp.status_code} {resp.text}", file=__import__('sys').stderr)
        return
    print(
        f"  Langfuse model pricing registered: {model_name} "
        f"(${model_pricing['input_per_1m']}/1M in, ${model_pricing['output_per_1m']}/1M out)"
    )


_lf_handler: Any = None
if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
    from langfuse.langchain import CallbackHandler

    _lf_handler = CallbackHandler()
    _register_langfuse_model_price(VLLM_MODEL)


def main() -> None:
    client = _get_db_client()
    try:
        client.get_collection(COLLECTION_NAME)
    except Exception as exc:
        raise SystemExit(
            f"Collection '{COLLECTION_NAME}' not found. "
            "Ingest documents first using src/rag/ingestion.py."
        ) from exc

    print(f"Connected to collection: {COLLECTION_NAME}")
    if _lf_handler:
        print("Langfuse tracing enabled")
    print("Type your question (or 'quit' to exit):\n")

    config: dict[str, Any] = {
        "callbacks": [_lf_handler] if _lf_handler else [],
    }

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        state = AgentState(question=question)
        result = graph.invoke(state, config)

        answer = result.get("answer", "No answer.")
        iterations = result.get("iteration", 0)
        ok = result.get("verify_ok", False)

        print(f"\nAssistant: {answer}")
        print(f"  [iterations: {iterations}, verified: {ok}]\n")


if __name__ == "__main__":
    main()
