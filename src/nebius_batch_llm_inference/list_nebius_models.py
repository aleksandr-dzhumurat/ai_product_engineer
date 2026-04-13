"""Probe the Nebius Token Factory model catalog.

Reads the API key from either the NEBIUS_API_KEY env var or a local
`nebius_api_key` file (in that order). Prints model IDs to stdout; the
key value is never logged.

Usage:
    DOTENV_FILE="$(pwd)/.env" DATA_DIR="$(pwd)" python src/nebius_batch_llm_inference/list_nebius_models.py
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

from dotenv import load_dotenv

DOTENV_FILE = os.environ.get("DOTENV_FILE", ".env")

print(f'load_dotenv({DOTENV_FILE}): {load_dotenv(DOTENV_FILE)}')


def load_key() -> str:
    env_key = os.environ.get("NEBIUS_API_KEY")
    if env_key:
        return env_key.strip()
    candidate = pathlib.Path("mlops-hw-tf-api-key")
    if not candidate.exists():
        print(
            "No API key found. Set NEBIUS_API_KEY env var or create a `nebius_api_key` file.",
            file=sys.stderr,
        )
        sys.exit(1)
    raw = candidate.read_bytes()
    has_bom = raw.startswith(b"\xef\xbb\xbf")
    text = candidate.read_text(encoding="utf-8-sig")
    key = text.strip().strip("'\"")
    # Safe diagnostics: never prints any character of the key.
    print(
        f"diag: file_bytes={len(raw)} has_bom={has_bom} "
        f"key_len={len(key)} all_ascii={key.isascii()} "
        f"has_inner_ws={any(c.isspace() for c in key)}",
        file=sys.stderr,
    )
    return key


# Pricing per 1M tokens (input, output) in USD.
# Source: https://tokenfactory.nebius.com/endpoints (2026-06-30)
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "Kimi-K2.7-Code": (0.95, 4.00),
    "GLM-5.2": (1.40, 4.40),
    "Nemotron-3-Ultra-550b-a55b": (1.00, 3.00),
    "Cosmos3-Super-Reasoner": (0.10, 0.30),
    "openbmb/MiniCPM-V-4_5": (0.658, 1.11),
    "Kimi-K2.6": (0.95, 4.00),
    "DeepSeek-V4-Pro": (1.75, 3.50),
    "Nemotron-3-Nano-Omni": (0.06, 0.24),
    "GLM-5.1": (1.40, 4.40),
    "MiniMax-M2.5": (0.30, 1.20),
    "Nemotron-3-Super-120b-a12b": (0.30, 0.90),
    "Qwen3.5-397B-A17B": (0.60, 3.60),
    "Hermes-4-405B": (1.00, 3.00),
    "Hermes-4-70B": (0.13, 0.40),
    "gpt-oss-120b": (0.15, 0.60),
    "Qwen3-235B-A22B-Instruct-2507": (0.20, 0.60),
    "Qwen3-30B-A3B-Instruct-2507": (0.10, 0.30),
    "Qwen3-Embedding-8B": (0.01, 0.0),
    "Qwen3-Next-80B-A3B-Thinking": (0.15, 1.20),
    "Qwen3-32B": (0.10, 0.30),
    "Gemma-3-27b-it": (0.10, 0.30),
    "Llama-3_1-Nemotron-Ultra-253B-v1": (0.60, 1.80),
    "Nemotron-3-Nano-30B-A3B": (0.06, 0.24),
    "Qwen2.5-VL-72B-Instruct": (0.25, 0.75),
    "Llama-3.3-70B-Instruct": (0.13, 0.40),
}
# Case-insensitive lookup index
_PRICING_LOWER = {k.lower(): v for k, v in MODEL_PRICING.items()}


def _lookup_pricing(model_id: str) -> tuple[float, float] | None:
    short = model_id.split("/")[-1]
    # Try: full id, short name, short name without vendor prefix (e.g. "NVIDIA-")
    for candidate in [model_id, short]:
        result = _PRICING_LOWER.get(candidate.lower())
        if result:
            return result
    # Strip known vendor prefixes (e.g. "NVIDIA-Nemotron-..." -> "Nemotron-...")
    for prefix in ("NVIDIA-",):
        if short.upper().startswith(prefix):
            stripped = short[len(prefix):]
            result = _PRICING_LOWER.get(stripped.lower())
            if result:
                return result
    return None


def main() -> None:
    key = load_key()
    base_url = os.environ.get(
        "NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/"
    )

    try:
        from openai import OpenAI
    except ImportError:
        print("Install the openai package: pip install openai", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=key, base_url=base_url)
    try:
        page = client.models.list()
    except Exception as exc:
        # openai-python masks the api key in exception strings, but stay defensive.
        print(f"models.list failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)

    output = {}
    print(f"{'Model':<45} {'Input $/1M':>10} {'Output $/1M':>11}")
    print("-" * 68)
    for m in sorted(page.data, key=lambda x: x.id):
        pricing = _lookup_pricing(m.id)
        if pricing:
            in_price, out_price = pricing
            print(f"{m.id:<45} {in_price:>10.3f} {out_price:>11.3f}")
            output[m.id] = {"input_per_1m": in_price, "output_per_1m": out_price}
        else:
            print(f"{m.id:<45} {'N/A':>10} {'N/A':>11}")
            output[m.id] = {"input_per_1m": None, "output_per_1m": None}

    out_path = os.path.join(os.path.dirname(__file__), "nebius_models.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
