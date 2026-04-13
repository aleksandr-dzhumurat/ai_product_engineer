"""
Count total token usage from a batch inference results JSONL file.

Usage:
  python src/nebius_batch_llm_inference/count_tokens.py $(pwd)/data/4dfb25617ded4cef81b20e01d5954f01.jsonl
"""

import json
import os
import sys
from pathlib import Path

# Pricing per 1M tokens
INPUT_COST_1M = 0.02
OUTPUT_COST_1M = 0.06


def count_tokens(path: str) -> None:
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    rows = 0
    errors = 0

    input_path = Path(path)
    output_path = input_path.parent / f"processed_{input_path.name}"

    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1

            # 1. Update token stats
            usage = (row.get("raw_response") or {}).get("usage")
            if not usage:
                errors += 1
            else:
                prompt_tokens += usage.get("prompt_tokens", 0)
                completion_tokens += usage.get("completion_tokens", 0)
                total_tokens += usage.get("total_tokens", 0)

            # 2. Transform completion field: extract content from the array
            custom_id = row.get("custom_id")
            completion_arr = row.get("completion")
            completion_text = ""
            if isinstance(completion_arr, list) and len(completion_arr) > 0:
                completion_text = completion_arr[0].get("content", "")

            # 3. Save processed row (only keep custom_id and flattened completion)
            processed_row = {
                "custom_id": custom_id,
                "completion": completion_text
            }
            f_out.write(json.dumps(processed_row, ensure_ascii=False) + "\n")

    print(f"Rows processed : {rows}")
    print(f"Rows with errors (no usage): {errors}")
    print(f"Prompt tokens  : {prompt_tokens:,}")
    print(f"Completion tokens: {completion_tokens:,}")
    print(f"Total tokens   : {total_tokens:,}")

    prompt_cost = (prompt_tokens / 1_000_000) * INPUT_COST_1M
    completion_cost = (completion_tokens / 1_000_000) * OUTPUT_COST_1M
    total_cost = prompt_cost + completion_cost
    print(f"\nEstimated Cost : ${total_cost:.4f}")
    print(f"Processed file saved to: {output_path}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/4dfb25617ded4cef81b20e01d5954f01.jsonl"
    count_tokens(path)
