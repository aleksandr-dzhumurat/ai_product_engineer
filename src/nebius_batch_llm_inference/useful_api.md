# Nebius TokenFactory API – Useful Links

## Documentation

- [Nebius TokenFactory API – Main Documentation](https://docs.tokenfactory.nebius.com/)

## API Endpoints Used

Base URL: `https://api.tokenfactory.nebius.com`

### Datasets

- [POST /v1/datasets – Create and upload a dataset with schema and rows](https://api.tokenfactory.nebius.com/v1/datasets)
- [GET /v1/datasets/{id}/export?format=jsonl – Export dataset results as JSONL](https://api.tokenfactory.nebius.com/v1/datasets/{id}/export)

### Batch Inference Operations

- [POST /v1/operations – Start a batch inference operation](https://api.tokenfactory.nebius.com/v1/operations)
- [GET /v1/operations/{id} – Poll operation status (queued → running → completed)](https://api.tokenfactory.nebius.com/v1/operations/{id})

## Reasoning Control

The model `openai/gpt-oss-20b` is a **reasoning model** — it generates a silent chain-of-thought before the answer, billed as completion tokens (~4× overhead).

### Options to reduce reasoning tokens:

**Option A — Switch to a non-reasoning model** (recommended, eliminates reasoning entirely):
```python
MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct"
```

**Option B — Try `reasoning_effort` via extra params** (OpenAI-compatible, not guaranteed on Nebius):

The Nebius API is OpenAI-compatible and exposes "the full set of vLLM parameters". You can try passing `reasoning_effort` inside `params`:
```python
"params": {
    "model": MODEL,
    "completion_window": COMPLETION_WINDOW,
    "reasoning_effort": "none",   # try: "none" | "low" | "medium" | "high"
}
```
If the model/platform supports it, `"none"` disables reasoning entirely.

**Option C — Cap `max_tokens`** (limits damage, doesn't disable reasoning):

From the existing results dataset analysis (p50=653, p90=1314, p99=2638), setting `max_tokens=3500` covers 99%+ of translations while bounding runaway reasoning. Currently set in the script.

### References
- [Nebius inference overview — model flavors and vLLM parameters](https://docs.tokenfactory.nebius.com/ai-models-inference/overview)
- [Text generation examples — supported params](https://docs.tokenfactory.nebius.com/api-reference/examples/text-generation)

---

## Key Concepts

- **Dataset schema**: columns typed as `string` or `json`; the `prompt` column must contain pre-formatted JSON message arrays (`[{"role": "user", "content": "..."}]`), not plain text
- **`text_messages` mapping**: used in batch inference `src.mapping` to map dataset columns to model input; requires a `prompt` column with JSON messages and a `custom_id` column
- **`folder`**: required field on dataset creation — Nebius Cloud folder ID (format: `e00xxxxxxxxx`), found in Nebius Console project settings
- **`current_version`**: field in the `POST /v1/datasets` response containing the version ID needed for batch inference
