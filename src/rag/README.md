# Utility Scripts

### Interview History Export
- `python3 scripts/process_interview.py ~/Downloads/Interview_History_Ashby.mhtml`
- `python3 scripts/chart_interviews.py data/interview_history.csv --reference-date 2025-10-09`
- `python3 scripts/sum_interview_hours.py data/interview_history.csv --reference-date 2025-10-09`

### Notebook → Markdown
- Convert a single notebook: `python3 scripts/ipynb2md.py path/to/notebook.ipynb`
- Convert a directory: `python3 scripts/ipynb2md.py /path/to/notebooks/` (outputs flat files under `data/md_docs/`)
- Each conversion appends a JSON entry to `data/md_docs/conversion_log.jsonl` with `source_dir`, `source_file_name`, and `desctination_file` for checkpointing.

### Text Processing
- Split text with LangChain splitter:
  ```bash
  python3 - <<'PY'
  from src.text_processing import split_text
  chunks = split_text("long text goes here", chunk_size=500, chunk_overlap=100)
  print(chunks)
  PY
  ```
- Chunk-size references:
  - OpenAI `text-embedding-3-*` models accept up to 8,192 tokens; in practice, ~700–900 tokens (≈3,000–3,600 characters) balances recall and cost.
  - Gemini embedding models (e.g., `text-embedding-004`) allow up to 32,768 characters; ~1,000–1,200 tokens (≈4,000–5,000 characters) keeps overlap manageable while staying well within limits.
  - Popular Hugging Face sentence transformers such as `all-MiniLM-L6-v2` work best with shorter passages (≤512 tokens). Using ~350–450 tokens (≈1,400–1,800 characters) per chunk helps avoid truncation and preserves semantic cohesion.

### Quickstart
- RAG flow overview:

  ```
      +-----------------------+          +-----------------------+
      |  Notebook Conversion  |          |       Retrieval       |
      |  (ipynb2md.py + log)  |          |   (src/retrieve.py)   |
      +-----------+-----------+          +-----------+-----------+
                  |                                   ^
                  v                                   |
      +-----------------------+          +-----------+-----------+
      |  Ingestion Pipeline   |--------->|       ChromaDB        |
      |   (src/ingestion.py)  |   RAG    |   (Vector Database)   |
      +-----------------------+          +-----------------------+
  ```

- Launch Chroma locally:
  ```bash
  make run-chroma
  ```
- Ingest converted notebooks (filters by `source_dir` in `data/md_docs/conversion_log.jsonl`):
  ```bash
  uv run python src/ingestion.py "/Users/adzhumurat/PycharmProjects/ai_product_engineer/jupyter_notebooks"
  ```
- `src/ingestion.py` batches chunks (default 50) and for each batch calls Ollama’s embedding API (default model `granite4:350m`) before upserting into Chroma. Tune the batch/embedding parameters via CLI flags (`--batch-size`, `--chunk-size`, `--chunk-overlap`, `--embedding-model`, `--ollama-host`, `--request-timeout`).
