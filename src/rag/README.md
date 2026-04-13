# RAG Pipeline

### Prerequisites

Pull an embedding model via Ollama:
```bash
ollama pull nomic-embed-text
```

Check available models:
```bash
ollama list
```

Launch Chroma:
```bash
make run-chroma
```

### Notebook → Markdown

Convert a directory of notebooks:
```bash
python src/rag/ipynb2md.py $(pwd)/jupyter_notebooks
```
Each conversion appends a JSON entry to `data/md_docs/conversion_log.jsonl` with `source_dir`, `source_file_name`, and `desctination_file`.

### HTML → Markdown

Convert downloaded Nebius HTML pages:
```bash
python src/rag/nebius_html2md.py --input data/nebius_site/
```
Conversion log is saved at the top level of `--input` directory (e.g. `data/nebius_site/conversion_log_811e314c.jsonl`).

### Text Processing

Chunk a Markdown file and print stats:
```bash
python src/rag/files_processing.py --input path/to/file.md
python src/rag/files_processing.py --input path/to/file.md --chunk-size 500 --chunk-overlap 100
```

### Ingestion

Ingest converted documents into ChromaDB:
```bash
DATA_DIR="$(pwd)" python src/rag/ingestion.py --log-path data/nebius_site/conversion_log_811e314c.jsonl --embedding-model nomic-embed-text --reset-collection
DATA_DIR="$(pwd)" python src/rag/ingestion.py --log-path data/md_docs/conversion_log.jsonl --collection my_collection --embedding-model nomic-embed-text
```

Key flags: `--batch-size`, `--chunk-size`, `--chunk-overlap`, `--embedding-model`, `--ollama-host`, `--request-timeout`, `--reset-collection`.

### Retrieval

Query the collection:
```bash
DATA_DIR="$(pwd)" python src/rag/retrieve.py "your query" --embedding-model nomic-embed-text
DATA_DIR="$(pwd)" python src/rag/retrieve.py "your query" --collection my_collection --limit 10
```

### LLM Pricing

Nebius Token Factory endpoints and pricing: https://tokenfactory.nebius.com/endpoints

### RAG Flow

```
    +-----------------------+          +-----------------------+
    |  Notebook / HTML      |          |       Retrieval       |
    |  Conversion + log     |          |   (src/retrieve.py)   |
    +-----------+-----------+          +-----------+-----------+
                |                                   ^
                v                                   |
    +-----------------------+          +-----------+-----------+
    |  Ingestion Pipeline   |--------->|       ChromaDB        |
    |  (src/ingestion.py)   |   embed  |   (Vector Database)   |
    +-----------------------+          +-----------------------+
```
