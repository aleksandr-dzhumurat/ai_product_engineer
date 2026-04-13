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

Launch Chroma (for local vector storage):
```bash
make run-chroma
```

### Nebius Docs Scraping Pipeline

Step 1 – Extract links from MHTML and crawl to discover all sub-pages:
```bash
python src/scraping/scrape_nebius.py --input data/nebius_site/nebius_main_page.mhtml --max-depth 4
```
Use `--max-depth` to control how deep the crawler follows sidebar links (default: 1). Higher values discover nested sub-pages (e.g. `/kubernetes/gpu/set-up` at depth 2, `/kubernetes/gpu/topology-aware-scheduling` at depth 3). The JSONL output includes a `depth` field for each discovered URL.

Step 2 – Download HTML pages listed in the JSONL index:
```bash
python src/scraping/scrape_nebius.py --input data/nebius_site/nebius_main_page.jsonl --download
```

Step 3 – Convert downloaded HTML to Markdown:
```bash
DATA_DIR="$(pwd)" python src/rag/nebius_html2md.py --input data/nebius_site/
```

Step 4 – Ingest jupyter into ChromaDB (for correct `.jsonl` looks at logs from previous step):
```bash
DATA_DIR="$(pwd)" python src/rag/ingestion.py --log-path data/nebius_site/conversion_log_<hash>.jsonl --reset-collection
```

Each step prints the command for the next step.

### Notebook → Markdown

Convert a single notebook or a directory of notebooks:
```bash
PYTHONPATH="$(pwd)" python src/rag/ipynb2md.py path/to/notebook.ipynb
PYTHONPATH="$(pwd)" python src/rag/ipynb2md.py path/to/notebooks/
PYTHONPATH="$(pwd)" python src/rag/ipynb2md.py path/to/notebook.ipynb --output-dir /tmp/md_output
```
Each conversion appends a JSON entry to `data/md_docs/conversion_log.jsonl` with `source_dir`, `source_file_name`, and `desctination_file`.

### HTML → Markdown

Convert downloaded Nebius HTML pages:
```bash
DATA_DIR="$(pwd)" python src/rag/nebius_html2md.py --input data/nebius_site/
```
Conversion log is saved at the top level of `--input` directory (e.g. `data/nebius_site/conversion_log_811e314c.jsonl`).

### Text Processing

Chunk a Markdown file and print stats:
```bash
PYTHONPATH="$(pwd)" python src/rag/files_processing.py --input path/to/file.md
PYTHONPATH="$(pwd)" python src/rag/files_processing.py --input path/to/file.md --chunk-size 500 --chunk-overlap 100
```

List all links found in Markdown files under a directory:
```bash
PYTHONPATH="$(pwd)" python src/rag/files_processing.py --show-links --input-dir data/md_docs
```

### Ingestion

Ingest converted documents into ChromaDB:
```bash
DATA_DIR="$(pwd)" python src/rag/ingestion.py --log-path data/nebius_site/conversion_log_811e314c.jsonl --embedding-model nomic-embed-text --reset-collection
DATA_DIR="$(pwd)" python src/rag/ingestion.py --log-path data/md_docs/conversion_log.jsonl --collection my_collection --embedding-model nomic-embed-text
```

Key flags: `--batch-size`, `--chunk-size`, `--chunk-overlap`, `--embedding-model`, `--embedding-size`, `--ollama-host`, `--request-timeout`, `--reset-collection`.

### Retrieval

Query the collection:
```bash
DATA_DIR="$(pwd)" python src/rag/retrieve.py "your query"
DATA_DIR="$(pwd)" python src/rag/retrieve.py "your query" --collection my_collection --limit 10
```

Key flags: `--embedding-model` (default: `granite4:350m`), `--ollama-host`, `--collection`, `--limit`, `--no-distances`.

### Chat

Interactive RAG chat with verify+revise loop:
```bash
DATA_DIR="$(pwd)" python src/rag/chat.py
DATA_DIR="$(pwd)" python src/rag/chat.py --retrieval-top 10
DOTENV_FILE="$(pwd)/.env" DATA_DIR="$(pwd)" python src/rag/chat.py --retrieval-top 10
```

Key flags: `--retrieval-top` (number of top chunks to retrieve, default: 5).

### Telegram Bot

RAG-powered Telegram bot with image OCR support (Qwen2.5-VL via Nebius API):
```bash
TG_BOT_TOKEN="..." TG_BOT_DATA="/tmp/tg_images" DATA_DIR="$(pwd)" python src/rag/tg_bot.py
```

Environment variables: `TG_BOT_TOKEN`, `TG_BOT_DATA`, `DATA_DIR`, `LLM_BASE_URL`, `NEBIUS_API_KEY`, `DOTENV_FILE`.

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

# Example

Prepare TokenFactory
```shell
python src/scraping/scrape_nebius.py --input data/tokenfactory_docs/tokenfactory_docs.jsonl --download

DATA_DIR="$(pwd)" python src/rag/nebius_html2md.py --input data/tokenfactory_docs/
```

Prepare Tavily

```shell
python src/scraping/scrape_tavily.py --input data/scraping/tavily_docs/tavily_docs.jsonl --download

 DATA_DIR="$(pwd)" python src/rag/tavily_html2md.py --input data/scraping/tavily_docs/
 ```

Then

```shell
DATA_DIR="$(pwd)" python src/rag/ingestion.py --log-path data/scraping/tavily_docs/conversion_log_8b23aad2.jsonl

DATA_DIR="$(pwd)" python src/rag/ingestion.py --log-path data/tokenfactory_docs/conversion_log_d996ed3f.jsonl   
```