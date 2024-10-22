"""
Ingest notebook-derived Markdown documents into ChromaDB using the conversion log.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Iterable, List, Sequence

import chromadb
import requests
from chromadb.api.models.Collection import Collection
from connections import get_chroma_client
from text_processing import split_text

DEFAULT_LOG_PATH = Path(os.environ.get('DATA_DIR', 'data')) / "md_docs/conversion_log.jsonl"


def load_conversion_log(log_path: Path) -> List[dict]:
    if not log_path.exists():
        raise FileNotFoundError(f"Conversion log not found: {log_path}")
    entries: List[dict] = []
    with log_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON line in log: {line}") from exc
            entries.append(entry)
    return entries


def read_text_file(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
    except OSError:
        return None


def chunk_document(path: Path, chunk_size: int, chunk_overlap: int) -> List[str]:
    content = read_text_file(path)
    if content is None:
        return []
    return split_text(content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def fetch_embeddings(
    texts: Sequence[str],
    *,
    host: str,
    model: str,
    timeout: float,
) -> List[List[float]]:
    """
    Fetch embeddings from Ollama API.

    Note: Ollama's /api/embeddings endpoint accepts one text at a time.
    We need to make individual requests for each text.
    """
    if not texts:
        return []

    url = host.rstrip("/") + "/api/embeddings"
    embeddings = []

    for idx, text in enumerate(texts):
        # Skip empty texts
        if not text or not text.strip():
            print(f"Warning: Skipping empty text at index {idx}")
            # Return zero vector for empty text
            embeddings.append([0.0] * 384)  # Default embedding size
            continue

        try:
            response = requests.post(
                url,
                json={"model": model, "prompt": text},
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()

            # Ollama returns embedding in "embeddings" or "embedding" field
            embedding = data.get("embedding") or data.get("embeddings")

            if not embedding or (isinstance(embedding, list) and len(embedding) == 0):
                print(f"Warning: Empty embedding for text at index {idx} (length: {len(text)})")
                print(f"Text preview: {text[:100]}...")
                print(f"Response: {data}")
                # Use zero vector as fallback
                if embeddings:
                    embedding = [0.0] * len(embeddings[0])
                else:
                    embedding = [0.0] * 384  # Default size

            embeddings.append(embedding)

        except Exception as e:
            print(f"Error fetching embedding for text {idx}: {e}")
            print(f"Text preview: {text[:100]}...")
            raise

    return embeddings


def flush_batch(
    collection: Collection,
    documents: List[str],
    metadatas: List[dict[str, str]],
    ids: List[str],
    *,
    host: str,
    model: str,
    timeout: float,
) -> int:
    if not documents:
        return 0
    embeddings = fetch_embeddings(documents, host=host, model=model, timeout=timeout)
    collection.upsert(
        documents=list(documents),
        metadatas=list(metadatas),
        ids=list(ids),
        embeddings=embeddings,
    )
    count = len(documents)
    documents.clear()
    metadatas.clear()
    ids.clear()
    return count


def ingest_from_log(
    entries: Iterable[dict],
    *,
    client: chromadb.Client,
    collection_name: str,
    batch_size: int,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str,
    ollama_host: str,
    request_timeout: float,
) -> int:
    # Try to get existing collection, or create new one
    try:
        collection = client.get_collection(collection_name)
        print(f"Using existing collection: {collection_name}")
    except Exception:
        # Create new collection - ChromaDB will auto-detect dimensions from first insert
        print(f"Creating new collection: {collection_name}")
        collection = client.create_collection(collection_name)

    documents: List[str] = []
    metadatas: List[dict[str, str]] = []
    ids: List[str] = []
    total_chunks = 0

    for n, entry in enumerate(entries):
        if n % 2 == 0:
            print(f'{n} from {len(entries)}')
        destination = entry.get("desctination_file")
        source_dir = entry.get("source_dir", "")
        source_file = entry.get("source_file_name", "")

        if not destination:
            continue

        file_path = Path(destination)
        chunks = chunk_document(file_path, chunk_size, chunk_overlap)
        if not chunks:
            continue

        file_hash = hashlib.md5(str(file_path.resolve()).encode("utf-8")).hexdigest()
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{file_hash}_{idx}"
            documents.append(chunk)
            metadatas.append(
                {
                    "source": destination,
                    "chunk_index": str(idx),
                    "source_dir": source_dir,
                    "source_file_name": source_file,
                }
            )
            ids.append(chunk_id)
            total_chunks += 1

            if len(documents) >= batch_size:
                flush_batch(
                    collection,
                    documents,
                    metadatas,
                    ids,
                    host=ollama_host,
                    model=embedding_model,
                    timeout=request_timeout,
                )

    if documents:
        flush_batch(
            collection,
            documents,
            metadatas,
            ids,
            host=ollama_host,
            model=embedding_model,
            timeout=request_timeout,
        )

    # Note: HttpClient doesn't have persist() - data is automatically persisted by the server
    return total_chunks


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        help="Filter entries from conversion log by matching source_dir (exact string match).",
    )
    parser.add_argument(
        "--log-path",
        default=str(DEFAULT_LOG_PATH),
        help="Path to conversion log JSONL (default: data/md_docs/conversion_log.jsonl).",
    )
    parser.add_argument(
        "--persist-dir",
        default="data/chroma",
        help="ChromaDB persistence directory (default: data/chroma).",
    )
    parser.add_argument(
        "--collection",
        default="documents",
        help="ChromaDB collection name (default: documents).",
    )
    parser.add_argument(
        "--reset-collection",
        action="store_true",
        help="Delete and recreate the collection if it exists (useful for dimension changes).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of chunks to embed and upsert per batch (default: 50).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size for the splitter (default: 1000).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Chunk overlap for the splitter (default: 200).",
    )
    parser.add_argument(
        "--embedding-model",
        default="granite4:350m",
        help="Ollama embedding model to use (default: granite4:350m).",
    )
    parser.add_argument(
        "--ollama-host",
        default="http://localhost:11434",
        help="Base URL for the Ollama API (default: http://localhost:11434).",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=120.0,
        help="Timeout in seconds for Ollama embedding requests (default: 120).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    log_path = Path(args.log_path).expanduser()
    entries = load_conversion_log(log_path)

    filtered_entries = [entry for entry in entries if args.directory in entry["source_dir"]]
    if not filtered_entries:
        raise SystemExit(
            f"No entries found in {log_path} with source_dir = '{args.directory}'."
        )

    # Connect to ChromaDB service (persist_dir is no longer used)
    client = get_chroma_client()

    # Handle collection reset if requested
    if args.reset_collection:
        try:
            client.delete_collection(args.collection)
            print(f"Deleted existing collection: {args.collection}")
            # Give ChromaDB server time to process the deletion
            time.sleep(1)
        except Exception as e:
            print(f"No existing collection to delete: {e}")

    total = ingest_from_log(
        filtered_entries,
        client=client,
        collection_name=args.collection,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_model=args.embedding_model,
        ollama_host=args.ollama_host,
        request_timeout=args.request_timeout,
    )

    print(
        f"Ingested {total} chunks from {len(filtered_entries)} files into collection "
        f"'{args.collection}'"
    )


if __name__ == "__main__":
    main()
