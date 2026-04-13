"""Ingest Markdown documents into ChromaDB using the conversion log.

Usage:
    DATA_DIR="$(pwd)"python src/rag/ingestion.py --log-path data/nebius_site/conversion_log_811e314c.jsonl
    python src/rag/ingestion.py --log-path data/md_docs/conversion_log.jsonl --collection my_collection
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import requests
from connections import BaseConnection, ChromaConnection
from files_processing import chunk_document


@dataclass
class EmbeddingModel:
    name: str
    embedding_size: int


def load_transformation_log(log_path: Path) -> List[dict]:
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


def fetch_embeddings(
    texts: Sequence[str],
    *,
    host: str,
    model: EmbeddingModel,
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
        if not text or not text.strip():
            print(f"Warning: Skipping empty text at index {idx}")
            embeddings.append([0.0] * model.embedding_size)
            continue

        try:
            response = requests.post(
                url,
                json={"model": model.name, "prompt": text},
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()

            embedding = data.get("embedding") or data.get("embeddings")

            if not embedding or (isinstance(embedding, list) and len(embedding) == 0):
                print(f"Warning: Empty embedding for text at index {idx} (length: {len(text)})")
                print(f"Text preview: {text[:100]}...")
                print(f"Response: {data}")
                embedding = [0.0] * model.embedding_size

            embeddings.append(embedding)

        except Exception as e:
            print(f"Error fetching embedding for text {idx}: {e}")
            print(f"Text preview: {text[:100]}...")
            raise

    return embeddings


def flush_batch(
    client: BaseConnection,
    collection_name: str,
    documents: List[str],
    metadatas: List[dict[str, str]],
    ids: List[str],
    *,
    host: str,
    model: EmbeddingModel,
    timeout: float,
) -> int:
    if not documents:
        return 0
    embeddings = fetch_embeddings(documents, host=host, model=model, timeout=timeout)
    client.upsert(
        collection_name=collection_name,
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


def ingest(
    entries: Iterable[dict],
    *,
    client: BaseConnection,
    collection_name: str,
    batch_size: int,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: EmbeddingModel,
    ollama_host: str,
    request_timeout: float,
) -> int:
    try:
        client.get_collection(collection_name)
        print(f"Using existing collection: {collection_name}")
    except Exception:
        print(f"Creating new collection: {collection_name}")
        client.create_collection(collection_name)

    documents: List[str] = []
    metadatas: List[dict[str, str]] = []
    ids: List[str] = []
    total_chunks = 0

    for n, entry in enumerate(entries):
        if n % 2 == 0:
            print(f'{n} from {len(entries)}')
        file_path = Path(entry["desctination_file"])
        chunks = chunk_document(file_path, chunk_size, chunk_overlap)
        if not chunks:
            continue

        for chunk in chunks:
            documents.append(chunk.body)
            metadatas.append(
                {
                    "source": str(chunk.source),
                    "source_dir": str(chunk.source.parent),
                    "source_file_name": chunk.source.name,
                    "length_chars": str(chunk.length_chars),
                    "length_lines": str(chunk.length_lines),
                }
            )
            ids.append(chunk.id)
            total_chunks += 1

            if len(documents) >= batch_size:
                flush_batch(
                    client,
                    collection_name,
                    documents,
                    metadatas,
                    ids,
                    host=ollama_host,
                    model=embedding_model,
                    timeout=request_timeout,
                )

    if documents:
        flush_batch(
            client,
            collection_name,
            documents,
            metadatas,
            ids,
            host=ollama_host,
            model=embedding_model,
            timeout=request_timeout,
        )
    return total_chunks


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-path",
        required=True,
        help="Path to conversion log JSONL.",
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
        default="nomic-embed-text",
        help="Ollama embedding model name (default: nomic-embed-text).",
    )
    parser.add_argument(
        "--embedding-size",
        type=int,
        default=768,
        help="Embedding vector size (default: 768).",
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
    entries = load_transformation_log(log_path)
    if not entries:
        raise SystemExit(f"No entries found in {log_path}.")

    model = EmbeddingModel(name=args.embedding_model, embedding_size=args.embedding_size)
    client = ChromaConnection()

    # Handle collection reset if requested
    if args.reset_collection:
        try:
            client.delete_collection(args.collection)
            print(f"Deleted existing collection: {args.collection}")
            time.sleep(1)
        except Exception as e:
            print(f"No existing collection to delete: {e}")

    total = ingest(
        entries,
        client=client,
        collection_name=args.collection,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_model=model,
        ollama_host=args.ollama_host,
        request_timeout=args.request_timeout,
    )

    print(
        f"Ingested {total} chunks from {len(entries)} files into collection "
        f"'{args.collection}'"
    )


if __name__ == "__main__":
    main()
