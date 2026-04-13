"""Ingest Markdown documents into ChromaDB using the conversion log.

Usage:
    DATA_DIR="$(pwd)"python src/rag/ingestion.py --log-path data/nebius_site/conversion_log_811e314c.jsonl
    python src/rag/ingestion.py --log-path data/md_docs/conversion_log.jsonl --collection my_collection
"""
from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable, Sequence
from pathlib import Path

from agent.config import (
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_SIZE,
    OLLAMA_HOST,
)
from connections import BaseConnection, ChromaConnection
from embedder import Embedder, EmbeddingModel, OllamaEmbedder
from files_processing import chunk_document


def load_transformation_log(log_path: Path) -> list[dict]:
    if not log_path.exists():
        raise FileNotFoundError(f"Conversion log not found: {log_path}")
    entries: list[dict] = []
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


def flush_batch(
    client: BaseConnection,
    collection_name: str,
    documents: list[str],
    metadatas: list[dict[str, str]],
    ids: list[str],
    *,
    embedder: Embedder,
) -> int:
    if not documents:
        return 0
    embeddings = embedder.embed(documents)
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
    embedder: Embedder,
) -> int:
    try:
        client.get_collection(collection_name)
        print(f"Using existing collection: {collection_name}")
    except Exception:
        print(f"Creating new collection: {collection_name}")
        client.create_collection(collection_name)

    documents: list[str] = []
    metadatas: list[dict[str, str]] = []
    ids: list[str] = []
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
                    embedder=embedder,
                )

    if documents:
        flush_batch(
            client,
            collection_name,
            documents,
            metadatas,
            ids,
            embedder=embedder,
        )
    return total_chunks


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-path",
        default="data/md_docs/conversion_log.jsonl",
        help="Path to conversion log JSONL (default: data/md_docs/conversion_log.jsonl).",
    )
    parser.add_argument(
        "--persist-dir",
        default="data/chroma",
        help="ChromaDB persistence directory (default: data/chroma).",
    )
    parser.add_argument(
        "--collection",
        default=COLLECTION_NAME,
        help=f"ChromaDB collection name (default: {COLLECTION_NAME}).",
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
        default=EMBEDDING_MODEL_NAME,
        help=f"Ollama embedding model name (default: {EMBEDDING_MODEL_NAME}).",
    )
    parser.add_argument(
        "--embedding-size",
        type=int,
        default=EMBEDDING_SIZE,
        help=f"Embedding vector size (default: {EMBEDDING_SIZE}).",
    )
    parser.add_argument(
        "--ollama-host",
        default=OLLAMA_HOST,
        help=f"Base URL for the Ollama API (default: {OLLAMA_HOST}).",
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
    embedder = OllamaEmbedder(host=args.ollama_host, model=model, timeout=args.request_timeout)
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
        embedder=embedder,
    )

    collection = client.get_collection(args.collection)
    doc_count = collection.count()
    print(
        f"Ingested {total} chunks from {len(entries)} files into collection "
        f"'{args.collection}' (total documents in collection: {doc_count})"
    )


if __name__ == "__main__":
    main()
