"""
Ingest notebook-derived Markdown documents into ChromaDB using the conversion log.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable, List, Sequence

import chromadb
import requests
from chromadb.api.models.Collection import Collection

from connections import get_chroma_client
from text_processing import split_text


DEFAULT_LOG_PATH = Path("data/md_docs/conversion_log.jsonl")


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
    if not texts:
        return []
    url = host.rstrip("/") + "/api/embeddings"
    response = requests.post(
        url,
        json={"model": model, "prompt": list(texts)},
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    embeddings = data.get("embeddings")
    if not isinstance(embeddings, list) or len(embeddings) != len(texts):
        raise ValueError(
            f"Unexpected embeddings response shape: {data}"
        )
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
    collection = client.get_or_create_collection(collection_name)

    documents: List[str] = []
    metadatas: List[dict[str, str]] = []
    ids: List[str] = []
    total_chunks = 0

    for entry in entries:
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

    client.persist()
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

    filtered_entries = [entry for entry in entries if entry.get("source_dir") == args.directory]
    if not filtered_entries:
        raise SystemExit(
            f"No entries found in {log_path} with source_dir = '{args.directory}'."
        )

    persist_dir = Path(args.persist_dir).expanduser()
    client = get_chroma_client(persist_dir)

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
        f"'{args.collection}' at {persist_dir}"
    )


if __name__ == "__main__":
    main()
