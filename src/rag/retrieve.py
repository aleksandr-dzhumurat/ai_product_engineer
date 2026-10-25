"""
Query documents stored in a ChromaDB collection.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.agent.config import EMBEDDING_MODEL_NAME, OLLAMA_HOST, VECTOR_DB_BACKEND
from rag.connections import BaseConnection, ChromaConnection, WeaviateConnection


def fetch_query_embedding(query: str, host: str, model: str, timeout: float = 60.0) -> List[float]:
    """
    Fetch embedding for a query from Ollama.

    Args:
        query: The query text to embed
        host: Ollama host URL
        model: Ollama model name
        timeout: Request timeout in seconds

    Returns:
        Embedding vector for the query
    """
    url = host.rstrip("/") + "/api/embeddings"
    response = requests.post(
        url,
        json={"model": model, "prompt": query},
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()

    embedding = data.get("embedding") or data.get("embeddings")
    if not embedding:
        raise ValueError(f"No embedding found in response: {data}")

    return embedding


def query_collection(
    client: BaseConnection,
    collection_name: str,
    limit: int,
    include_distances: bool,
    query_embedding: List[float],
) -> dict:
    include = ["documents", "metadatas"]
    if include_distances:
        include.append("distances")

    return client.query(
        collection_name=collection_name,
        query_embeddings=[query_embedding],
        n_results=limit,
        include=include,
    )


def print_results(results: dict, include_distances: bool) -> None:
    documents = results.get("documents") or []
    metadatas = results.get("metadatas") or []
    distances = results.get("distances") if include_distances else None

    if not documents:
        print("No results.")
        return

    docs = documents[0]
    metas = metadatas[0] if metadatas else [{}] * len(docs)
    dists = distances[0] if distances else [None] * len(docs)

    for idx, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
        source = meta.get("source", "unknown")
        chunk_index = meta.get("chunk_index")
        header = f"{idx}. {source}"
        if chunk_index is not None:
            header += f" (chunk {chunk_index})"
        if dist is not None:
            header += f" — distance: {dist:.4f}"
        print(header)
        print(doc)
        print("-" * 80)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Natural language query to search for.")
    parser.add_argument(
        "--persist-dir",
        default="data/chroma",
        help="ChromaDB persistence directory, Chroma backend only (default: data/chroma).",
    )
    parser.add_argument(
        "--backend",
        default=VECTOR_DB_BACKEND,
        choices=["chroma", "weaviate"],
        help=f"Vector store backend (default: {VECTOR_DB_BACKEND}, override with VECTOR_DB_BACKEND).",
    )
    parser.add_argument(
        "--collection",
        default="documents",
        help="Vector DB collection name (default: documents).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to return (default: 5).",
    )
    parser.add_argument(
        "--no-distances",
        action="store_true",
        help="Do not display similarity distances.",
    )
    parser.add_argument(
        "--embedding-model",
        default=EMBEDDING_MODEL_NAME,
        help=f"Ollama embedding model (default: {EMBEDDING_MODEL_NAME}).",
    )
    parser.add_argument(
        "--ollama-host",
        default=OLLAMA_HOST,
        help=f"Ollama API host (default: {OLLAMA_HOST}).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.backend == "weaviate":
        client = WeaviateConnection()
    else:
        client = ChromaConnection()

    try:
        try:
            client.get_collection(args.collection)
        except Exception as exc:
            raise SystemExit(
                f"Collection '{args.collection}' was not found: {exc}.\n"
                "Ingest documents first using src/rag/ingestion.py."
            ) from exc

        print(f"Generating embedding for query: '{args.query}'")
        query_embedding = fetch_query_embedding(
            args.query,
            host=args.ollama_host,
            model=args.embedding_model,
        )
        print(f"Embedding dimension: {len(query_embedding)}")
        print(f"Searching in collection: {args.collection}")
        print()

        results = query_collection(
            client,
            args.collection,
            args.limit,
            include_distances=not args.no_distances,
            query_embedding=query_embedding,
        )
        print_results(results, include_distances=not args.no_distances)
    finally:
        close = getattr(client.client, "close", None)
        if close is not None:
            close()


if __name__ == "__main__":
    main()
