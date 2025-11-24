"""
Query documents stored in a ChromaDB collection.
"""
from __future__ import annotations

import argparse
from typing import List, Sequence

import requests
from chromadb.api.models.Collection import Collection
from connections import get_chroma_client


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
    collection: Collection,
    query: str,
    limit: int,
    include_distances: bool,
    query_embedding: List[float] | None = None,
) -> dict:
    """
    Query the collection with either text or embedding.

    Args:
        collection: ChromaDB collection
        query: Query text (used if query_embedding is None)
        limit: Number of results
        include_distances: Whether to include distances
        query_embedding: Pre-computed embedding vector (optional)

    Returns:
        Query results dictionary
    """
    include = ["documents", "metadatas"]
    if include_distances:
        include.append("distances")

    if query_embedding:
        # Use embedding-based query (more accurate)
        return collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            include=include,
        )
    else:
        # Fallback to text query (ChromaDB will use its default embedding)
        return collection.query(
            query_texts=[query],
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
        help="ChromaDB persistence directory (default: data/chroma).",
    )
    parser.add_argument(
        "--collection",
        default="documents",
        help="ChromaDB collection name (default: documents).",
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
        default="granite4:350m",
        help="Ollama embedding model (default: granite4:350m).",
    )
    parser.add_argument(
        "--ollama-host",
        default="http://localhost:11434",
        help="Ollama API host (default: http://localhost:11434).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    client = get_chroma_client()

    try:
        collection = client.get_collection(args.collection)
    except Exception as exc:
        raise SystemExit(
            f"Collection '{args.collection}' was not found. "
            "Ingest documents first using src/rag/ingestion.py."
        ) from exc

    # Fetch query embedding from Ollama
    print(f"Generating embedding for query: '{args.query}'")
    query_embedding = fetch_query_embedding(
        args.query,
        host=args.ollama_host,
        model=args.embedding_model,
    )
    print(f"Embedding dimension: {len(query_embedding)}")
    print(f"Searching in collection: {args.collection}")
    print()

    # Query the collection
    results = query_collection(
        collection,
        args.query,
        args.limit,
        include_distances=not args.no_distances,
        query_embedding=query_embedding,
    )
    print_results(results, include_distances=not args.no_distances)


if __name__ == "__main__":
    main()
