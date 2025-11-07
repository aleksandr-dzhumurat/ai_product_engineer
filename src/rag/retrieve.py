"""
Query documents stored in a ChromaDB collection.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from chromadb.errors import InvalidCollectionException
from chromadb.api.models.Collection import Collection

from connections import get_chroma_client


def query_collection(
    collection: Collection,
    query: str,
    limit: int,
    include_distances: bool,
) -> dict:
    include = ["documents", "metadatas"]
    if include_distances:
        include.append("distances")
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
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    persist_dir = Path(args.persist_dir).expanduser()
    client = get_chroma_client(persist_dir)

    try:
        collection = client.get_collection(args.collection)
    except InvalidCollectionException as exc:
        raise SystemExit(
            f"Collection '{args.collection}' was not found in {persist_dir}. "
            "Ingest documents first using src/ingestion.py."
        ) from exc

    results = query_collection(collection, args.query, args.limit, include_distances=not args.no_distances)
    print_results(results, include_distances=not args.no_distances)


if __name__ == "__main__":
    main()
