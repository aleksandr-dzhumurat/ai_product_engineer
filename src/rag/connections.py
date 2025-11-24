"""
Shared helpers for working with external services (e.g., ChromaDB).
"""
from __future__ import annotations

import os

import chromadb


def get_chroma_client(
    host: str | None = None,
    port: int = 8000,
) -> chromadb.Client:
    """
    Create a ChromaDB HTTP client to connect to the service.

    Args:
        host: ChromaDB service host (default: from CHROMA_HOST env or 'localhost').
        port: ChromaDB service port (default: from CHROMA_PORT env or 8000).

    Returns:
        Configured ``chromadb.HttpClient`` instance.

    Examples:
        # Connect to Chroma service
        client = get_chroma_client(host="chroma", port=8000)

        # Auto-detect from environment (CHROMA_HOST, CHROMA_PORT)
        client = get_chroma_client()
    """
    # Get host from parameter or environment
    if host is None:
        host = os.getenv('CHROMA_HOST', 'localhost')

    # Get port from environment or use default
    chroma_port = int(os.getenv('CHROMA_PORT', str(port)))

    print(f"Connecting to ChromaDB service at http://{host}:{chroma_port}")
    return chromadb.HttpClient(host=host, port=chroma_port)
