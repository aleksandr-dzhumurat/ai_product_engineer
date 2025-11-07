"""
Shared helpers for working with external services (e.g., ChromaDB).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import chromadb
from chromadb.config import Settings


def get_chroma_client(persist_dir: str | Path, **settings_overrides: Any) -> chromadb.Client:
    """
    Create (and ensure) a persistent ChromaDB client rooted at ``persist_dir``.

    Args:
        persist_dir: Directory where ChromaDB should persist its data.
        **settings_overrides: Optional keyword overrides passed to ``Settings``.

    Returns:
        Configured ``chromadb.Client`` instance.
    """
    persist_path = Path(persist_dir).expanduser()
    persist_path.mkdir(parents=True, exist_ok=True)

    settings: Dict[str, Any] = {
        "chroma_db_impl": "duckdb+parquet",
        "persist_directory": str(persist_path),
    }
    settings.update(settings_overrides)

    return chromadb.Client(Settings(**settings))
