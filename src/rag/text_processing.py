"""
Utility helpers for text processing tasks.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover
    RecursiveCharacterTextSplitter = None  # type: ignore[assignment]


DEFAULT_CONVERSION_LOG = Path("data/md_docs/conversion_log.jsonl")


def split_text(
    text: str,
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: Sequence[str] | None = None,
) -> list[str]:
    """
    Split the provided text into manageable chunks using LangChain's recursive splitter.

    Args:
        text: Raw text that should be split.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Character overlap between consecutive chunks.
        separators: Optional custom separators passed to the splitter.

    Returns:
        List of text chunks.
    """
    if RecursiveCharacterTextSplitter is None:  # pragma: no cover
        raise ImportError(
            "langchain is not installed. Install it via `pip install langchain` to use split_text()."
        )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=list(separators) if separators is not None else None,
    )
    return splitter.split_text(text)


def record_conversion(
    source_path: Path,
    destination_path: Path,
    *,
    log_path: Path = DEFAULT_CONVERSION_LOG,
) -> None:
    """
    Append a JSON line describing a completed notebook-to-markdown conversion.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "source_dir": str(source_path.parent),
        "source_file_name": source_path.name,
        "desctination_file": str(destination_path),
    }
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
