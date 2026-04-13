"""
Utility helpers for text processing tasks.

Usage:
    python src/rag/files_processing.py --input path/to/file.md
    python src/rag/files_processing.py --input path/to/file.md --chunk-size 500 --chunk-overlap 100
"""
from __future__ import annotations

import argparse
import hashlib
import json
import uuid
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Sequence

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class DocumentChunk:
    body: str
    source: Path
    id: str = field(init=False)
    chunk_title: str = field(init=False)
    length_chars: int = field(init=False)
    length_lines: int = field(init=False)

    def __post_init__(self):
        self.id = str(uuid.uuid4())
        self.chunk_title = hashlib.md5((self.source.name + self.body).encode()).hexdigest()
        self.length_chars = len(self.body)
        self.length_lines = self.body.count("\n") + 1


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
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=list(separators) if separators is not None else None,
    )
    return splitter.split_text(text)


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


_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\(https?://[^\)]*\)")
_BARE_URL_RE = re.compile(r"https?://\S+")


def strip_urls(text: str) -> str:
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _BARE_URL_RE.sub("", text)
    return text


def chunk_document(path: Path, chunk_size: int, chunk_overlap: int) -> list[DocumentChunk]:
    content = read_text_file(path)
    if content is None:
        return []
    chunks = split_text(content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return [DocumentChunk(body=strip_urls(c), source=path) for c in chunks]


DEFAULT_OUTPUT_DIR = Path(os.environ.get('DATA_DIR', 'data')) / "md_docs"


def target_markdown_path(input_path: Path, output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / (input_path.stem + ".md")


def convert_file(
    input_path: Path,
    output_path: Path,
    converter: Callable[[Path], str],
) -> tuple[Path, Path]:
    """Run converter on input_path, write result to output_path. Returns (source, destination)."""
    markdown = converter(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    return input_path, output_path


def log_entry_to_document_index(
    source_path: Path,
    destination_path: Path,
    *,
    log_path: Path = None,
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


EXCLUDED_DIRS = {
    ".venv",
    ".env",
    ".git",
    ".cache",
    "__pycache__",
    "uv",
    ".ipynb_checkpoints",
    "catboost_info",
}
EXCLUDED_PREFIXES = (".",)


def get_directory_tree(directory: Path, extension: str = ".ipynb") -> List[dict]:
    entries: List[dict] = []
    for root, dirs, files in os.walk(directory):
        filtered_dirs: List[str] = []
        for d in dirs:
            full_path = Path(root) / d
            if d in EXCLUDED_DIRS or d.startswith(EXCLUDED_PREFIXES):
                print(f"Skipping directory {full_path}")
                continue
            filtered_dirs.append(d)
        dirs[:] = filtered_dirs
        for name in files:
            if not name.endswith(extension):
                continue
            source_path = Path(root) / name
            entries.append({"input_path": source_path, "output_path": None})
    return entries


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Chunk a Markdown file and print stats.")
    parser.add_argument("--input", required=True, help="Path to a .md file to chunk.")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size (default: 1000).")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap (default: 200).")
    args = parser.parse_args(argv)

    file_path = Path(args.input).expanduser()
    if file_path.suffix != ".md":
        raise SystemExit(f"Expected a .md file, got: {file_path}")
    if not file_path.exists():
        raise SystemExit(f"File not found: {file_path}")

    chunks = chunk_document(file_path, args.chunk_size, args.chunk_overlap)
    if not chunks:
        raise SystemExit(f"No content to chunk in {file_path}")

    lengths = [c.length_chars for c in chunks]
    line_counts = [c.length_lines for c in chunks]
    print(f"File: {file_path}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Avg chunk length: {sum(lengths) / len(lengths):.0f} chars")
    print(f"Min chunk length: {min(lengths)} chars")
    print(f"Max chunk length: {max(lengths)} chars")
    print(f"Avg lines per chunk: {sum(line_counts) / len(line_counts):.0f}")
    print(f"Min lines per chunk: {min(line_counts)}")
    print(f"Max lines per chunk: {max(line_counts)}")
    print(f"\n--- First chunk ---\n{chunks[0].body}")
    print(f"\n--- Last chunk ---\n{chunks[-1].body}")


if __name__ == "__main__":
    main()
