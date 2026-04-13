"""
Utility helpers for text processing tasks.

Usage examples::

    # Chunk a single Markdown file
    PYTHONPATH="$(pwd)" python src/rag/files_processing.py --input path/to/file.md
    PYTHONPATH="$(pwd)" python src/rag/files_processing.py --input path/to/file.md --chunk-size 500 --chunk-overlap 100

    # List all links found in Markdown files under a directory
    PYTHONPATH="$(pwd)" python src/rag/files_processing.py --show-links --input-dir data/md_docs

    # Filter links.jsonl to only include entries with review_link = true
    cat slides/links.jsonl | python3 -c "import sys,json; [print(line.strip()) for line in sys.stdin if json.loads(line).get('review_link')]" > slides/links_filtered.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import uuid
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
_MD_LINK_WITH_URL_RE = re.compile(r"\[([^\]]*)\]\((https?://[^\)]*)\)")
_BARE_URL_RE = re.compile(r"https?://\S+")

_GENERIC_TITLES = frozenset({
    "link", "here", "click here", "click", "source", "ref", "url", "this",
    "see", "more", "read more", "details", "info", "paper", "video", "image",
    "img", "fig", "figure", "table", "code", "demo", "example", "blog", "post",
    "article", "site", "page", "web", "docs", "doc", "note", "notes",
    "documentation", "reference", "tutorial", "guide", "manual", "wiki",
    "readme", "faq", "download", "install", "setup", "official",
    "official site", "official docs", "official documentation",
    "google drive", "colab notebook",
})

_SKIP_DOMAINS = frozenset({
    "drive.google.com",
    "localhost",
    "127.0.0.1",
})

_FILENAME_RE = re.compile(r"^[\w\.-]+\.\w{1,5}$")


def _is_informative_title(title: str, url: str = "") -> bool:
    from urllib.parse import urlparse

    t = title.strip()
    lower = t.lower()
    if not t:
        return False
    if not any(c.isalnum() for c in t):
        return False
    if lower in _GENERIC_TITLES:
        return False
    if t.replace(".", "").replace("-", "").isdigit():
        return False
    if re.match(r"^https?://", t):
        return False
    # image alt-text (![...)
    if t.startswith("!"):
        return False
    # title looks like a filename (e.g. "data.csv", "script.py")
    if _FILENAME_RE.match(t):
        return False
    # title contains LinkedIn activity IDs
    if "activity-" in t and re.search(r"activity-\d{10,}", t):
        return False
    # single-word titles (not informative enough on their own)
    if len(t.split()) == 1:
        return False
    # two-word titles where both words are short
    words = t.split()
    if len(words) == 2 and all(len(w) <= 5 for w in words):
        return False
    # URL-based checks
    if url:
        domain = urlparse(url).netloc
        if any(skip in domain for skip in _SKIP_DOMAINS):
            return False
        # image URLs
        path_lower = urlparse(url).path.lower()
        if any(path_lower.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".gif", ".svg")):
            return False
    return True


_CODE_BLOCK_RE = re.compile(r"```python\s*\n.*?```", re.DOTALL)
_LATEX_BLOCK_RE = re.compile(r"\$\$.*?\$\$", re.DOTALL)
_LATEX_INLINE_RE = re.compile(r"\$[^\$\n]+?\$")


def strip_code_blocks(text: str) -> str:
    """Remove fenced Python code blocks from markdown text."""
    return _CODE_BLOCK_RE.sub("", text)


def strip_latex(text: str) -> str:
    """Remove LaTeX formulas: multi-line $$...$$ and inline $...$."""
    text = _LATEX_BLOCK_RE.sub("", text)
    text = _LATEX_INLINE_RE.sub("", text)
    return text


def strip_urls(text: str) -> str:
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _BARE_URL_RE.sub("", text)
    return text


def chunk_document(path: Path, chunk_size: int, chunk_overlap: int) -> list[DocumentChunk]:
    content = read_text_file(path)
    if content is None:
        return []
    content = strip_code_blocks(content)
    content = strip_latex(content)
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


def show_links(input_dir: Path) -> None:
    """Walk *input_dir* for .md files, print every Markdown link, and save to links.jsonl."""
    entries = get_directory_tree(input_dir, extension=".md")
    if not entries:
        raise SystemExit(f"No .md files found under {input_dir}")

    output_path = input_dir / "links.jsonl"
    with output_path.open("w", encoding="utf-8") as fh:
        for entry in entries:
            md_path: Path = entry["input_path"]
            content = read_text_file(md_path)
            if content is None:
                continue
            links = _MD_LINK_WITH_URL_RE.findall(content)
            if not links:
                continue
            print(f"\n=== {md_path} ===")
            for title, url in links:
                print(f"  {title}  ->  {url}")
                record = {
                    "source": str(md_path),
                    "title": title,
                    "url": url,
                    "review_link": _is_informative_title(title, url),
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nLinks saved to {output_path}")


def build_index(input_dir: Path) -> None:
    """Scan input_dir for .md files and write a conversion_log.jsonl for ingestion."""
    entries = get_directory_tree(input_dir, extension=".md")
    if not entries:
        raise SystemExit(f"No .md files found under {input_dir}")

    log_path = input_dir / "conversion_log.jsonl"
    with log_path.open("w", encoding="utf-8") as fh:
        for entry in entries:
            md_path: Path = entry["input_path"]
            record = {
                "source_dir": str(md_path.parent),
                "source_file_name": md_path.name,
                "desctination_file": str(md_path),
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"  {md_path}")

    print(f"\nIndexed {len(entries)} files -> {log_path}")
    print(f'\nTo ingest into ChromaDB run:\n  DATA_DIR="$(pwd)" python src/rag/ingestion.py --log-path {log_path} --reset-collection')


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Chunk a Markdown file and print stats.")
    parser.add_argument("--input", help="Path to a .md file to chunk.")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size (default: 1000).")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap (default: 200).")
    parser.add_argument("--show-links", action="store_true", help="List all links in .md files.")
    parser.add_argument("--build-index", action="store_true", help="Build conversion_log.jsonl from .md files for ingestion.")
    parser.add_argument("--input-dir", help="Directory to scan for .md files.")
    args = parser.parse_args(argv)

    if args.build_index:
        if not args.input_dir:
            raise SystemExit("--input-dir is required when using --build-index")
        dir_path = Path(args.input_dir).expanduser()
        if not dir_path.is_dir():
            raise SystemExit(f"Directory not found: {dir_path}")
        build_index(dir_path)
        return

    if args.show_links:
        if not args.input_dir:
            raise SystemExit("--input-dir is required when using --show-links")
        dir_path = Path(args.input_dir).expanduser()
        if not dir_path.is_dir():
            raise SystemExit(f"Directory not found: {dir_path}")
        show_links(dir_path)
        return

    if not args.input:
        raise SystemExit("--input is required for chunking mode")

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
