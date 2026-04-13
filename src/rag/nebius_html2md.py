"""Convert downloaded HTML pages to Markdown.

Usage:
    DATA_DIR="$(pwd)" python src/rag/nebius_html2md.py --input data/nebius_site/
"""

import argparse
import fnmatch
import hashlib
from pathlib import Path

from files_processing import (
    convert_file,
    get_directory_tree,
    log_entry_to_document_index,
)
from markdownify import markdownify as md


SKIP_PATTERNS = [
    "llms_txt.html",
    "legal_*.html",
    "*billing*.html",
]


def should_skip(filename: str) -> bool:
    """Check if a file matches any blacklist pattern."""
    return any(fnmatch.fnmatch(filename, pat) for pat in SKIP_PATTERNS)


def extract_content_area(html: str) -> str:
    """Extract the main content div (id='content-area') from full page HTML."""
    marker = 'id="content-area"'
    idx = html.find(marker)
    if idx == -1:
        return html

    # Walk back to find the opening tag start
    tag_start = html.rfind("<", 0, idx)
    # Find matching closing div by counting nesting
    depth = 0
    i = tag_start
    while i < len(html):
        if html[i:i+4] == "<div":
            depth += 1
        elif html[i:i+6] == "</div>":
            depth -= 1
            if depth == 0:
                return html[tag_start:i + 6]
        i += 1
    # Fallback: return from marker to end
    return html[tag_start:]

def convert_html(html_path: Path) -> str:
    html_text = html_path.read_text(encoding="utf-8")
    content_html = extract_content_area(html_text)
    markdown_text = md(content_html, heading_style="ATX")
    return markdown_text

def main():
    parser = argparse.ArgumentParser(description="Convert downloaded HTML pages to Markdown")
    parser.add_argument("--input", required=True, help="Path to directory with HTML files")
    args = parser.parse_args()

    input_dir = Path(args.input).expanduser()
    if not input_dir.is_dir():
        raise SystemExit(f"Directory not found: {input_dir}")

    entries = get_directory_tree(input_dir, extension=".html")
    if not entries:
        raise SystemExit(f"No HTML files found under {input_dir}")

    dir_hash = hashlib.md5(str(input_dir).encode()).hexdigest()[:8]
    log_path = input_dir / f"conversion_log_{dir_hash}.jsonl"

    skipped = [e for e in entries if should_skip(e["input_path"].name)]
    entries = [e for e in entries if not should_skip(e["input_path"].name)]
    for e in skipped:
        print(f"  SKIP {e['input_path'].name} (blacklisted)")
    print(f"Found {len(entries)} HTML files ({len(skipped)} skipped by blacklist)")
    for entry in entries:
        source_path, out_path = convert_file(entry["input_path"], entry["input_path"].with_suffix(".md"), convert_html)
        log_entry_to_document_index(source_path, out_path, log_path=log_path)
        print(f"  {source_path.name} -> {out_path.name}")

    print(f"\nConversion log: {log_path}")
    print(f'\nTo ingest into ChromaDB run:\n  DATA_DIR="$(pwd)" python src/rag/ingestion.py --log-path {log_path} --reset-collection')


if __name__ == "__main__":
    main()
