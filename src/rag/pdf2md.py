#!/usr/bin/env python3
"""Convert PDF documents to Markdown using docling (via mindbase_layer.tools).

The .md output goes to the output directory (default: data/md_docs) so the
standard pipeline (files_processing.py --build-index + ingestion.py) can pick
it up. Each conversion appends an entry to the conversion log.

Usage examples::

    # Convert a single PDF
    PYTHONPATH="$(pwd)" python src/rag/pdf2md.py path/to/document.pdf

    # Convert all PDFs in a directory (recursive)
    PYTHONPATH="$(pwd)" python src/rag/pdf2md.py path/to/pdfs/

    # Convert into a custom output directory
    PYTHONPATH="$(pwd)" python src/rag/pdf2md.py path/to/document.pdf --output-dir /tmp/md_output
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Sequence

from files_processing import (
    DEFAULT_OUTPUT_DIR,
    get_directory_tree,
    log_entry_to_document_index,
)


def convert_pdf(input_path: Path, output_dir: Path) -> Path:
    """Convert a single PDF to Markdown placed in output_dir.

    mindbase_layer's pdf_to_md writes the .md next to the source PDF
    (plus an images directory), so we convert first, then move the result.
    """
    from mindbase_layer.tools import pdf_to_md

    print(f"Converting {input_path} ...")
    result = pdf_to_md(str(input_path))
    produced = input_path.parent / f"{input_path.stem}.md"
    if not produced.exists():
        raise RuntimeError(f"pdf_to_md did not produce {produced}: {result}")

    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / produced.name
    shutil.move(str(produced), target)
    return target


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        help="Path to a source .pdf file or a directory containing PDFs.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for converted Markdown files (default: data/md_docs).",
    )
    args = parser.parse_args(argv)

    input_path = Path(args.input).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    if not input_path.exists():
        raise SystemExit(f"Input path not found: {input_path}")

    if input_path.is_file():
        sources = [input_path]
    else:
        entries = get_directory_tree(input_path, extension=".pdf")
        sources = [entry["input_path"] for entry in entries]
        if not sources:
            raise SystemExit(f"No PDF files found under {input_path}")

    for source in sources:
        out_path = convert_pdf(source, output_dir)
        log_entry_to_document_index(source, out_path)
        print(f"Wrote Markdown to {out_path}")

    print(
        "\nTo build a clean index for ingestion run:\n"
        f'  PYTHONPATH="$(pwd)" python src/rag/files_processing.py --build-index --input-dir {output_dir}'
    )


if __name__ == "__main__":
    main()
