#!/usr/bin/env python3
"""
Convert a Jupyter Notebook (.ipynb) into a Markdown document saved under data/md_docs/.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable, List

from text_processing import record_conversion  # noqa: E402


def _to_text(source: Any) -> str:
    if isinstance(source, list):
        return "".join(str(part) for part in source)
    if source is None:
        return ""
    return str(source)


def _language_from_metadata(nb_root: dict[str, Any]) -> str:
    kernelspec = nb_root.get("metadata", {}).get("kernelspec", {})
    language = kernelspec.get("language") or kernelspec.get("name") or ""
    language = str(language).strip().lower()
    if not language:
        return "text"
    # Normalize a few known kernel ids.
    mapping = {
        "python3": "python",
        "python2": "python",
        "python": "python",
        "ir": "r",
        "julia-1.0": "julia",
    }
    return mapping.get(language, language)


def _format_outputs(outputs: Iterable[dict[str, Any]]) -> List[str]:
    blocks: List[str] = []
    for output in outputs:
        output_type = output.get("output_type")
        if output_type == "stream":
            text = _to_text(output.get("text"))
            if text:
                blocks.append("```text\n" + text.rstrip("\n") + "\n```")
        elif output_type in {"execute_result", "display_data"}:
            data = output.get("data", {})
            if "text/markdown" in data:
                blocks.append(_to_text(data["text/markdown"]).strip())
            elif "text/plain" in data:
                text = _to_text(data["text/plain"])
                if text:
                    blocks.append("```text\n" + text.rstrip("\n") + "\n```")
        elif output_type == "error":
            traceback = output.get("traceback") or []
            text = "\n".join(traceback)
            if text:
                blocks.append("```text\n" + text.rstrip("\n") + "\n```")
    return blocks


def convert_notebook(input_path: Path) -> str:
    with input_path.open("r", encoding="utf-8") as fh:
        notebook = json.load(fh)

    language = _language_from_metadata(notebook)
    md_sections: List[str] = []

    for cell in notebook.get("cells", []):
        cell_type = cell.get("cell_type")
        if cell_type == "markdown":
            md_sections.append(_to_text(cell.get("source")).rstrip())
        elif cell_type == "code":
            source = _to_text(cell.get("source")).rstrip()
            if source:
                md_sections.append(f"```{language}\n{source}\n```")
            output_blocks = _format_outputs(cell.get("outputs", []))
            md_sections.extend(output_blocks)
        elif cell_type == "raw":
            md_sections.append(_to_text(cell.get("source")).rstrip())

    # Filter out leading/trailing empty sections.
    trimmed = [section for section in (section.strip("\n") for section in md_sections) if section]
    return "\n\n".join(trimmed) + ("\n" if trimmed else "")


def target_markdown_path(input_path: Path) -> Path:
    output_dir = Path("data/md_docs")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / (input_path.stem + ".md")


def convert_file(input_path: Path) -> Path:
    markdown = convert_notebook(input_path)
    output_path = target_markdown_path(input_path)
    output_path.write_text(markdown, encoding="utf-8")
    record_conversion(input_path, output_path)
    return output_path


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


def process_directory(directory: Path) -> List[Path]:
    converted: List[Path] = []
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
            if not name.endswith(".ipynb"):
                continue
            source_path = Path(root) / name
            converted.append(convert_file(source_path))
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        help="Path to the source .ipynb notebook or directory containing notebooks.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    if not input_path.exists():
        raise SystemExit(f"Input path not found: {input_path}")

    if input_path.is_dir():
        results = process_directory(input_path)
        if not results:
            print(f"No notebooks found under {input_path}")
        else:
            for path in results:
                print(f"Wrote Markdown to {path}")
    elif input_path.is_file():
        output_path = convert_file(input_path)
        print(f"Wrote Markdown to {output_path}")
    else:
        raise SystemExit(f"Unsupported input type: {input_path}")


if __name__ == "__main__":
    main()
