"""Open links from a Markdown file in CloakBrowser and save each page as Markdown.

Usage::

    PYTHONPATH="$(pwd)" python src/scraping/browser_use.py --scenario explore --input slides/ml_breadth_reading_list.md --output-dir data/md_docs
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import time
from pathlib import Path

from cloakbrowser import launch_persistent_context

try:
    from patchright.sync_api import TimeoutError  # CloakBrowser >= 0.3.0
except ImportError:
    from playwright.sync_api import TimeoutError  # fallback

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

_MD_LINK_WITH_URL_RE = re.compile(r"\[([^\]]*)\]\((https?://[^\)]*)\)")


def extract_links_from_md(path: Path) -> list[tuple[str, str]]:
    """Return (title, url) pairs from Markdown links in *path*."""
    content = path.read_text(encoding="utf-8")
    links = _MD_LINK_WITH_URL_RE.findall(content)
    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for title, url in links:
        url_clean = url.rstrip("/")
        if url_clean in seen or "linkedin.com" in url_clean or "github.com" in url_clean:
            continue
        seen.add(url_clean)
        unique.append((title, url))
    return unique


def sanitize_filename(text: str) -> str:
    name = re.sub(r"[^\w\s-]", "", text)
    name = re.sub(r"[\s-]+", "_", name).strip("_")
    return name[:120] or "page"


def page_to_markdown(page) -> str:
    """Extract page content as Markdown-ish text via innerText."""
    title = page.title() or ""
    body = page.evaluate("() => document.body?.innerText || ''")
    parts = []
    if title:
        parts.append(f"# {title}\n")
    parts.append(body)
    return "\n".join(parts)


def explore(input_path: Path, output_dir: Path) -> None:
    links = extract_links_from_md(input_path)
    if not links:
        raise SystemExit(f"No links found in {input_path}")

    logger.info(f"Found {len(links)} unique links in {input_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    profile_dir = os.path.join(os.path.dirname(__file__), "profile")
    context = launch_persistent_context(profile_dir, headless=False, humanize=True)
    page = context.pages[0] if context.pages else context.new_page()

    page.on("console", lambda msg: logger.debug(f"[browser console] {msg.type}: {msg.text}"))
    page.on("pageerror", lambda err: logger.warning(f"[browser error] {err}"))
    page.on("framenavigated", lambda frame: logger.info(f"[browser nav] {frame.url}") if frame == page.main_frame else None)

    for i, (title, url) in enumerate(links, 1):
        fname = f"{i:03d}_{sanitize_filename(title)}.md"
        dest = output_dir / fname
        if dest.exists():
            logger.info(f"[{i}/{len(links)}] EXISTS {dest.name}")
            continue

        logger.info(f"[{i}/{len(links)}] {url}")
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            time.sleep(3)
            md_content = page_to_markdown(page)
            dest.write_text(md_content, encoding="utf-8")
            logger.info(f"  Saved {dest.name} ({len(md_content)} chars)")
        except Exception as e:
            logger.error(f"  FAILED {url}: {e}")

        time.sleep(2)

    logger.info(f"Done. Saved pages to {output_dir}")
    context.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Browse links from a Markdown file and save pages.")
    parser.add_argument("--scenario", required=True, choices=["explore"], help="Scenario to run.")
    parser.add_argument("--input", required=True, help="Path to .md file with links.")
    parser.add_argument("--output-dir", default=None, help="Directory to save output .md files (default: data/md_docs).")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    if not input_path.exists():
        raise SystemExit(f"File not found: {input_path}")

    output_dir = Path(args.output_dir) if args.output_dir else Path(os.environ.get("DATA_DIR", "data")) / "md_docs"

    if args.scenario == "explore":
        explore(input_path, output_dir)


if __name__ == "__main__":
    main()
