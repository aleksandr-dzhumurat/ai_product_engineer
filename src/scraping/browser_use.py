"""Open links in CloakBrowser and save each page as Markdown or MHTML.

Usage::

    PYTHONPATH="$(pwd)" python src/scraping/browser_use.py --scenario explore --input slides/ml_breadth_reading_list.md --output-dir data/md_docs
    PYTHONPATH="$(pwd)" python src/scraping/browser_use.py --scenario feed
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
import uuid
from pathlib import Path

from cloakbrowser import launch_persistent_context

import metrics
from utils import DEFAULT_EVENTS_LOG_PATH, configure_logging, push_event

configure_logging()
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


DEFAULT_SITES_PATH = Path(__file__).with_name("browser_sites.jsonl")


def load_browser_sites(path: Path = DEFAULT_SITES_PATH) -> list[tuple[str, str]]:
    """Return (stem, source URL) pairs from browser_sites.jsonl, in file order -
    the same registry blog_feed_scrapers.py uses to know which parser to run
    on each stem's .mhtml. Single source of truth: previously feed() re-derived
    its own stems from docs/scraping.md's table order/names, which could
    silently drift from browser_sites.jsonl's hand-written stems."""
    rows: list[tuple[str, str]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            rows.append((entry["stem"], entry["source"]))
    return rows


def save_mhtml(page, dest: Path) -> None:
    client = page.context.new_cdp_session(page)
    result = client.send("Page.captureSnapshot", {"format": "mhtml"})
    dest.write_text(result["data"], encoding="utf-8")


def page_to_markdown(page) -> str:
    """Extract page content as Markdown-ish text via innerText."""
    title = page.title() or ""
    body = page.evaluate("() => document.body?.innerText || ''")
    parts = []
    if title:
        parts.append(f"# {title}\n")
    parts.append(body)
    return "\n".join(parts)


def explore(input_path: Path, output_dir: Path, headless: bool = False) -> None:
    links = extract_links_from_md(input_path)
    if not links:
        raise SystemExit(f"No links found in {input_path}")

    logger.info(f"Found {len(links)} unique links in {input_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    profile_dir = os.path.join(os.path.dirname(__file__), "profile")
    context = launch_persistent_context(profile_dir, headless=headless, humanize=True)
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


def feed(
    output_dir: Path,
    headless: bool = False,
    sites_path: Path = DEFAULT_SITES_PATH,
    force: bool = False,
    run_id: str | None = None,
) -> None:
    """force=True recaptures even if the .mhtml already exists (used by
    feed.py's weekly orchestration, which wants fresh data every cycle rather
    than the skip-if-exists behavior standalone/manual runs default to).
    run_id, when given, is attached to each 'mhtml_retrieved' event as
    parent_run, tying it back to the overall scrape cycle that triggered it."""
    rows = load_browser_sites(sites_path)
    if not rows:
        raise SystemExit(f"No sites found in {sites_path}")

    logger.info(f"Found {len(rows)} sites in {sites_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    profile_dir = os.path.join(os.path.dirname(__file__), "profile")
    context = launch_persistent_context(profile_dir, headless=headless, humanize=True)
    page = context.pages[0] if context.pages else context.new_page()

    page.on("console", lambda msg: logger.debug(f"[browser console] {msg.type}: {msg.text}"))
    page.on("pageerror", lambda err: logger.warning(f"[browser error] {err}"))
    page.on("framenavigated", lambda frame: logger.info(f"[browser nav] {frame.url}") if frame == page.main_frame else None)

    for i, (stem, url) in enumerate(rows, 1):
        dest = output_dir / f"{stem}.mhtml"
        if dest.exists() and not force:
            logger.info(f"[{i}/{len(rows)}] EXISTS {dest.name}")
            continue

        logger.info(f"[{i}/{len(rows)}] {url}")
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            time.sleep(3)
            page.mouse.wheel(0, 800)
            time.sleep(1)
            # Headless runs (e.g. on a server) skip the screenshot entirely - it's a
            # convenience for eyeballing a headed capture locally, not needed otherwise.
            if headless:
                screenshot_dest = None
            else:
                screenshot_dest = dest.with_suffix(".png")
                page.screenshot(path=str(screenshot_dest))
            save_mhtml(page, dest)
            mhtml_id = str(uuid.uuid4())
            event_fields = {
                "worker_name": "browser_scraper",
                "mhtml_id": mhtml_id,
                "stem": stem,
                "url": url,
                "mhtml_path": str(dest),
            }
            if run_id:
                event_fields["parent_run"] = run_id
            push_event(DEFAULT_EVENTS_LOG_PATH, "mhtml_retrieved", **event_fields)
            metrics.record_mhtml_capture(stem)
            saved_msg = f"  Saved {dest.name}" + (f" and {screenshot_dest.name}" if screenshot_dest else "")
            logger.info(f"{saved_msg} (mhtml_id={mhtml_id})")
        except Exception as e:
            logger.error(f"  FAILED {url}: {e}")

        time.sleep(2)

    logger.info(f"Done. Saved MHTML pages to {output_dir}")
    context.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Browse links and save pages via CloakBrowser.")
    parser.add_argument("--scenario", required=True, choices=["explore", "feed"], help="Scenario to run.")
    parser.add_argument("--input", default=None, help="Path to .md file with links (explore scenario only).")
    parser.add_argument("--output-dir", default=None, help="Directory to save output files (default: data/md_docs for explore, docs/mhtml for feed).")
    parser.add_argument("--sites", default=str(DEFAULT_SITES_PATH), help="Path to the sites JSONL registry (feed scenario only).")
    parser.add_argument("--force", action="store_true", help="Recapture even if the .mhtml already exists (feed scenario only).")
    parser.add_argument("--headless", action="store_true", help="Run the browser with no visible window (needed on servers without a display).")
    args = parser.parse_args()

    if args.scenario == "explore":
        if not args.input:
            raise SystemExit("--input is required for scenario 'explore'")
        input_path = Path(args.input).expanduser()
        if not input_path.exists():
            raise SystemExit(f"File not found: {input_path}")
        output_dir = Path(args.output_dir) if args.output_dir else Path(os.environ.get("DATA_DIR", "data")) / "md_docs"
        explore(input_path, output_dir, headless=args.headless)
    elif args.scenario == "feed":
        output_dir = Path(args.output_dir) if args.output_dir else Path("docs/mhtml")
        feed(output_dir, headless=args.headless, sites_path=Path(args.sites), force=args.force)


if __name__ == "__main__":
    main()
