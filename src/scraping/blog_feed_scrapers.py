"""Per-site BeautifulSoup scrapers that turn saved MHTML blog-index snapshots into events.

Each record has four fields: ``source`` (the origin blog's index URL, as listed in
browser_sites.jsonl), ``link`` (URL of one specific blogpost), ``date`` (published date,
ISO ``YYYY-MM-DD``), and ``title`` (best-effort, "" if not found). See utils.BlogPost.

The mhtml-filename-prefix -> (source URL, parser name) registry lives in browser_sites.jsonl
next to this file (one JSON object per line: stem, source, parser); parser names are
resolved against the PARSERS dict below.

Every run appends one line to a shared journal (--journal, default
data/runs_journal.jsonl) recording how many sites/posts were scraped, and
pushes a 'blog post' event (--events, default data/events_log.jsonl) per
newly-seen link, with `parent_uuid` set to that mhtml file's `mhtml_id` (from
the 'mhtml_retrieved' event browser_use.py's feed() scenario pushed when it
captured the snapshot). events_log.jsonl - not a separate output file - is the
one place this data lives; see scraping_docs/blogposts.md for why a
once-planned separate blogposts.jsonl snapshot was dropped in favor of it.

Usage::

    PYTHONPATH="$(pwd)" python src/scraping/blog_feed_scrapers.py --mhtml-dir docs/mhtml --sites src/scraping/browser_sites.jsonl
"""
from __future__ import annotations

import argparse
import email
import json
import logging
import re
import uuid
from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path

from bs4 import BeautifulSoup

import metrics
from utils import (
    DEFAULT_EVENTS_LOG_PATH,
    DEFAULT_JOURNAL_PATH,
    BlogPost,
    append_run,
    canonical,
    configure_logging,
    dedupe_posts,
    load_logged_links,
    load_mhtml_ids,
    push_event,
)

configure_logging()
logger = logging.getLogger(__name__)

_MONTHS = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1
)}


def extract_mhtml_html(path: Path) -> str:
    """Return the decoded text/html part of a saved MHTML snapshot."""
    with path.open("rb") as f:
        msg = email.message_from_bytes(f.read())
    for part in msg.walk():
        if part.get_content_type() == "text/html":
            payload = part.get_payload(decode=True)
            charset = part.get_content_charset() or "utf-8"
            return payload.decode(charset, errors="replace")
    raise ValueError(f"No text/html part found in {path}")


def _abs_month_day(text: str, captured_at: date) -> str | None:
    m = re.match(r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) (\d{1,2})(?:,\s*(\d{4}))?$", text)
    if not m:
        return None
    month, day, year = m.group(1), int(m.group(2)), m.group(3)
    year = int(year) if year else captured_at.year
    return date(year, _MONTHS[month], day).isoformat()


def _relative_ago(text: str, captured_at: date) -> str | None:
    m = re.match(r"^(\d+)([dhm]) ago$", text)
    if not m:
        return None
    amount, unit = int(m.group(1)), m.group(2)
    if unit == "d":
        return (captured_at - timedelta(days=amount)).isoformat()
    return captured_at.isoformat()  # "h"/"m" ago is still today


_HEADING_RE = re.compile(r"^h[1-6]$")


def _extract_title(container, link_el) -> str:
    """Prefer a heading tag inside *container* (the card, or link_el itself
    when there's no separate card element) - most sites mark the title with
    an h1/h2/h3. Falls back to link_el's own text, which is the whole title
    on sites where the anchor wraps nothing but the title (no card markup).

    Uses find_all(string=True) rather than .get_text(): .get_text()/.strings
    silently drop text nested in <template> ancestors (bs4.TemplateString) -
    some lazy-loaded cards (e.g. Unsloth) wrap their title anchor that way."""
    el = container.find(_HEADING_RE) or link_el
    return "".join(el.find_all(string=True)).strip()


def parse_unsloth(html: str, source: str, captured_at: date) -> list[BlogPost]:
    soup = BeautifulSoup(html, "html.parser")
    posts: list[BlogPost] = []
    for card in soup.select("article.ub-card"):
        link_el = card.select_one("a.ub-title[href]")
        time_el = card.find("time")
        if not link_el or not time_el:
            continue
        if time_el.get("datetime"):
            iso_date = time_el["datetime"]
        else:
            # .get_text() silently drops text nested in <template> ancestors (bs4.TemplateString);
            # find_all(string=True) still returns it, unlike .strings/.get_text().
            text = "".join(time_el.find_all(string=True)).strip()
            try:
                iso_date = datetime.strptime(text, "%b %d, %Y").date().isoformat()
            except ValueError:
                continue
        posts.append(BlogPost(source=source, link=canonical(link_el["href"]), date=iso_date, title=_extract_title(card, link_el)))
    return dedupe_posts(posts)


def parse_anyscale(html: str, source: str, captured_at: date) -> list[BlogPost]:
    soup = BeautifulSoup(html, "html.parser")
    posts: list[BlogPost] = []
    date_re = re.compile(r"^\d{2}\.\d{2}\.\d{2}$")
    for card in soup.select("div.group.relative.flex.cursor-pointer"):
        link_el = card.find("a", href=True)
        date_el = card.find(string=date_re)
        if not link_el or not date_el:
            continue
        mm, dd, yy = date_el.strip().split(".")
        posts.append(BlogPost(source=source, link=canonical(link_el["href"]), date=f"20{yy}-{mm}-{dd}", title=_extract_title(card, link_el)))
    return dedupe_posts(posts)


def parse_lightning(html: str, source: str, captured_at: date) -> list[BlogPost]:
    soup = BeautifulSoup(html, "html.parser")
    posts: list[BlogPost] = []
    date_re = re.compile(r"[A-Z][a-z]+ \d{1,2}, \d{4}")
    href_re = re.compile(r"^https://lightning\.ai/blog/[^\s?]+$")
    for link_el in soup.find_all("a", href=href_re):
        date_text = link_el.find(string=date_re)
        if not date_text:
            continue
        m = date_re.search(date_text)
        dt = datetime.strptime(m.group(), "%B %d, %Y").date()
        posts.append(BlogPost(source=source, link=canonical(link_el["href"]), date=dt.isoformat(), title=_extract_title(link_el, link_el)))
    return dedupe_posts(posts)


def parse_hopsworks(html: str, source: str, captured_at: date) -> list[BlogPost]:
    soup = BeautifulSoup(html, "html.parser")
    posts: list[BlogPost] = []
    date_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    for link_el in soup.select("a.group.block[href]"):
        date_el = link_el.find(string=date_re)
        if not date_el:
            continue
        posts.append(BlogPost(source=source, link=canonical(link_el["href"]), date=date_el.strip(), title=_extract_title(link_el, link_el)))
    return dedupe_posts(posts)


def parse_instacart(html: str, source: str, captured_at: date) -> list[BlogPost]:
    soup = BeautifulSoup(html, "html.parser")
    posts: list[BlogPost] = []
    post_re = re.compile(r"^https://tech\.instacart\.com/[a-z0-9-]+-[0-9a-f]{12}")
    rel_re = re.compile(r"^\d+[dhm] ago$")
    abs_re = re.compile(r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{1,2}(,\s*\d{4})?$")
    for card in soup.find_all("article", attrs={"data-testid": "post-preview"}):
        link_el = card.find("a", href=post_re)
        if not link_el:
            continue
        date_text = card.find(string=rel_re) or card.find(string=abs_re)
        if not date_text:
            continue
        date_text = date_text.strip()
        iso_date = _relative_ago(date_text, captured_at) or _abs_month_day(date_text, captured_at)
        if not iso_date:
            continue
        posts.append(BlogPost(source=source, link=canonical(link_el["href"]), date=iso_date, title=_extract_title(card, link_el)))
    return dedupe_posts(posts)


def parse_nebius(html: str, source: str, captured_at: date) -> list[BlogPost]:
    soup = BeautifulSoup(html, "html.parser")
    posts: list[BlogPost] = []
    date_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    for link_el in soup.select("a.pc-post-card__card[href]"):
        date_el = link_el.find(string=date_re)
        if not date_el:
            continue
        posts.append(BlogPost(source=source, link=canonical(link_el["href"]), date=date_el.strip(), title=_extract_title(link_el, link_el)))
    return dedupe_posts(posts)


def parse_uber(html: str, source: str, captured_at: date) -> list[BlogPost]:
    soup = BeautifulSoup(html, "html.parser")
    posts: list[BlogPost] = []
    date_re = re.compile(r"^[A-Z][a-z]+ \d{1,2}, \d{4}$")
    for card in soup.find_all("div", attrs={"data-testid": "newsroom-article-feed-card"}):
        link_el = card.find("a", href=True)
        date_el = card.find(string=date_re)
        if not link_el or not date_el:
            continue
        dt = datetime.strptime(date_el.strip(), "%B %d, %Y").date()
        posts.append(BlogPost(source=source, link=canonical(link_el["href"]), date=dt.isoformat(), title=_extract_title(card, link_el)))
    return dedupe_posts(posts)


def parse_databricks(html: str, source: str, captured_at: date) -> list[BlogPost]:
    soup = BeautifulSoup(html, "html.parser")
    posts: list[BlogPost] = []
    date_re = re.compile(r"^[A-Z][a-z]+ \d{1,2}, \d{4}$")

    def add(container, link_el, date_el) -> None:
        if not link_el or not date_el:
            return
        dt = datetime.strptime(date_el.strip(), "%B %d, %Y").date()
        posts.append(BlogPost(source=source, link=canonical(link_el["href"]), date=dt.isoformat(), title=_extract_title(container, link_el)))

    featured = soup.find("article")
    if featured:
        add(featured, featured.find("a", href=True), featured.find(string=date_re))

    editors_picks = soup.find("aside")
    if editors_picks:
        for card in editors_picks.select("div.flex.flex-col.gap-1"):
            add(card, card.find("a", href=True), card.find(string=date_re))

    for li in soup.select("ul.border-gray-lines li"):
        add(li, li.find("a", href=True), li.find(string=date_re))

    return dedupe_posts(posts)


def parse_pinecone(html: str, source: str, captured_at: date) -> list[BlogPost]:
    soup = BeautifulSoup(html, "html.parser")
    posts: list[BlogPost] = []
    date_re = re.compile(r"^[A-Z][a-z]+ \d{1,2}, \d{4}$")
    href_re = re.compile(r"^https://www\.pinecone\.io/blog/[^\s?]+/?$")
    for link_el in soup.find_all("a", href=href_re):
        date_text = link_el.find(string=date_re)
        if not date_text:
            continue
        dt = datetime.strptime(date_text.strip(), "%b %d, %Y").date()
        # No heading tag and no separate card wrapper here - the anchor's
        # text is category+date+title+authors run together with no
        # separator, so the title is whatever string immediately follows
        # the date in document order.
        strings = list(link_el.stripped_strings)
        title = ""
        if date_text in strings:
            idx = strings.index(date_text)
            if idx + 1 < len(strings):
                title = strings[idx + 1]
        posts.append(BlogPost(source=source, link=canonical(link_el["href"]), date=dt.isoformat(), title=title))
    return dedupe_posts(posts)


# Name -> parser function, looked up when resolving browser_sites.json entries.
PARSERS: dict[str, Callable[[str, str, date], list[BlogPost]]] = {
    "parse_unsloth": parse_unsloth,
    "parse_anyscale": parse_anyscale,
    "parse_lightning": parse_lightning,
    "parse_hopsworks": parse_hopsworks,
    "parse_instacart": parse_instacart,
    "parse_nebius": parse_nebius,
    "parse_uber": parse_uber,
    "parse_databricks": parse_databricks,
    "parse_pinecone": parse_pinecone,
}

SITES_PATH = Path(__file__).with_name("browser_sites.jsonl")


def load_sites(path: Path = SITES_PATH) -> dict[str, tuple[str, Callable[[str, str, date], list[BlogPost]]]]:
    """Load the mhtml-filename-prefix -> (source URL, parser) registry from JSONL
    (one JSON object per line: stem, source, parser)."""
    sites: dict[str, tuple[str, Callable[[str, str, date], list[BlogPost]]]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            sites[entry["stem"]] = (entry["source"], PARSERS[entry["parser"]])
    return sites


def run_parse_cycle(
    mhtml_dir: Path,
    sites_path: Path = SITES_PATH,
    journal_path: Path = DEFAULT_JOURNAL_PATH,
    events_path: Path = DEFAULT_EVENTS_LOG_PATH,
) -> list[BlogPost]:
    """Parse every site's saved .mhtml snapshot, push a 'blog post' event per
    newly-seen link (parent_uuid = that mhtml's mhtml_id), and append one
    runs_journal row. Used by both main() (CLI) and feed.py (weekly
    orchestration alongside the RSS leg) - kept as one function so the two
    callers can't drift. No separate output file: events_log.jsonl is the
    only place this data lives (see blogposts.md)."""
    sites = load_sites(sites_path)
    mhtml_ids = load_mhtml_ids(events_path)
    seen_links = load_logged_links(events_path)
    all_posts: list[BlogPost] = []
    sites_scraped = 0

    for stem, (source, parse_fn) in sites.items():
        mhtml_path = mhtml_dir / f"{stem}.mhtml"
        if not mhtml_path.exists():
            logger.warning(f"Missing {mhtml_path}, skipping")
            continue
        captured_at = datetime.fromtimestamp(mhtml_path.stat().st_mtime).date()
        html = extract_mhtml_html(mhtml_path)
        posts = parse_fn(html, source, captured_at)
        logger.info(f"{stem}: {len(posts)} posts")
        all_posts.extend(posts)
        sites_scraped += 1

        mhtml_id = mhtml_ids.get(stem)
        if mhtml_id is None:
            logger.warning(f"No mhtml_retrieved event found for {stem}; logged posts will have parent_uuid=None")
        for post in posts:
            if post.link in seen_links:
                continue
            seen_links.add(post.link)
            push_event(
                events_path,
                "blog post",
                worker_name="mhtml_scraper",
                blog_post_id=str(uuid.uuid4()),
                parent_uuid=mhtml_id,
                link=post.link,
                source=post.source,
                date=post.date,
                title=post.title,
            )
            metrics.record_blog_post_found("mhtml_scraper")

    logger.info(f"Done. Parsed {len(all_posts)} posts from {sites_scraped}/{len(sites)} sites")
    append_run(
        journal_path,
        worker="blog_feed_scrapers",
        sites=sites_scraped,
        sites_total=len(sites),
        posts=len(all_posts),
    )
    return all_posts


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape blogpost links/dates from saved MHTML snapshots.")
    parser.add_argument("--mhtml-dir", default="docs/mhtml", help="Directory with saved .mhtml files.")
    parser.add_argument("--sites", default=str(SITES_PATH), help="Path to the sites JSONL registry.")
    parser.add_argument("--journal", default=str(DEFAULT_JOURNAL_PATH), help="Path to the shared run journal JSONL.")
    parser.add_argument("--events", default=str(DEFAULT_EVENTS_LOG_PATH), help="Path to the shared events log JSONL.")
    args = parser.parse_args()

    run_parse_cycle(
        Path(args.mhtml_dir),
        Path(args.sites),
        Path(args.journal),
        Path(args.events),
    )


if __name__ == "__main__":
    main()
