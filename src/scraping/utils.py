"""Shared helpers for this directory's blog scrapers (blog_feed_scrapers.py,
which parses saved MHTML snapshots, and blog_rss_scrapers.py, which fetches
live RSS feeds/HTML). Both produce the same three-field record shape
(BlogPost: source, link, date) and both benefit from the same URL
canonicalization, dedup, run journal, and logging setup - kept here once
instead of drifting between two copies.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

DEFAULT_JOURNAL_PATH = Path(os.environ.get("DATA_DIR", "data")) / "runs_journal.jsonl"
DEFAULT_EVENTS_LOG_PATH = Path(os.environ.get("DATA_DIR", "data")) / "events_log.jsonl"


@dataclass
class BlogPost:
    """One row of a scraper's output JSONL: source blog's index URL, one
    specific blogpost's canonical URL, its published date (ISO YYYY-MM-DD),
    and its title (best-effort; "" when a parser/fetch couldn't find one)."""

    source: str
    link: str
    date: str
    title: str = ""

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def dedupe_posts(posts: list[BlogPost]) -> list[BlogPost]:
    """Keep the first BlogPost seen for each unique link. Lets each parser
    build its raw list without tracking a `seen` set inline - call this once
    before returning."""
    seen: set[str] = set()
    deduped: list[BlogPost] = []
    for post in posts:
        if post.link in seen:
            continue
        seen.add(post.link)
        deduped.append(post)
    return deduped


def canonical(url: str, strip_query: bool = True) -> str:
    """Normalize a URL for dedup/storage: lowercase host, drop trailing
    slash, and either drop the query entirely (strip_query=True, the
    default) or keep it minus common tracking params (utm_*, source, trk)."""
    p = urlparse(url)
    q = ""
    if not strip_query:
        keep = [(k, v) for k, v in parse_qsl(p.query) if not k.startswith(("utm_", "source", "trk"))]
        q = urlencode(keep)
    path = p.path.rstrip("/") or "/"
    return urlunparse((p.scheme or "https", p.netloc.lower(), path, "", q, ""))


def configure_logging() -> None:
    """Call once at module import time in each script. Callers still use
    their own `logging.getLogger(__name__)` - this only sets the shared
    format/level so log lines look the same across both scripts."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def append_run(journal_path: Path = DEFAULT_JOURNAL_PATH, **fields) -> None:
    """Append one line to the shared run journal (default data/runs_journal.jsonl):
    {"run_time": <UTC ISO timestamp>, **fields}. Gives an auditable history of
    when each scraper last ran and what it produced."""
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"run_time": datetime.now(timezone.utc).isoformat(), **fields}
    with journal_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def push_event(path: Path, event: str, **fields) -> None:
    """Append one line to the shared events log (default data/events_log.jsonl):
    {"event": ..., "timestamp": <UTC ISO>, **fields}. Used by every scraper
    that reports individual, queryable occurrences (a scrape cycle starting,
    an mhtml snapshot landing, a blogpost being discovered) rather than just
    one summary row per run (that's what append_run/runs_journal.jsonl is for)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"event": event, "timestamp": datetime.now(timezone.utc).isoformat(), **fields}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_logged_links(path: Path = DEFAULT_EVENTS_LOG_PATH) -> set[str]:
    """Every link already recorded via a 'blog post' event, across all past
    runs of every scraper that pushes to this log - so no scraper (RSS-based
    or mhtml-based) ever emits a second event for the same link."""
    if not path.exists():
        return set()
    links: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("event") == "blog post" and "link" in entry:
                links.add(entry["link"])
    return links


def load_mhtml_ids(path: Path = DEFAULT_EVENTS_LOG_PATH) -> dict[str, str]:
    """Most recent mhtml_id per stem, from 'mhtml_retrieved' events (pushed by
    browser_use.py's feed() scenario) - lets blog_feed_scrapers.py attach a
    parent_uuid to each 'blog post' event it emits, linking it back to the
    specific browser capture it was parsed from."""
    ids: dict[str, str] = {}
    if not path.exists():
        return ids
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("event") == "mhtml_retrieved" and "stem" in entry and "mhtml_id" in entry:
                ids[entry["stem"]] = entry["mhtml_id"]
    return ids
