#!/usr/bin/env python3
"""
blog_rss_scrapers.py — collect recent posts from corporate tech blogs into JSONL.

Output: one JSON object per line, same three-field shape as blog_feed_scrapers.py
(see utils.BlogPost): source (blog index URL), link (canonical article URL),
date (ISO YYYY-MM-DD).

Usage:
    pip install requests feedparser beautifulsoup4 python-dateutil
    python blog_rss_scrapers.py                         # last 14 days -> articles.jsonl
    python blog_rss_scrapers.py --days 30 -o out.jsonl  # custom window / output
    python blog_rss_scrapers.py --only "Pinterest Engineering" "Netflix Tech Blog"
    python blog_rss_scrapers.py --list                  # print configured sources

Strategy per source (in order):
  1. RSS/Atom feed (explicit `feed`, or auto-discovered from the index page)
  2. Fallback: scrape the blog index for article links, open each article
     and read its publish date from <meta>/JSON-LD/<time>. Each article's
     date is cached (--article-cache, default data/article_date_cache.json)
     so a re-run only fetches articles not already dated.

Every run appends one line to a shared journal (--journal, default
data/runs_journal.jsonl) recording sources attempted/failed and rows written.
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import feedparser
import requests
from bs4 import BeautifulSoup
from dateutil import parser as dateparser

from utils import (
    DEFAULT_JOURNAL_PATH,
    BlogPost,
    append_run,
    canonical,
    configure_logging,
    dedupe_posts,
)

configure_logging()
log = logging.getLogger(__name__)

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
TIMEOUT = 20
SLEEP = 0.5            # politeness delay between requests to the same host
MAX_ARTICLES = 25      # max article pages opened per source in HTML fallback

# HTML-fallback strategy re-derives an article's publish date by fetching its
# page - this cache (canonical article URL -> ISO date) means only articles
# not already dated in a prior run get fetched at all.
ARTICLE_DATE_CACHE_PATH = Path(os.environ.get("DATA_DIR", "data")) / "article_date_cache.json"


# --------------------------------------------------------------------------
# Source registry — loaded from blog_sources.jsonl (one JSON object per line:
# name, url, feed, link_pattern, strip_query)
# --------------------------------------------------------------------------
@dataclass
class Source:
    name: str
    url: str                       # blog index page
    feed: str | None = None     # known RSS/Atom feed
    link_pattern: str | None = None  # regex an article URL must match (HTML fallback)
    strip_query: bool = True


SOURCES_PATH = Path(__file__).with_name("blog_sources.jsonl")


def load_sources(path: Path = SOURCES_PATH) -> list[Source]:
    sources = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sources.append(Source(**json.loads(line)))
    return sources

# --------------------------------------------------------------------------
# HTTP helpers
# --------------------------------------------------------------------------
_session = requests.Session()
_session.headers.update({
    "User-Agent": UA,
    "Accept-Language": "en-US,en;q=0.8",
    # Some edges (e.g. Uber's) return 406 Not Acceptable to a request with no
    # Accept header at all - every real browser sends one, so this is a safe
    # global default rather than a per-source special case.
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
})
_last_hit: dict[str, float] = {}


def get(url: str) -> requests.Response | None:
    host = urlparse(url).netloc
    wait = SLEEP - (time.time() - _last_hit.get(host, 0))
    if wait > 0:
        time.sleep(wait)
    for attempt in range(3):
        try:
            r = _session.get(url, timeout=TIMEOUT, allow_redirects=True)
        except requests.RequestException:
            # Transient (timeout, connection reset) - retry with backoff
            # rather than giving up after a single hiccup, same as the 429
            # case below. A source silently producing 0 items for an entire
            # cycle over one dropped connection is worse than a short delay.
            time.sleep(2 * (attempt + 1))
            continue
        _last_hit[host] = time.time()
        if r.status_code == 429:          # Substack & co. rate-limit bursts
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code >= 400:
            return None
        return r
    return None


def to_date(value) -> str | None:
    """Parse anything date-like into YYYY-MM-DD (UTC)."""
    if not value:
        return None
    if isinstance(value, time.struct_time):
        return time.strftime("%Y-%m-%d", value)
    try:
        dt = dateparser.parse(str(value), fuzzy=True)
    except (ValueError, OverflowError, TypeError):
        return None
    if dt is None:
        return None
    if dt.tzinfo:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%d")


# --------------------------------------------------------------------------
# Strategy 1: RSS / Atom
# --------------------------------------------------------------------------
def discover_feed(index_url: str) -> str | None:
    r = get(index_url)
    if not r:
        return None
    soup = BeautifulSoup(r.text, "html.parser")
    for link in soup.find_all("link", rel=lambda v: v and "alternate" in v):
        t = (link.get("type") or "").lower()
        if "rss" in t or "atom" in t or "xml" in t:
            return urljoin(index_url, link.get("href"))
    # common conventions
    for cand in ("feed", "feed/", "rss", "rss.xml", "feed.xml", "index.xml", "atom.xml"):
        u = urljoin(index_url.rstrip("/") + "/", cand)
        rr = get(u)
        if rr and ("<rss" in rr.text[:2000] or "<feed" in rr.text[:2000]):
            return u
    return None


def from_feed(src: Source, feed_url: str) -> list[BlogPost]:
    r = get(feed_url)
    if not r:
        return []
    parsed = feedparser.parse(r.content)
    if not parsed.entries:
        return []
    out = []
    for e in parsed.entries:
        date = to_date(e.get("published_parsed") or e.get("updated_parsed")
                       or e.get("published") or e.get("updated"))
        link = e.get("link")
        if not (date and link):
            continue
        title = (e.get("title") or "").strip()
        out.append(BlogPost(source=src.url, link=canonical(link, src.strip_query), date=date, title=title))
    return out


def load_article_date_cache(path: Path = ARTICLE_DATE_CACHE_PATH) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        log.warning(f"Malformed article date cache at {path}, starting fresh")
        return {}


def save_article_date_cache(cache: dict[str, str], path: Path = ARTICLE_DATE_CACHE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------
# Strategy 2: HTML index -> article pages -> date from metadata
# --------------------------------------------------------------------------
DATE_META = [
    ("meta", {"property": "article:published_time"}),
    ("meta", {"name": "article:published_time"}),
    ("meta", {"property": "og:published_time"}),
    ("meta", {"name": "publish_date"}),
    ("meta", {"name": "publishdate"}),
    ("meta", {"name": "date"}),
    ("meta", {"name": "dc.date"}),
    ("meta", {"name": "DC.date.issued"}),
    ("meta", {"itemprop": "datePublished"}),
    ("meta", {"name": "parsely-pub-date"}),
    ("meta", {"name": "sailthru.date"}),
]


def date_from_html(html: str) -> str | None:
    soup = BeautifulSoup(html, "html.parser")
    for tag, attrs in DATE_META:
        m = soup.find(tag, attrs=attrs)
        if m and m.get("content"):
            d = to_date(m["content"])
            if d:
                return d
    # JSON-LD
    for s in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(s.string or "")
        except (json.JSONDecodeError, TypeError):
            continue
        stack = [data]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                for k in ("datePublished", "dateCreated", "uploadDate"):
                    if node.get(k):
                        d = to_date(node[k])
                        if d:
                            return d
                stack.extend(node.values())
            elif isinstance(node, list):
                stack.extend(node)
    # <time datetime="...">
    t = soup.find("time", attrs={"datetime": True})
    if t:
        d = to_date(t["datetime"])
        if d:
            return d
    return None


def title_from_html(html: str) -> str | None:
    soup = BeautifulSoup(html, "html.parser")
    m = soup.find("meta", attrs={"property": "og:title"}) or soup.find("meta", attrs={"name": "twitter:title"})
    if m and m.get("content"):
        return m["content"].strip()
    h1 = soup.find("h1")
    if h1:
        t = h1.get_text(strip=True)
        if t:
            return t
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    return None


def article_links(src: Source, html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    pat = re.compile(src.link_pattern) if src.link_pattern else None
    base_host = urlparse(src.url).netloc
    seen, links = set(), []
    for a in soup.find_all("a", href=True):
        u = canonical(urljoin(src.url, a["href"]), src.strip_query)
        if urlparse(u).netloc.replace("www.", "") != base_host.replace("www.", ""):
            continue
        if u == canonical(src.url):
            continue
        if pat and not pat.search(u):
            continue
        if u not in seen:
            seen.add(u)
            links.append(u)
    return links[:MAX_ARTICLES]


def from_html(src: Source, cache: dict[str, str]) -> list[BlogPost]:
    r = get(src.url)
    if not r:
        return []
    out = []
    for u in article_links(src, r.text):
        cached = cache.get(u)
        # A cache entry only counts as "resolved" once it has a title too -
        # entries written before `title` was added are a plain date string
        # (or a dict with no title), and treating those as fully cached would
        # permanently skip title extraction for every URL crawled before this
        # field existed. Re-fetching once self-heals the cache; after that
        # it's a normal cache hit with no further fetches.
        cached_title = cached.get("title") if isinstance(cached, dict) else None
        if cached_title:
            d, title = cached["date"], cached_title
        else:
            ar = get(u)
            if not ar:
                # Can't re-fetch right now - fall back to whatever date we
                # already have (title stays empty) rather than dropping the
                # post entirely.
                d = cached.get("date") if isinstance(cached, dict) else cached
                if d is None:
                    continue
                title = ""
            else:
                d = date_from_html(ar.text)
                if not d:
                    continue
                title = title_from_html(ar.text) or ""
                cache[u] = {"date": d, "title": title}
        out.append(BlogPost(source=src.url, link=u, date=d, title=title))
    return out


# --------------------------------------------------------------------------
def crawl(src: Source, cache: dict[str, str]) -> list[BlogPost]:
    items: list[BlogPost] = []
    feed = src.feed or discover_feed(src.url)
    if feed:
        items = from_feed(src, feed)
        if items:
            log.info(f"  [feed] {src.name}: {len(items)} items via {feed}")
    if not items:
        items = from_html(src, cache)
        log.info(f"  [html] {src.name}: {len(items)} items")

    # A few sources schedule/backdate an article's publish_time meta ahead of
    # today (e.g. an embargoed announcement). Keep the post rather than drop
    # it - a future-dated record is still real content worth having, and the
    # date field itself (visibly > today) is the flag; a downstream reader
    # can decide to hold or verify it rather than lose it outright.
    today = datetime.now(timezone.utc).date().isoformat()
    future = [p for p in items if p.date > today]
    if future:
        log.warning(f"  {src.name}: {len(future)} future-dated post(s) (publish_date > today): "
                    + ", ".join(p.link for p in future))
    return items


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=14, help="keep posts from the last N days (default 14)")
    ap.add_argument("--since", help="alternative to --days: keep posts on/after YYYY-MM-DD")
    ap.add_argument("-o", "--output", default="articles.jsonl")
    ap.add_argument("--sources", default=str(SOURCES_PATH), help="path to the sources JSONL registry")
    ap.add_argument("--only", nargs="*", help="crawl only these source names")
    ap.add_argument("--list", action="store_true", help="list sources and exit")
    ap.add_argument("--all-dates", action="store_true", help="do not filter by date")
    ap.add_argument("--journal", default=str(DEFAULT_JOURNAL_PATH), help="path to the shared run journal JSONL")
    ap.add_argument("--article-cache", default=str(ARTICLE_DATE_CACHE_PATH),
                     help="path to the article-date cache used by the HTML fallback")
    args = ap.parse_args(argv)

    all_sources = load_sources(Path(args.sources))

    if args.list:
        for s in all_sources:
            print(f"{s.name:35s} {s.feed or s.url}")
        return 0

    since = (args.since if args.since
             else (datetime.now(timezone.utc) - timedelta(days=args.days)).strftime("%Y-%m-%d"))
    sources = [s for s in all_sources if not args.only or s.name in set(args.only)]

    cache_path = Path(args.article_cache)
    cache = load_article_date_cache(cache_path)
    cache_size_before = len(cache)

    all_posts: list[BlogPost] = []
    sources_failed = 0
    for src in sources:
        log.info(f"* {src.name}")
        try:
            all_posts.extend(crawl(src, cache))
        except Exception as exc:  # keep going on any single-source failure
            log.error(f"  !! {src.name}: {exc}")
            sources_failed += 1

    if not args.all_dates:
        all_posts = [p for p in all_posts if p.date >= since]
    all_posts.sort(key=lambda p: p.date, reverse=True)
    all_posts = dedupe_posts(all_posts)

    with open(args.output, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(post.as_dict(), ensure_ascii=False) + "\n" for post in all_posts)

    save_article_date_cache(cache, cache_path)
    total = len(all_posts)
    log.info(f"wrote {total} records since {since} -> {args.output}")
    append_run(
        Path(args.journal),
        worker="blog_rss_scrapers",
        sources=len(sources),
        sources_failed=sources_failed,
        written=total,
        since=since,
        output=args.output,
        article_cache_size=len(cache),
        article_cache_new=len(cache) - cache_size_before,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())