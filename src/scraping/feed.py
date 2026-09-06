"""Long-running scheduler loop: once a week (Tuesday at 09:15 local time), runs
two independent scraping legs and appends one JSON event per run and one per
discovered blogpost to events_log.jsonl:

  1. RSS/HTML leg (blog_rss_scrapers.crawl, one source at a time)
  2. Browser/mhtml leg (browser_use.feed in headless mode, forced fresh each
     cycle, then blog_feed_scrapers.run_parse_cycle to parse the snapshots)

A post's `link` is unique across the whole log, regardless of which leg found
it - a link already recorded by a past 'blog post' event is skipped, never
re-logged. The mhtml leg is wrapped so a CloakBrowser failure (e.g. missing
browser binary) degrades gracefully instead of losing the RSS leg's results.

A background thread (digest_watchdog_loop) separately watches for a scrape
cycle finishing (the 'scraping run complete' event) and auto-generates +
sends the Telegram digest for it - same generation/send path as --generate,
just triggered automatically instead of run by hand.

Usage::

    PYTHONPATH="$(pwd)" python src/scraping/feed.py
    PYTHONPATH="$(pwd)" python src/scraping/feed.py --force     # run one cycle now, then resume the weekly wait loop
    PYTHONPATH="$(pwd)" python src/scraping/feed.py --generate  # build a digest, send it to Telegram, print it, and exit (needs NEBIUS_API_KEY, TG_BOT_TOKEN)
"""
from __future__ import annotations

import argparse
import json
import logging
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import browser_use
import llm_client
import metrics
import tg_pusher
from blog_feed_scrapers import SITES_PATH as MHTML_SITES_PATH
from blog_feed_scrapers import run_parse_cycle
from blog_rss_scrapers import (
    ARTICLE_DATE_CACHE_PATH,
    SOURCES_PATH,
    crawl,
    load_article_date_cache,
    load_sources,
    save_article_date_cache,
)

from utils import (
    DEFAULT_EVENTS_LOG_PATH,
    configure_logging,
    load_logged_links,
    push_event,
)

configure_logging()
logger = logging.getLogger(__name__)

SLEEP_SECONDS = 60 * 60  # digest_watchdog_loop's poll interval (also its "was the finished run recent" window, see there)
TRIGGER_POLL_SECONDS = 60  # main()'s scheduler loop poll interval - must be <= 60s to land on TARGET_MINUTE
TARGET_WEEKDAY = 1  # datetime.weekday(): Monday=0 ... Friday=4
TARGET_HOUR = 9      # local time
TARGET_MINUTE = 15

# Self-contained: the mhtml leg's captures land next to the other
# DATA_DIR-relative files (events_log.jsonl, runs_journal.jsonl, ...) rather
# than the docs/ layout browser_use.py defaults to when run standalone from
# the repo root.
MHTML_DIR = DEFAULT_EVENTS_LOG_PATH.parent / "mhtml"


def load_recent_blog_posts(path: Path = DEFAULT_EVENTS_LOG_PATH, days: int = 14) -> list[dict]:
    """Read-only: load events_log.jsonl and return only 'blog post' events
    whose `date` is within the last `days` days (default 14) of today - e.g.
    for a digest or dashboard. Does not modify the file.

    Deduped by `link`, keeping the first occurrence: cycles fired close
    together (e.g. repeated manual --force runs) can each log the same link
    before the other's write is visible, leaving duplicate rows in the log
    even though the uniqueness check holds under the normal weekly cadence."""
    if not path.exists():
        return []
    cutoff = (datetime.now().date() - timedelta(days=days)).isoformat()
    posts = []
    seen_links: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("event") != "blog post" or entry.get("date", "") < cutoff:
                continue
            link = entry.get("link")
            if link in seen_links:
                continue
            seen_links.add(link)
            posts.append(entry)
    return posts


def find_last_scrape_completion(events_path: Path) -> tuple[str, datetime] | None:
    """Read-only: scan events_log.jsonl for the most recent 'scraping run
    complete' event (pushed at the end of run_scrape_cycle, once both legs
    have finished) and return (run_id, finished_at) - or None if no run has
    completed yet. finished_at is UTC-aware, matching push_event's timestamps."""
    if not events_path.exists():
        return None
    last: tuple[str, datetime] | None = None
    with events_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("event") != "scraping run complete":
                continue
            try:
                finished_at = datetime.fromisoformat(entry["timestamp"])
            except (KeyError, ValueError):
                continue
            if last is None or finished_at > last[1]:
                last = (entry.get("run_id", ""), finished_at)
    return last


def hours_until_next_run(now: datetime) -> float:
    candidate = now.replace(hour=TARGET_HOUR, minute=TARGET_MINUTE, second=0, microsecond=0)
    candidate += timedelta(days=(TARGET_WEEKDAY - now.weekday()) % 7)
    if candidate <= now:
        candidate += timedelta(days=7)
    return (candidate - now).total_seconds() / 3600


def check_day_of_week() -> None:
    """Heartbeat log of the current day/hour - informational only, doesn't
    itself gate whether a scrape runs (see main: the actual trigger check
    happens after the sleep, not here). Only logs on the hour - the
    scheduler loop itself polls every TRIGGER_POLL_SECONDS (to land on
    TARGET_MINUTE), which would otherwise spam this line every minute."""
    now = datetime.now()
    if now.minute != 0:
        return
    hours_left = hours_until_next_run(now)
    logger.info(f"Tick: {now.strftime('%A')} {now.hour:02d}:00, hours until next run: {hours_left:.2f}")


def run_scrape_cycle(events_path: Path, sources_path: Path, cache_path: Path) -> None:
    run_id = str(uuid.uuid4())
    push_event(events_path, "scraping runned", worker_name="feed_scraper", run_id=run_id)
    metrics.record_scrape_cycle_run()
    logger.info(f"Scraping run started: run_id={run_id}")

    sources = load_sources(sources_path)
    cache = load_article_date_cache(cache_path)
    seen_links = load_logged_links(events_path)
    posts_found = 0
    posts_skipped = 0

    for src in sources:
        try:
            posts = crawl(src, cache)
        except Exception as exc:  # keep going on any single-source failure
            logger.error(f"  !! {src.name}: {exc}")
            metrics.record_rss_source_failed(src.name)
            continue
        for post in posts:
            if post.link in seen_links:
                posts_skipped += 1
                continue
            seen_links.add(post.link)
            push_event(
                events_path,
                "blog post",
                worker_name="rss_scraper",
                blog_post_id=str(uuid.uuid4()),
                parent_run=run_id,
                link=post.link,
                source=post.source,
                date=post.date,
                title=post.title,
            )
            metrics.record_blog_post_found("rss_scraper")
            posts_found += 1

    save_article_date_cache(cache, cache_path)
    logger.info(
        f"RSS leg complete: run_id={run_id}, posts_found={posts_found}, "
        f"posts_skipped_duplicate={posts_skipped}"
    )

    # --- mhtml leg: headless CloakBrowser capture, forced fresh every cycle,
    # then parsed the same way blog_feed_scrapers.py's own CLI would. Wrapped
    # so a browser failure (e.g. missing browser binary on a fresh host)
    # doesn't erase the RSS leg's results above.
    try:
        browser_use.feed(MHTML_DIR, headless=True, force=True, run_id=run_id)
        mhtml_posts = run_parse_cycle(MHTML_DIR, MHTML_SITES_PATH, events_path=events_path)
        logger.info(f"mhtml leg complete: run_id={run_id}, posts_found={len(mhtml_posts)}")
    except Exception as exc:
        logger.error(f"mhtml leg failed: {exc}")
        metrics.record_mhtml_leg_failed()

    logger.info(f"Scraping run complete: run_id={run_id}")
    # Marks both legs done regardless of the mhtml leg's outcome above (it's
    # wrapped to degrade gracefully) - this is the signal digest_watchdog_loop
    # waits for, so it must fire even when the mhtml leg failed.
    push_event(events_path, "scraping run complete", worker_name="feed_scraper", run_id=run_id)


def digest_watchdog_loop(events_path: Path) -> None:
    """Background thread, started alongside the weekly scheduler loop in
    main(): every hour, check whether the most recent scrape cycle (both
    legs - see the 'scraping run complete' push at the end of
    run_scrape_cycle) finished within the last hour, and if so - once per
    run_id - generate the digest (Nebius, via llm_client) and send it to
    Telegram (tg_pusher), exactly like `feed.py --generate` does by hand.
    Decoupled from the weekly trigger itself so a scrape kicked off with
    --force, or one that runs long, still gets its digest sent without
    manual intervention. A run whose digest generation/send fails is not
    retried - by the next hourly check its age is past the 1h window."""
    last_sent_run_id: str | None = None
    while True:
        time.sleep(SLEEP_SECONDS)
        last = find_last_scrape_completion(events_path)
        if last is None:
            continue
        run_id, finished_at = last
        if run_id == last_sent_run_id:
            continue
        age = datetime.now(timezone.utc) - finished_at
        if age > timedelta(hours=1):
            continue

        logger.info(f"digest watchdog: run_id={run_id} finished {age.total_seconds() / 60:.0f} min ago, generating digest")
        try:
            llm_client.validate_credentials()
            posts = load_recent_blog_posts(events_path, days=7)
            jsonl_text = "\n".join(
                json.dumps({
                    "link": post["link"],
                    "title": post.get("title", ""),
                    "publish_date": post.get("date", ""),
                }, ensure_ascii=False)
                for post in posts
            )
            digest = llm_client.generate_digest(jsonl_text)
            message_ids = tg_pusher.send_digest(digest, events_path=events_path)
            logger.info(f"digest watchdog: sent digest to Telegram as {len(message_ids)} message(s): {message_ids}")
            last_sent_run_id = run_id
        except Exception as exc:
            logger.error(f"digest watchdog: failed to generate/send digest for run_id={run_id}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Weekly (Tuesday 09:15 local time) blog-scrape scheduler.")
    parser.add_argument(
        "--force", action="store_true",
        help="Run one scrape cycle immediately, skipping the day/hour check, then resume the normal weekly wait loop.",
    )
    parser.add_argument(
        "--generate", action="store_true",
        help="Build a JSONL digest (link, title, publish_date) from 'blog post' events in the last 7 days, "
             "pass it to the LLM to produce a topic-organized Markdown reading digest, send it to Telegram "
             "(as HTML, split into multiple messages if needed), print it, and exit.",
    )
    args = parser.parse_args()

    events_path = DEFAULT_EVENTS_LOG_PATH

    if args.generate:
        llm_client.validate_credentials()
        posts = load_recent_blog_posts(events_path, days=7)
        jsonl_text = "\n".join(
            json.dumps({
                "link": post["link"],
                "title": post.get("title", ""),
                "publish_date": post.get("date", ""),
            }, ensure_ascii=False)
            for post in posts
        )
        digest = llm_client.generate_digest(jsonl_text)
        print(digest)
        message_ids = tg_pusher.send_digest(digest, events_path=events_path)
        logger.info(f"sent digest to Telegram as {len(message_ids)} message(s): {message_ids}")
        return

    logger.info(f"feed.py started - watching for Tuesday {TARGET_HOUR:02d}:{TARGET_MINUTE:02d}, events -> {events_path}")

    threading.Thread(target=digest_watchdog_loop, args=(events_path,), daemon=True, name="digest-watchdog").start()

    if args.force:
        logger.info("--force given: running scrape cycle immediately")
        run_scrape_cycle(events_path, SOURCES_PATH, ARTICLE_DATE_CACHE_PATH)

    while True:
        check_day_of_week()
        time.sleep(TRIGGER_POLL_SECONDS)
        now = datetime.now()
        if now.weekday() == TARGET_WEEKDAY and now.hour == TARGET_HOUR and now.minute == TARGET_MINUTE:
            run_scrape_cycle(events_path, SOURCES_PATH, ARTICLE_DATE_CACHE_PATH)


if __name__ == "__main__":
    main()
