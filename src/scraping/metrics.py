"""Grafana Cloud OTLP metrics for this pipeline - simple counters meant as
dead-man's switches, not success/failure metrics: if one stops incrementing,
a no-data alert fires, meaning that code path stopped running entirely (not
that one call failed - see record_rss_source_failed / record_mhtml_leg_failed
for that).

Copied from src/scraping/data/newsfeed/src/newsfeed/metrics.py and adapted:
same Grafana Cloud account/stack (GRAFANA_CLOUD_TOKEN is already shared via
.env), but service.name/meter name changed to "blogfeed" so this pipeline's
series don't collide with newsfeed's "newsfeed" ones in the same account. See
newsfeed's docs/gafana.md for how to find GRAFANA_OTLP_INSTANCE_ID/
GRAFANA_OTLP_ENDPOINT if the defaults below don't match your stack.
"""
import atexit
import base64
import logging
import os
import socket
from pathlib import Path

from dotenv import load_dotenv
from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource

log = logging.getLogger(__name__)

# Load .env here too, not just in llm_client.py/tg_pusher.py - GRAFANA_CLOUD_TOKEN
# below is read once at import time, and Python caches modules, so whichever
# file imports metrics.py *first* (e.g. browser_use.py, which never touches
# dotenv) would otherwise freeze this empty regardless of what any other
# module's own load_dotenv() call does afterward.
load_dotenv(Path(__file__).with_name(".env"))

GRAFANA_CLOUD_TOKEN = os.environ.get("GRAFANA_CLOUD_TOKEN", "")
GRAFANA_OTLP_ENDPOINT = os.environ.get("GRAFANA_OTLP_ENDPOINT", "https://otlp-gateway-prod-eu-west-2.grafana.net/otlp")
GRAFANA_OTLP_INSTANCE_ID = os.environ.get("GRAFANA_OTLP_INSTANCE_ID", "1813255")

_meter = None
_counters: dict[str, object] = {}


def _get_meter():
    global _meter
    if _meter is not None:
        return _meter
    if not GRAFANA_CLOUD_TOKEN:
        return None

    auth = base64.b64encode(f"{GRAFANA_OTLP_INSTANCE_ID}:{GRAFANA_CLOUD_TOKEN}".encode()).decode()
    reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(
            endpoint=f"{GRAFANA_OTLP_ENDPOINT}/v1/metrics",
            headers={"Authorization": f"Basic {auth}"},
        ),
        export_interval_millis=60_000,
    )
    provider = MeterProvider(
        # service.instance.id pinned to the hostname - otherwise the SDK
        # auto-generates a random one per process, causing "instance
        # sprawl" (a new series on every restart) in per-instance panels.
        resource=Resource.create({"service.name": "blogfeed", "service.instance.id": socket.gethostname()}),
        metric_readers=[reader],
    )
    metrics.set_meter_provider(provider)
    # feed.py's long-running scheduler lives long enough for the 60s periodic
    # export to fire on its own, but a one-shot invocation (--generate,
    # --force) can exit well before that timer - shutdown() forces a final
    # flush so those short-lived runs don't silently drop their metrics.
    atexit.register(provider.shutdown)
    _meter = metrics.get_meter("blogfeed")
    return _meter


def _record(counter_name: str, amount: int = 1, attributes: dict | None = None) -> None:
    meter = _get_meter()
    if meter is None:
        return
    counter = _counters.get(counter_name)
    if counter is None:
        counter = meter.create_counter(counter_name)
        _counters[counter_name] = counter
    counter.add(amount, attributes)


def record_scrape_cycle_run() -> None:
    """Call once per run_scrape_cycle() invocation (feed.py) - the weekly
    dead-man's switch. If this stops incrementing, the scheduler process
    itself died, not just one source or site. A no-op if
    GRAFANA_CLOUD_TOKEN isn't configured."""
    _record("scrape_cycle.runs")


def record_blog_post_found(worker_name: str) -> None:
    """Call once per newly-logged 'blog post' event, labeled by which leg
    found it (rss_scraper/mhtml_scraper) - lets a Grafana panel break out
    discovery rate per leg over time. A no-op if GRAFANA_CLOUD_TOKEN isn't
    configured."""
    _record("blog_post.found", attributes={"worker_name": worker_name})


def record_rss_source_failed(source: str) -> None:
    """Call whenever crawl(src, cache) raises for one RSS source (feed.py) -
    a single bad source shouldn't stop the cycle, but a spike here means a
    site changed its markup/feed. A no-op if GRAFANA_CLOUD_TOKEN isn't
    configured."""
    _record("rss_source.failed", attributes={"source": source})


def record_mhtml_capture(stem: str) -> None:
    """Call once per successful .mhtml capture (browser_use.py's feed()),
    labeled by site stem - a per-site dead-man's switch: if one site stops
    incrementing, CloakBrowser is silently failing against just that page.
    A no-op if GRAFANA_CLOUD_TOKEN isn't configured."""
    _record("mhtml.captured", attributes={"stem": stem})


def record_mhtml_leg_failed() -> None:
    """Call from feed.py's except block around the whole browser/mhtml leg -
    that leg is wrapped so a CloakBrowser failure degrades gracefully
    instead of losing the RSS leg's results, but the failure should still be
    visible. A no-op if GRAFANA_CLOUD_TOKEN isn't configured."""
    _record("mhtml_leg.failed")


def record_llm_request() -> None:
    """Call once per generate() attempt (llm_client.py), regardless of
    outcome - a dead-man's switch for the digest-generation path. A no-op if
    GRAFANA_CLOUD_TOKEN isn't configured."""
    _record("llm.requests")


def record_llm_tokens(prompt_tokens: int, completion_tokens: int, model: str | None = None) -> None:
    """Adds prompt/completion tokens (per response.usage) to a running
    llm.tokens counter, labeled by token_type=input/output and model. A
    no-op if GRAFANA_CLOUD_TOKEN isn't configured."""
    _record("llm.tokens", prompt_tokens, {"token_type": "input", "model": model or ""})
    _record("llm.tokens", completion_tokens, {"token_type": "output", "model": model or ""})
