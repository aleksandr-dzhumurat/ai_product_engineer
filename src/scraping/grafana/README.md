# blogfeed Grafana dashboard

`blogfeed-dashboard.json` — dashboard for the `service.name="blogfeed"` counters emitted by `src/scraping/metrics.py`. Separate from the existing **tts-pipeline** dashboard, which covers `service.name="tts-pipeline"` (`src/scraping/data/newsfeed/src/newsfeed/metrics.py`).

## Import

1. Grafana → **Dashboards → New → Import**
2. Upload the JSON (or paste it)
3. When prompted for **Prometheus**, pick the hosted Prometheus that receives your OTLP data — usually `grafanacloud-graynettle2032-prom`
4. Save

The dashboard declares `${DS_PROMETHEUS}` as an import input, so it carries no hard-coded datasource UID and stays portable across stacks.

UID is `blogfeed-pipeline`. Re-importing with the same UID overwrites in place, so this file can live in git as the source of truth.

## Metric name translation

The OTLP gateway rewrites names on the way into Prometheus. Dots become underscores, and monotonic counters gain `_total`:

| `metrics.py` | queried as |
|---|---|
| `scrape_cycle.runs` | `scrape_cycle_runs_total` |
| `blog_post.found` | `blog_post_found_total` |
| `rss_source.failed` | `rss_source_failed_total` |
| `mhtml.captured` | `mhtml_captured_total` |
| `mhtml_leg.failed` | `mhtml_leg_failed_total` |
| `llm.requests` | `llm_requests_total` |
| `llm.tokens` | `llm_tokens_total` |

Resource attributes become labels: `service.name` → `job`, `service.instance.id` → `instance`.

**If every panel is empty**, check the real name in Explore first: `{job="blogfeed"}`. Some stacks are configured without suffixes; if yours is, strip `_total` from all seven queries.

## Layout

| Row | Panels |
|---|---|
| **Pipeline health** | six stat tiles — scrape cycles, posts found, mHTML captures, LLM requests, RSS source failures, mHTML leg failures |
| **Discovery** | posts/day split by leg (bar chart); mHTML captures per site (bar gauge) |
| **Failures** | RSS failures by source (table); failures per day (bars) |
| **LLM usage** | requests/day; tokens/day input vs output; tokens by model (table) |
| **Notes** | collapsed — how to read the dashboard, and known gaps |

Variables: `$job` (default `blogfeed`), `$instance` (multi-select hostname, defaults to All), `$window` (8d / **14d** / 30d / 90d).

## Two design decisions worth knowing

**Windows are ≥ 8 days.** `run_scrape_cycle()` fires weekly (Friday 14:00, `TARGET_WEEKDAY = 4`). A 6h or 24h window would show red six days out of seven. `$window` defaults to 14d so two consecutive cycles are always in view — one missed cycle is visible without being ambiguous.

**`noValue` is `0`, not blank.** Because these are dead-man's switches, a metric that has never arrived and a metric that arrived as zero mean the same thing operationally: the code path isn't running. Setting `noValue: "0"` makes the presence counters render red rather than a neutral grey "No data", so a never-fired counter looks like the alarm it is.

That matters right now for `blog_post.found`, `mhtml.captured`, `rss_source.failed` and `mhtml_leg.failed` — all wired before the `GRAFANA_CLOUD_TOKEN` import-order fix, so their earlier calls were lost. Those four tiles will read red until the next real cycle. `scrape_cycle.runs`, `llm.requests` and `llm.tokens` were verified sending.

## Alerting

Alert on **no data**, not on a threshold. Suggested rules, all `for: 0m` on a 1h evaluation:

```promql
# scheduler died — the top-level switch
sum(increase(scrape_cycle_runs_total{job="blogfeed"}[8d])) == 0

# cycles run but a leg produces nothing
sum(increase(scrape_cycle_runs_total{job="blogfeed"}[8d])) > 0
  and sum(increase(blog_post_found_total{job="blogfeed"}[8d])) == 0

# the mhtml leg is degrading gracefully, repeatedly
sum(increase(mhtml_leg_failed_total{job="blogfeed"}[8d])) > 0
```

Set each rule's **No data** state to `Alerting` — that is the whole point of the pattern.

## Known gaps

**A site that stops being captured disappears from the per-site panel** rather than dropping to zero. Prometheus has no series for a label value that stopped being emitted, so `mhtml.captured{stem=...}` simply vanishes. Read that panel against the site list in `MHTML_SITES_PATH`, not on its own. If you want a true per-site switch, the fix is on the emitting side: record a zero for every configured site at the start of each cycle.

**`llm.tokens` carries no cost.** The newsfeed pipeline records `nebius.cost_usd` from `config.NEBIUS_MODEL_PRICING`; blogfeed doesn't. Porting that pricing block from `record_nebius_tokens` into `record_llm_tokens` would let a cost panel drop straight in.

**`model` can be an empty string** — `record_llm_tokens` passes `model or ""`. The tokens-by-model table maps that to `(model unset)`.

**`llm.requests` counts backoff retries.** `generate()` is wrapped in `@backoff.on_exception(..., max_tries=5)`, and backoff re-invokes the whole decorated function per attempt, so one logical translation can add up to 5. The counter answers "are we reaching the API at all", not "how many digests were produced".
