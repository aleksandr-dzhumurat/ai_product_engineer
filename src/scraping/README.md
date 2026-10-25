# Blog scraping

[`feed.py`](feed.py) runs two scraping passes every Tuesday at 09:15 local time. The first uses RSS/Atom or direct HTML fetching. The second uses headless Chromium to capture blog index pages that need JavaScript to render. The browser saves snapshots; a separate parser extracts posts without opening the individual articles.

## How headless capture works

1. [`browser_use.py`](browser_use.py) reads [`browser_sites.jsonl`](browser_sites.jsonl). Each entry supplies a `stem` (the snapshot filename without the extension), a `source` URL, and a `parser` name used in the next stage.
2. CloakBrowser starts one Chromium context and reuses one tab for all sites, visiting them in registry order. It uses the persistent `profile/` directory next to the script, so state such as cookies can carry over between runs. Headless mode means there is no visible window; Chromium still executes JavaScript and renders the page.
3. For each site, the tab waits for `DOMContentLoaded` (up to 30 seconds), pauses for three seconds, scrolls down 800 pixels, and pauses for another second. Chromium's DevTools `Page.captureSnapshot` command saves the rendered page as `<stem>.mhtml`. With a visible browser, the script also saves a PNG screenshot; headless mode skips it.
4. [`blog_feed_scrapers.py`](blog_feed_scrapers.py) reads the HTML part of each MHTML archive and runs the site-specific BeautifulSoup parser named in the registry. It appends a `blog post` event only for links not already recorded in the shared event log, including links found by the RSS pass.

The scheduler uses `headless=True` and `force=True`, writing snapshots to `$DATA_DIR/mhtml` (`data/mhtml` by default). Standalone capture opens a visible browser unless you pass `--headless`, and skips existing snapshots unless you pass `--force`.

## Run capture and parsing separately

From the repository root, set up the local environment with [uv](https://docs.astral.sh/uv/):

```sh
cd src/scraping
uv venv
uv pip install --python .venv/bin/python -r requirements.txt
```

From `src/scraping`, run:

```sh
.venv/bin/python browser_use.py --scenario feed --headless --force --output-dir data/mhtml
.venv/bin/python blog_feed_scrapers.py --mhtml-dir data/mhtml
```

Capture and parsing are separate commands, so you can rerun the parser on saved snapshots without visiting the sites again. For a different workflow, `browser_use.py --scenario explore --input <file.md>` visits links from a Markdown file and saves page titles and visible text instead of MHTML. That mode also accepts `--headless`.

## Current limitations

- **A failed refresh can leave an old snapshot in use.** `--force` attempts a new capture but does not remove the old file first. If a capture fails before writing, the parser can still read the previous file. A successful capture list scoped to the current run would prevent that.
- **A saved snapshot doesn't guarantee the post list loaded.** Capture does not check the HTTP status or wait for a post-list selector, so an error page or partially rendered index can be saved as a success.
- **One bad snapshot can stop the parse pass.** Capture logs failures per site and continues, but an archive or parser exception can stop parsing before the remaining sites are processed.

See [the blog scraping notes](scraping_docs/blogposts.md) for more on the two passes and the shared event log.

## WikiArt dataset

The [WikiArt pipeline](scrape_wikiart.py) runs separately from the blog scheduler. It fetches HTML over HTTP without Chromium and collects image URLs, not image files.

### Run and resume

After installing the scraping requirements above, run from the repository root:

```sh
mkdir -p data/artworks
src/scraping/.venv/bin/python -u src/scraping/scrape_wikiart.py \
  --data-dir data/artworks --artwork-download-workers 8 \
  >> data/artworks/wikiart_run_gzip.log 2>&1
```

Watch progress from another terminal with `tail -f data/artworks/wikiart_run_gzip.log`. The default is four download threads; the flag changes artwork-page downloads only. Use `--artworks-per-artist N` to limit pages per artist, preferably with a separate `--data-dir` so a trial run does not overwrite full-run outputs.

To resume after stopping, rerun with the **same `--data-dir`**. Existing `.html` and `.html.gz` artwork pages are reused; only missing pages are downloaded. New pages are saved individually as `.html.gz`, and the parser reads them without unpacking to disk. The download counter starts over for the remaining queue, not the entire cache. Do not run two scrapers against the same directory. Reduce concurrency if WikiArt returns HTTP 429 or timeouts increase.

### Stages and outputs

All paths below are relative to `--data-dir`.

| Stage | Output |
| --- | --- |
| 1. Collect artist pages | `06_artists_pages.csv` |
| 2. Collect artist metadata and biographies | `06_artists_info.csv`, `06_artists_wiki_texts.csv` |
| 3. Collect artwork pages and image URLs | Listings in `art_links_lists/`, HTML in `artworks_raw_html/`, and `06_artists_artworks.jsonl` |
| 4. Build the merged dataset and tag vocabulary | `06_content_db.csv.gz`, `06_tags_db.csv.gz` |
| 5. Extract artwork descriptions | `artworks_json/*.json`, generated with two worker processes; existing JSON files are skipped |
| 6. Build the quality report | `artworks_json_descriptive_stats.json` |

### Download speed and disk usage

The main cost is waiting for one HTTP response per uncached artwork, not transferring a large file. In a spot check, a roughly 52 KB response took about one second to deliver its first byte, including on a reused connection. One live run measured about 3.2 pages/s with four threads and 6.2 pages/s with eight threads and gzip. At that rate, 180,000 uncached pages take roughly eight hours to download, before parsing and dataset generation.

Gzip reduces disk usage, not server response time. On a sample of 2,374 cached pages, it reduced storage from about 816 MiB to 143 MiB (83%). Both speed and compression vary with the pages and the site.

### Serve scraped content

After step 4 finishes, start the recommendation API from the repository root using an environment with the root `requirements.txt` installed:

```sh
WIKIART_DATA_DIR=data/artworks python -m uvicorn src.recsys.content_api:app
```

The API loads `06_artists_info.csv`, `06_content_db.csv.gz`, and `06_tags_db.csv.gz` on startup. It reads image URLs from the merged CSV and does not require per-artwork JSON or the final stats report. Set `WIKIART_DATA_DIR` to the same directory passed to the scraper's `--data-dir`.

For the ArtSwipe interface, run `make artworks-ui` from the repository root and open <http://127.0.0.1:8501>. This starts both FastAPI and Streamlit using `.venv/bin/python`; press Ctrl+C to stop both. Override `ARTWORKS_API_PORT`, `ARTWORKS_UI_PORT`, or `WIKIART_DATA_DIR` on the `make` command line if needed. User actions are stored in memory and reset when the API stops.

See the [scraper refactoring plan](scraping_docs/refactoring_plan.md) for implementation notes.
