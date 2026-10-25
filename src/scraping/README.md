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

See the [scraper refactoring plan](scraping_docs/refactoring_plan.md) for planned cleanup and progress.

The [WikiArt pipeline](scrape_wikiart.py) runs separately from the blog scheduler and uses HTTP requests rather than Chromium. After installing the scraping requirements above, run from the repository root:

```sh
src/scraping/.venv/bin/python src/scraping/scrape_wikiart.py --data-dir data
```

Add `--artworks-per-artist N` to limit the number of artwork pages collected per artist. The pipeline collects alphabet listings, artist metadata and bios, and artwork image URLs; then it writes the merged `06_content_db.csv.gz` dataset and `06_tags_db.csv.gz` tag vocabulary. It uses two worker processes to turn `data/artworks_raw_html/*.html` into `data/artworks_json/*.json`, skipping files that already have a JSON output. The final report is `data/artworks_json_descriptive_stats.json`.
