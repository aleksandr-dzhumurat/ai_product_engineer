"""Extract links from a Tavily docs MHTML page and download documentation pages.

Step 1 – Extract links from MHTML and crawl to discover all sub-pages:
    python src/scraping/scrape_tavily.py --input data/scraping/tavily_docs/tavily_docs.mhtml [--max-depth N]

    Output: data/scraping/tavily_docs/tavily_docs.jsonl
    Extracts seed links from the MHTML navbar, then fetches each page to discover
    sidebar sub-pages up to --max-depth levels. All discovered URLs are written
    to the JSONL index with their depth:
        {"url": "https://docs.tavily.com/...", "page_title": "...", "parent_page_title": "Agents", "depth": 0}

Step 2 – Download HTML pages listed in the JSONL index:
    python src/scraping/scrape_tavily.py --input data/scraping/tavily_docs/tavily_docs.jsonl --download

    Downloads each URL from the JSONL, saves HTML files into per-section
    subdirectories under data/scraping/tavily_docs/, and updates the JSONL with "local_path".
"""

import argparse
import email
import json
import re
import time
from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path

import requests

BASE_DOCS_URL = "https://docs.tavily.com"

# Only these top-level sections should be scraped.
TARGET_SECTIONS = {
    "Agents",
    "Introduction",
    "API & SDKs",
    "Ecosystem",
    "Examples",
}

# Map each section to its URL path prefix so that discovered sub-pages are
# assigned to the correct section (multiple sections share /documentation/).
# Longer prefixes are checked first so /documentation/api-reference/ wins
# over /documentation/.
SECTION_PATH_PREFIX = {
    "Agents": "/agents",
    "API & SDKs": "/documentation/api-reference",
    "Ecosystem": "/documentation/mcp",
    "Introduction": "/documentation",
    "Examples": "/examples",
}

SKIP_DOMAINS = {"app.tavily.com", "discord.gg", "tavily.com", "www.tavily.com", "github.com"}

SKIP_LINK_PATTERNS = (
    ".md", ".css", ".js", ".xml", ".txt", ".png", ".jpg",
    ".ico", ".svg", ".json", ".woff", ".woff2",
)


# ---------------------------------------------------------------------------
# MHTML parsing
# ---------------------------------------------------------------------------

def decode_quoted_printable(raw_html: str) -> str:
    text = raw_html.replace("=\n", "")
    text = re.sub(r"=([0-9A-Fa-f]{2})", lambda m: chr(int(m.group(1), 16)), text)
    return text


def extract_html_from_mhtml(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    msg = email.message_from_string(raw)
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/html":
                payload = part.get_payload(decode=False)
                return decode_quoted_printable(payload)

    return decode_quoted_printable(msg.get_payload(decode=False))


# ---------------------------------------------------------------------------
# Extract navbar section links from MHTML HTML
# ---------------------------------------------------------------------------

class NavTabExtractor(HTMLParser):
    """Extract <a class="... nav-tabs-item ..."> links from the top navbar."""

    def __init__(self):
        super().__init__()
        self.sections = []  # list of (href, section_name)
        self._in_nav_tab = False
        self._current_href = None
        self._text_parts = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            attrs_dict = dict(attrs)
            cls = attrs_dict.get("class", "")
            if "nav-tabs-item" in cls:
                self._in_nav_tab = True
                self._current_href = attrs_dict.get("href", "")
                self._text_parts = []

    def handle_data(self, data):
        if self._in_nav_tab:
            self._text_parts.append(data.strip())

    def handle_endtag(self, tag):
        if tag == "a" and self._in_nav_tab:
            name = " ".join(p for p in self._text_parts if p)
            if self._current_href and name:
                self.sections.append((self._current_href, name))
            self._in_nav_tab = False
            self._current_href = None
            self._text_parts = []


def extract_nav_sections(html: str) -> list[dict]:
    parser = NavTabExtractor()
    parser.feed(html)

    seed_records = []
    for href, name in parser.sections:
        if name not in TARGET_SECTIONS:
            continue
        url = href if href.startswith("http") else BASE_DOCS_URL + href
        seed_records.append({
            "url": url,
            "page_title": name,
            "parent_page_title": name,
        })
    return seed_records


# ---------------------------------------------------------------------------
# Sidebar link extraction (from fetched pages)
# ---------------------------------------------------------------------------

def extract_sidebar_links(html: str, base_path: str) -> list[str]:
    """Extract sidebar sub-page links that share the same URL prefix."""
    matches = re.findall(r'href="(/[^"]+)"', html)
    seen = set()
    links = []
    for path in matches:
        if any(path.endswith(ext) for ext in SKIP_LINK_PATTERNS):
            continue
        if "#" in path:
            continue
        if "/mintlify-assets/" in path or "/_mintlify/" in path or "/_next/" in path:
            continue
        if not path.startswith(base_path):
            continue
        path_clean = path.rstrip("/")
        if path_clean == base_path.rstrip("/"):
            continue
        if path_clean not in seen:
            seen.add(path_clean)
            links.append(BASE_DOCS_URL + path_clean)
    return links


# ---------------------------------------------------------------------------
# Crawling
# ---------------------------------------------------------------------------

def fetch_html(url: str) -> str | None:
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        print(f"  FAILED {url}: {e}")
        return None


def _section_for_url(url: str) -> str:
    """Determine the parent section for a URL based on its path prefix."""
    path = "/" + url.split("//")[1].split("/", 1)[-1]
    # Check longest prefixes first so /documentation/api-reference wins over /documentation
    for section, prefix in sorted(SECTION_PATH_PREFIX.items(), key=lambda x: -len(x[1])):
        if path.startswith(prefix):
            return section
    return ""


def discover_pages(seed_records: list[dict], max_depth: int) -> list[dict]:
    # Collect all seed URLs across sections into a single queue.
    # Each seed is crawled once; the section is determined by URL path prefix.
    all_records: list[dict] = []
    visited: set[str] = set()

    queue: list[tuple[str, str, int]] = []
    for rec in seed_records:
        queue.append((rec["url"], rec.get("page_title", ""), 0))

    while queue:
        url, title, depth = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        section = _section_for_url(url)
        if not section:
            continue

        all_records.append({
            "url": url,
            "page_title": title,
            "parent_page_title": section,
            "depth": depth,
        })

        if depth < max_depth:
            print(f"  CRAWL depth={depth} [{section}] {url}")
            html_text = fetch_html(url)
            if html_text is None:
                continue
            time.sleep(0.3)

            # Use first path segment as base_path for sidebar extraction
            url_path = "/" + url.split("//")[1].split("/", 1)[-1]
            parts = url_path.strip("/").split("/")
            base_path = "/" + parts[0] + "/"
            sub_links = extract_sidebar_links(html_text, base_path)
            for link in sub_links:
                if link not in visited:
                    queue.append((link, "", depth + 1))

    return all_records


# ---------------------------------------------------------------------------
# Downloading
# ---------------------------------------------------------------------------

def sanitize_dirname(title: str) -> str:
    name = title.strip().lower()
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"[\s-]+", "_", name).strip("_")
    return name


def url_to_filename(url: str) -> str:
    path = url.rstrip("/").split("//", 1)[-1]
    path = path.split("/", 1)[-1]
    if not path:
        return "index"
    return re.sub(r"[^\w]", "_", path).strip("_")


def download_html(url: str, dest: Path) -> bool:
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        dest.write_text(resp.text, encoding="utf-8")
        return True
    except requests.RequestException as e:
        print(f"  FAILED {url}: {e}")
        return False


def download_pages(records: list[dict], base_dir: Path) -> dict[str, Path]:
    url_to_path: dict[str, Path] = {}

    by_section: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_section[rec["parent_page_title"]].append(rec)

    for parent_title, children in by_section.items():
        subdir = base_dir / sanitize_dirname(parent_title)
        subdir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {parent_title} -> {subdir}/")

        for child in children:
            url = child["url"]
            fname = url_to_filename(url)
            html_path = subdir / f"{fname}.html"

            if html_path.exists():
                print(f"  EXISTS {html_path.name}")
            else:
                print(f"  GET {url}")
                if not download_html(url, html_path):
                    continue
                time.sleep(0.5)

            url_to_path[url] = html_path

    return url_to_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract links from a Tavily docs MHTML page and optionally download pages",
    )
    parser.add_argument("--input", required=True,
                        help="Path to MHTML file (step 1) or JSONL index (step 2 --download)")
    parser.add_argument("--download", action="store_true",
                        help="Download HTML pages from the JSONL index.")
    parser.add_argument("--max-depth", type=int, default=1,
                        help="Max crawl depth for sub-page discovery in step 1 (default: 1).")
    args = parser.parse_args()

    input_path = Path(args.input)

    if args.download:
        jsonl_path = input_path if input_path.suffix == ".jsonl" else input_path.with_suffix(".jsonl")
        if not jsonl_path.exists():
            raise SystemExit(f"JSONL index not found: {jsonl_path}")

        records = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))

        url_to_path = download_pages(records, jsonl_path.parent)

        for record in records:
            local = url_to_path.get(record["url"])
            if local:
                record["local_path"] = str(local)

        with open(jsonl_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        downloaded = sum(1 for r in records if "local_path" in r)
        print(f"\nUpdated {jsonl_path} ({downloaded}/{len(records)} pages downloaded)")
    else:
        output_path = input_path.with_suffix(".jsonl")

        html = extract_html_from_mhtml(args.input)
        seed_records = extract_nav_sections(html)

        if not seed_records:
            raise SystemExit("No target sections found in the MHTML. Check TARGET_SECTIONS.")

        print(f"Found {len(seed_records)} seed sections:")
        for rec in seed_records:
            print(f"  {rec['parent_page_title']}: {rec['url']}")

        all_records = discover_pages(seed_records, args.max_depth)

        with open(output_path, "w", encoding="utf-8") as f:
            for record in all_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        depth_counts = defaultdict(int)
        for r in all_records:
            depth_counts[r["depth"]] += 1
        depth_summary = ", ".join(f"depth {d}: {c}" for d, c in sorted(depth_counts.items()))

        print(f"\nWrote {len(all_records)} pages to {output_path} ({depth_summary})")
        print(f"\nTo download HTML pages run:\n  python {Path(__file__).relative_to(Path.cwd())} --input {output_path} --download")


if __name__ == "__main__":
    main()
