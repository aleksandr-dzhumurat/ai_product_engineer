"""Extract links from a Nebius MHTML page and download documentation pages.

Step 1 – Extract links from MHTML and crawl to discover all sub-pages:
    python src/scraping/scrape_nebius.py --input data/nebius_site/nebius_main_page.mhtml [--max-depth N]

    Output: data/nebius_site/nebius_main_page.jsonl
    Extracts seed links from the MHTML page, then fetches each page to discover
    sidebar sub-pages up to --max-depth levels. All discovered URLs are written
    to the JSONL index with their depth:
        {"url": "https://docs.nebius.com/...", "page_title": "...", "parent_page_title": "Infrastructure/Compute", "depth": 2}

Step 2 – Download HTML pages listed in the JSONL index:
    python src/scraping/scrape_nebius.py --input data/nebius_site/nebius_main_page.jsonl --download

    Downloads each URL from the JSONL, saves HTML files into per-section
    subdirectories under data/nebius_site/, and updates the JSONL with "local_path":
        {"url": "...", "page_title": "...", "parent_page_title": "...", "depth": 2, "local_path": "data/nebius_site/.../file.html"}

Next step – Convert downloaded HTML to Markdown:
    DATA_DIR="$(pwd)" python src/rag/nebius_html2md.py --input data/nebius_site/
"""

import argparse
import email
import json
import re
import time
from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote

import requests


# Mapping from h3 sub-section names to their parent navigation tab.
# "Infrastructure" groups several h3 sections; others map 1:1 to their tab name.
H3_TO_TAB = {
    "Compute": "Infrastructure",
    "Storage": "Infrastructure",
    "Network": "Infrastructure",
}


class LinkExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []  # list of (href, text, section)
        self._current_tag = None
        self._current_href = None
        self._current_text_parts = []
        self._in_h3 = False
        self._current_section = ""

    def handle_starttag(self, tag, attrs):
        if tag == "h3":
            self._in_h3 = True
            self._h3_parts = []
        elif tag == "a":
            attrs_dict = dict(attrs)
            href = attrs_dict.get("href", "")
            if href and not href.startswith(("cid:", "javascript:", "#")):
                self._current_tag = "a"
                self._current_href = href
                self._current_text_parts = []

    def handle_data(self, data):
        if self._in_h3:
            self._h3_parts.append(data.strip())
        if self._current_tag == "a":
            self._current_text_parts.append(data.strip())

    def handle_endtag(self, tag):
        if tag == "h3" and self._in_h3:
            section = " ".join(p for p in self._h3_parts if p)
            if section:
                tab = H3_TO_TAB.get(section)
                if tab:
                    self._current_section = f"{tab}/{section}"
                else:
                    self._current_section = section
            self._in_h3 = False
        elif tag == "a" and self._current_tag == "a":
            text = " ".join(p for p in self._current_text_parts if p)
            if self._current_href:
                self.links.append((self._current_href, text, self._current_section))
            self._current_tag = None
            self._current_href = None
            self._current_text_parts = []


def decode_quoted_printable(raw_html: str) -> str:
    """Decode quoted-printable soft line breaks and encoded chars."""
    # Join soft line breaks (=\n)
    text = raw_html.replace("=\n", "")
    # Decode =XX hex sequences
    text = re.sub(r"=([0-9A-Fa-f]{2})", lambda m: chr(int(m.group(1), 16)), text)
    return text


def extract_html_from_mhtml(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    msg = email.message_from_string(raw)
    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            if ct == "text/html":
                payload = part.get_payload(decode=False)
                return decode_quoted_printable(payload)

    # Non-multipart fallback
    return decode_quoted_printable(msg.get_payload(decode=False))


class TitleExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self._in_title = False
        self.title = ""

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self._in_title = True

    def handle_data(self, data):
        if self._in_title:
            self.title += data

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False


def extract_title(html: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", html, re.DOTALL)
    return match.group(1).strip() if match else ""


def extract_links(html: str) -> list[tuple[str, str, str]]:
    parser = LinkExtractor()
    parser.feed(html)
    return parser.links


SKIP_EXTENSIONS = (".css", ".js", ".png", ".jpg", ".jpeg", ".svg", ".ico", ".xml", ".md")


def filter_links(links: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    seen = set()
    filtered = []
    for href, text, section in links:
        href_clean = unquote(href).split("?")[0].rstrip("/")
        if href_clean in seen:
            continue
        if any(href_clean.endswith(ext) for ext in SKIP_EXTENSIONS):
            continue
        if "googletagmanager" in href or "onetrust.com" in href:
            continue
        seen.add(href_clean)
        filtered.append((href, text, section))
    return filtered


def sanitize_dirname(title: str) -> str:
    """Turn a page title into a filesystem-safe directory name."""
    name = title.split(" - ")[0].strip().lower()
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"[\s-]+", "_", name).strip("_")
    return name


def url_to_filename(url: str) -> str:
    """Derive a short filename from a docs URL path."""
    path = url.rstrip("/").split("//", 1)[-1]  # drop scheme
    path = path.split("/", 1)[-1]  # drop domain
    if not path:
        return "index"
    return re.sub(r"[^\w]", "_", path).strip("_")


SKIP_DOMAINS = {"console.nebius.com", "nebius.com", "www.nebius.com"}


def should_skip(url: str) -> bool:
    if "#" in url and url.index("#") == url.index("//") + 2:
        return True
    try:
        domain = url.split("//")[1].split("/")[0]
    except IndexError:
        return True
    return domain in SKIP_DOMAINS


BASE_DOCS_URL = "https://docs.nebius.com"


SKIP_LINK_PATTERNS = (".md", ".css", ".js", ".xml", ".txt", ".png", ".jpg",
                      ".ico", ".svg", ".json", ".woff", ".woff2")


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
        # Skip asset paths
        if "/mintlify-assets/" in path or "/_mintlify/" in path or "/_next/" in path:
            continue
        # Only keep links under the same section
        if not path.startswith(base_path):
            continue
        path_clean = path.rstrip("/")
        # Skip bare /index pages and the base path itself
        if path_clean.endswith("/index") or path_clean == base_path.rstrip("/"):
            continue
        if path_clean not in seen:
            seen.add(path_clean)
            links.append(BASE_DOCS_URL + path_clean)
    return links


def download_html(url: str, dest: Path) -> bool:
    """Download HTML from url to dest. Returns True on success."""
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        dest.write_text(resp.text, encoding="utf-8")
        return True
    except requests.RequestException as e:
        print(f"  FAILED {url}: {e}")
        return False



def fetch_html(url: str) -> str | None:
    """Fetch HTML from a URL without saving. Returns HTML text or None on failure."""
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        print(f"  FAILED {url}: {e}")
        return None


def discover_pages(
    seed_records: list[dict],
    max_depth: int,
) -> list[dict]:
    """Crawl seed URLs to discover all sub-pages up to max_depth.

    Fetches pages temporarily (not saved to disk) to extract sidebar links.
    Returns a list of records with url, page_title, parent_page_title, depth.
    """
    # Group seeds by section
    seeds_by_section: dict[str, list[dict]] = defaultdict(list)
    for rec in seed_records:
        seeds_by_section[rec["parent_page_title"]].append(rec)

    all_records: list[dict] = []
    global_visited: set[str] = set()

    for parent_title, children in seeds_by_section.items():
        print(f"\n=== Discovering: {parent_title}")

        queue: list[tuple[str, str, int]] = []  # (url, page_title, depth)
        for child in children:
            url = child["url"]
            if not should_skip(url):
                queue.append((url, child.get("page_title", ""), 0))

        visited: set[str] = set()
        while queue:
            url, title, depth = queue.pop(0)
            if url in visited or url in global_visited:
                continue
            visited.add(url)
            global_visited.add(url)

            all_records.append({
                "url": url,
                "page_title": title,
                "parent_page_title": parent_title,
                "depth": depth,
            })

            # Discover sub-pages from sidebar (only if within depth limit)
            if depth < max_depth:
                print(f"  CRAWL depth={depth} {url}")
                html_text = fetch_html(url)
                if html_text is None:
                    continue
                time.sleep(0.3)

                url_path = "/" + url.split("//")[1].split("/", 1)[-1]
                # Use top-level section as base_path (e.g. /kubernetes/)
                # so that sub-sections like /kubernetes/gpu/ are also discovered
                parts = url_path.strip("/").split("/")
                base_path = "/" + parts[0] + "/"
                sub_links = extract_sidebar_links(html_text, base_path)
                for link in sub_links:
                    if link not in visited and link not in global_visited:
                        # Fetch title will happen when this link is processed
                        queue.append((link, "", depth + 1))

    return all_records


def download_pages(records: list[dict], base_dir: Path) -> dict[str, Path]:
    """Download HTML pages listed in records.

    Returns a mapping of URL -> local file path for all downloaded pages.
    """
    url_to_path: dict[str, Path] = {}

    # Group by parent_page_title for directory structure
    by_section: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_section[rec["parent_page_title"]].append(rec)

    for parent_title, children in by_section.items():
        subdir = base_dir / sanitize_dirname(parent_title)
        subdir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {parent_title} -> {subdir}/")

        for child in children:
            url = child["url"]
            if should_skip(url):
                continue

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


def main():
    parser = argparse.ArgumentParser(description="Extract links from a Nebius MHTML page and optionally download pages")
    parser.add_argument("--input", required=True,
                        help="Path to MHTML file (step 1) or JSONL index (step 2 --download)")
    parser.add_argument("--download", action="store_true",
                        help="Download HTML pages from the JSONL index.")
    parser.add_argument("--max-depth", type=int, default=1,
                        help="Max crawl depth for sub-page discovery in step 1 (default: 1).")
    args = parser.parse_args()

    input_path = Path(args.input)

    if args.download:
        # Step 2: download pages listed in JSONL
        if input_path.suffix == ".jsonl":
            jsonl_path = input_path
        else:
            jsonl_path = input_path.with_suffix(".jsonl")
        if not jsonl_path.exists():
            raise SystemExit(f"JSONL index not found: {jsonl_path}")

        records = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))

        url_to_path = download_pages(records, jsonl_path.parent)

        # Update JSONL with local_path
        for record in records:
            local = url_to_path.get(record["url"])
            if local:
                record["local_path"] = str(local)

        with open(jsonl_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        downloaded = sum(1 for r in records if "local_path" in r)
        print(f"\nUpdated {jsonl_path} ({downloaded}/{len(records)} pages downloaded)")
        print(f'\nTo convert downloaded HTML to Markdown run:\n  DATA_DIR="$(pwd)" python src/rag/nebius_html2md.py --input {jsonl_path.parent}/')
    else:
        # Step 1: extract seed links from MHTML + crawl to discover sub-pages
        output_path = input_path.with_suffix(".jsonl")

        html = extract_html_from_mhtml(args.input)
        page_title = extract_title(html)
        links = extract_links(html)
        links = filter_links(links)

        # Build seed records from MHTML links
        seed_records = []
        for href, text, section in links:
            if not should_skip(href):
                seed_records.append({
                    "url": href,
                    "page_title": text if text else "",
                    "parent_page_title": section if section else page_title,
                })

        # Crawl to discover all sub-pages
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
