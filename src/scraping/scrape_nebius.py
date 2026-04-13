"""Extract links from a Nebius MHTML page and save as JSONL.

Usage:
    python src/scraping/scrape_nebius.py --input data/nebius_site/nebius_main_page.mhtml

Output:
    data/nebius_site/nebius_main_page.jsonl

    Each line is a JSON object with the following fields:
        {"url": "https://docs.nebius.com/...", "page_title": "Link text", "parent_page_title": "Section heading"}
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


def load_nebius_jsonl_index(input_path: Path) -> dict[str, list[dict]]:
    """Read JSONL and group records by parent_page_title."""
    pages_by_section = defaultdict(list)
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            pages_by_section[record["parent_page_title"]].append(record)
    return dict(pages_by_section)


def download_pages(pages_by_section: dict[str, list[dict]], base_dir: Path, max_depth: int) -> None:
    """Crawl and download HTML pages recursively."""
    for parent_title, children in pages_by_section.items():
        subdir = base_dir / sanitize_dirname(parent_title)
        subdir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {parent_title} -> {subdir}/")

        queue = []  # list of (url, depth)
        for child in children:
            url = child["url"]
            if not should_skip(url):
                queue.append((url, 0))

        visited = set()
        while queue:
            url, depth = queue.pop(0)
            if url in visited:
                continue
            visited.add(url)

            fname = url_to_filename(url)
            html_path = subdir / f"{fname}.html"

            # Download HTML (skip if already exists)
            if html_path.exists():
                print(f"  EXISTS {html_path.name}")
            else:
                print(f"  GET {url}")
                if not download_html(url, html_path):
                    continue
                time.sleep(0.5)

            # Discover sub-pages from sidebar (only if within depth limit)
            if depth < max_depth:
                html_text = html_path.read_text(encoding="utf-8")
                url_path = "/" + url.split("//")[1].split("/", 1)[-1]
                base_path = url_path.rstrip("/").rsplit("/", 1)[0] + "/"
                if base_path == "/":
                    base_path = url_path.rstrip("/") + "/"
                sub_links = extract_sidebar_links(html_text, base_path)
                for link in sub_links:
                    if link not in visited:
                        queue.append((link, depth + 1))


def main():
    parser = argparse.ArgumentParser(description="Extract links from a Nebius MHTML page and optionally download pages")
    parser.add_argument("--input", required=True, help="Path to MHTML file")
    parser.add_argument("--download", action="store_true",
                        help="After extracting links, download HTML pages from the JSONL index.")
    parser.add_argument("--max-depth", type=int, default=1,
                        help="Max crawl depth from each seed page (default: 1).")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = input_path.with_suffix(".jsonl")

    html = extract_html_from_mhtml(args.input)
    page_title = extract_title(html)
    links = extract_links(html)
    links = filter_links(links)

    with open(output_path, "w", encoding="utf-8") as f:
        for href, text, section in links:
            record = {
                "url": href,
                "page_title": text if text else "",
                "parent_page_title": section if section else page_title,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(links)} links to {output_path}")

    if args.download:
        pages_by_section = load_nebius_jsonl_index(output_path)
        download_pages(pages_by_section, output_path.parent, args.max_depth)


if __name__ == "__main__":
    main()
