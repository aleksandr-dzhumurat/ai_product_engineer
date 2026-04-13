import itertools
import re
import sys
import threading
import time
from pathlib import Path

from mindbase_layer.utils.retrieve_md import DocumentIndex, DocumentNode

# ANSI color codes
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_CYAN   = "\033[36m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_RED    = "\033[31m"
_BLUE   = "\033[34m"
_MAGENTA= "\033[35m"
_WHITE  = "\033[97m"
_BG_DARK= "\033[48;5;236m"


def _clickable(url: str, text: str | None = None) -> str:
    """Wrap text in an OSC 8 hyperlink escape sequence (clickable in modern terminals)."""
    label = text or url
    return f"\033]8;;{url}\033\\{label}\033]8;;\033\\"


def _score_color(score: float) -> str:
    if score >= 0.6:
        return _GREEN
    elif score >= 0.3:
        return _YELLOW
    else:
        return _RED


def _score_bar(score: float, width: int = 8) -> str:
    filled = round(score * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"{_score_color(score)}{bar}{_RESET}"


def _highlight_snippet(text: str, query: str, snippet_length: int = 120) -> str:
    if not text:
        return ""

    terms = [re.escape(t) for t in query.split() if t.strip()]
    if not terms:
        snippet = text[:snippet_length]
        return (snippet + "...") if len(text) > snippet_length else snippet

    pattern = re.compile(r'(' + '|'.join(terms) + r')', re.IGNORECASE)
    match = pattern.search(text)

    if match:
        start = max(0, match.start() - snippet_length // 2)
        end = min(len(text), match.end() + snippet_length // 2)
        snippet = text[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet += "..."

        highlighted = pattern.sub(lambda m: f"{_BOLD}\033[93m{m.group(0)}{_RESET}", snippet)
        return highlighted.replace("\n", " ")
    else:
        snippet = text[:snippet_length]
        if len(text) > snippet_length:
            snippet += "..."
        return snippet.replace("\n", " ")


def _spinner(stop_event: threading.Event) -> None:
    for frame in itertools.cycle(["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]):
        if stop_event.is_set():
            break
        sys.stdout.write(f"\r{_CYAN}🤖 Searching... {frame}{_RESET} ")
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write("\r" + " " * 30 + "\r")
    sys.stdout.flush()


def _source_label(node: DocumentNode) -> str:
    """Build a (possibly clickable) source label from a DocumentNode."""
    source_name = node.source.name if node.source else "Unknown"
    if node.source and node.line_start is not None:
        file_uri = f"file://{node.source.resolve()}#{node.line_start}"
        label = f"{source_name}:{node.line_start}"
        return _clickable(file_uri, f"{_CYAN}{label}{_RESET}")
    return f"{_CYAN}{source_name}{_RESET}"


def _print_result(rank: int, score: float, node: DocumentNode, snippet: str) -> None:
    divider = f"{_DIM}{'─' * 60}{_RESET}"
    score_bar = _score_bar(score)
    score_val = f"{_score_color(score)}{_BOLD}{score:.3f}{_RESET}"
    rank_tag = f"{_MAGENTA}#{rank}{_RESET}"
    src = _source_label(node)
    hdr = f"{_WHITE}{_BOLD}{node.header.strip()}{_RESET}"

    print(divider)
    print(f"  {rank_tag}  {score_bar} {score_val}  {src}")
    print(f"  {_BLUE}❯{_RESET} {hdr}")
    if snippet:
        print(f"  {_DIM}{snippet}{_RESET}")


if __name__ == "__main__":
    slides_dir = Path("slides")
    links_jsonl = slides_dir / "links.jsonl"

    print(f"\n{_CYAN}{_BOLD}{'═' * 60}{_RESET}")
    print(f"{_CYAN}{_BOLD}  🤖 Building DocumentIndex from 'slides'...{_RESET}")
    doc_index = DocumentIndex.from_dir(slides_dir)
    file_count = len(list(slides_dir.rglob("*.md")))
    section_count = len(doc_index._nodes)
    total_lines = sum(len((n.body or "").splitlines()) for n in doc_index._nodes)
    print(
        f"{_GREEN}  ✔ Indexed {_BOLD}{file_count}{_RESET}{_GREEN} files"
        f" · {_BOLD}{section_count}{_RESET}{_GREEN} sections"
        f" · {_BOLD}{total_lines}{_RESET}{_GREEN} lines{_RESET}"
    )

    # Links index
    if not links_jsonl.exists():
        raise SystemExit(
            f"\n{_RED}  ✘ {links_jsonl} not found.{_RESET}\n"
            f"  Run first:\n"
            f'    PYTHONPATH="$(pwd)" python src/rag/files_processing.py --show-links --input-dir slides\n'
        )
    links_index = DocumentIndex.from_jsonl(links_jsonl)
    print(
        f"{_GREEN}  ✔ Links: {_BOLD}{len(links_index._nodes)}{_RESET}{_GREEN} entries"
        f" from {_BOLD}{links_jsonl}{_RESET}"
    )

    print(f"{_CYAN}{_BOLD}{'═' * 60}{_RESET}")
    print(f"  {_DIM}Type your query or 'exit' to quit.{_RESET}\n")
    try:
        while True:
            user_input = input(f"{_BOLD}{_YELLOW}👤 You:{_RESET} ").strip()
            if user_input.lower() == "exit":
                print(f"\n{_CYAN}🤖 Goodbye!{_RESET}\n")
                break
            stop = threading.Event()
            spinner = threading.Thread(target=_spinner, args=(stop,), daemon=True)
            spinner.start()
            results = doc_index.search(user_input)
            link_results = links_index.search(user_input)
            stop.set()
            spinner.join()

            print(f"\n{_GREEN}{_BOLD}🤖 Found {len(results)} results{_RESET}")
            for rank, (score, node) in enumerate(results, start=1):
                snippet = _highlight_snippet(node.body, user_input)
                _print_result(rank, score, node, snippet)

            if link_results:
                print(f"\n{_GREEN}{_BOLD}🔗 Found {len(link_results)} relevant links{_RESET}")
                for rank, (score, node) in enumerate(link_results, start=1):
                    url = node.body or ""
                    clickable_url = _clickable(url) if url.startswith("http") else url
                    _print_result(rank, score, node, clickable_url)
            print(f"{_DIM}{'─' * 60}{_RESET}\n")
    finally:
        pass