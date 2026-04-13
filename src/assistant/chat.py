import itertools
import os
import re
import sys
import threading
import time
from pathlib import Path

from mindbase_layer.retrieve_md import DocumentIndex

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


def _print_result(rank: int, score: float, source_name: str, header: str, snippet: str) -> None:
    divider = f"{_DIM}{'─' * 60}{_RESET}"
    score_bar = _score_bar(score)
    score_val = f"{_score_color(score)}{_BOLD}{score:.3f}{_RESET}"
    rank_tag = f"{_MAGENTA}#{rank}{_RESET}"
    src = f"{_CYAN}{source_name}{_RESET}"
    hdr = f"{_WHITE}{_BOLD}{header.strip()}{_RESET}"

    print(divider)
    print(f"  {rank_tag}  {score_bar} {score_val}  {src}")
    print(f"  {_BLUE}❯{_RESET} {hdr}")
    if snippet:
        print(f"  {_DIM}{snippet}{_RESET}")


if __name__ == "__main__":
    slides_dir = Path("slides")
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
            results, total = doc_index.search(user_input)
            stop.set()
            spinner.join()

            print(f"\n{_GREEN}{_BOLD}🤖 Found {total} results — showing top {len(results)}{_RESET}")
            for rank, (score, node) in enumerate(results, start=1):
                source_name = node.source.name if node.source else "Unknown"
                snippet = _highlight_snippet(node.body, user_input)
                _print_result(rank, score, source_name, node.header, snippet)
            print(f"{_DIM}{'─' * 60}{_RESET}\n")
    finally:
        pass