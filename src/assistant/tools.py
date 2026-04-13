import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class DocumentNode:
    header: str
    body: str | None = None
    source: Path | None = None
    parent: "DocumentNode | None" = None
    node_name: str = field(init=False)

    def __post_init__(self):
        self.node_name = hashlib.md5((self.header + self.body).encode()).hexdigest()


def _header_level(header: str) -> int:
    """Return the depth of a markdown header (number of leading #)."""
    m = re.match(r'^(#+)', header)
    return len(m.group(1)) if m else 0


def _parse_sections(content: str) -> list[tuple[int, str, str]]:
    """
    Parse markdown content into sections, skipping headers inside fenced code blocks.
    Returns [(line_num, header_line, body_text), ...].
    """
    sections: list[tuple[int, str, str]] = []
    in_code_block = False
    current_header: str | None = None
    current_line_num = 0
    body_lines: list[str] = []

    for line_num, line in enumerate(content.splitlines(keepends=True), 1):
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            if current_header is not None:
                body_lines.append(line)
            continue

        if not in_code_block and re.match(r"^#{1,6}\s+", line):
            if current_header is not None:
                sections.append((current_line_num, current_header, "".join(body_lines).strip()))
            current_header = line.rstrip()
            current_line_num = line_num
            body_lines = []
        else:
            if current_header is not None:
                body_lines.append(line)

    if current_header is not None:
        sections.append((current_line_num, current_header, "".join(body_lines).strip()))

    return sections


def read_md_nodes(file_path: Path) -> list[DocumentNode]:
    """Read a markdown file and return a list of DocumentNode, one per header section."""
    if not file_path.exists():
        print(f"Error: File not found at {file_path}")
        return []

    content = file_path.read_text(encoding="utf-8")
    nodes = [
        DocumentNode(header=header_line, body=body, source=file_path)
        for _, header_line, body in _parse_sections(content)
    ]

    parent_level: int = 0
    parent_node: DocumentNode | None = None
    previous_level: int = 0
    previous_node: DocumentNode | None = None

    for node in nodes:
        current_level = _header_level(node.header)

        if parent_level > 0:
            diff = current_level - parent_level
            if diff == 1:
                node.parent = parent_node
            elif diff >= 2:
                parent_level = previous_level
                parent_node = previous_node
                node.parent = parent_node

        previous_level = current_level
        previous_node = node
        parent_level = current_level
        parent_node = node

    return nodes

class DocumentIndex:
    def __init__(self, nodes: list[DocumentNode]):
        self._nodes = nodes
        self._vectorizer = TfidfVectorizer()
        corpus = [f"{node.header}\n{node.body}" for node in nodes]
        self._matrix = self._vectorizer.fit_transform(corpus)

    def search(self, query: str, top_k: int = 5) -> tuple[list[tuple[float, DocumentNode]], int]:
        """Return (top_k results, total above-zero count) ranked by TF-IDF cosine similarity."""
        query_vec = self._vectorizer.transform([query])
        scores = (self._matrix @ query_vec.T).toarray().flatten()
        total_nonzero = int((scores > 0).sum())
        top_indices = scores.argsort()[::-1][:top_k]
        results = [(float(scores[i]), self._nodes[i]) for i in top_indices if scores[i] > 0]
        return results, total_nonzero

    @classmethod
    def from_md_file(cls, file_path: Path) -> "DocumentIndex":
        """Factory: build a DocumentIndex from a markdown file."""
        nodes = read_md_nodes(file_path)
        if not nodes:
            raise ValueError(f"No nodes parsed from {file_path}")
        return cls(nodes)

    @classmethod
    def from_dir(cls, dir_path: Path) -> "DocumentIndex":
        """Factory: build a DocumentIndex from all .md files in a directory and its subdirectories."""
        nodes = []
        for md_file in sorted(dir_path.rglob("*.md")):
            nodes.extend(read_md_nodes(md_file))
        if not nodes:
            raise ValueError(f"No nodes parsed from any .md file in {dir_path}")
        return cls(nodes)
