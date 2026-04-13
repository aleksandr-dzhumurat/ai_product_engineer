import os
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import shutil

from langfuse import get_client
from pydantic_ai import Agent, RunContext
from pydantic_ai.agent import InstrumentationSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.nebius import NebiusProvider

from prompts import (
    PROJECT_MANAGER_INSTRUCTIONS,
    RETRIEVAL_AGENT_INSTRUCTIONS,
    home_dir_prompt,
)
from mindbase_layer.retrieve_md import DocumentIndex, DocumentNode

Agent.instrument_all(InstrumentationSettings(include_content=True, version=1))
langfuse = get_client()


@dataclass
class SupportDependencies:
    home_dir: Path


@dataclass
class RetrievalDependencies:
    md_path: Path


_main_model = OpenAIChatModel(
    'Qwen/Qwen3-32B',
    provider=NebiusProvider(api_key=os.getenv('NEBIUS_API_KEY'))
)

_retrieval_model = OpenAIChatModel(
    'Qwen/Qwen3-30B-A3B-Instruct-2507',
    provider=NebiusProvider(api_key=os.getenv('NEBIUS_API_KEY'))
)

retrieval_agent = Agent(
    _retrieval_model,
    instructions=RETRIEVAL_AGENT_INSTRUCTIONS,
    deps_type=RetrievalDependencies,
    output_type=str,
)


@retrieval_agent.tool
def query_documents(ctx: RunContext[RetrievalDependencies], query: str) -> str:
    """Search the indexed markdown document(s) using TF-IDF cosine similarity."""
    md_path = ctx.deps.md_path
    if md_path.is_dir():
        index = DocumentIndex.from_dir(md_path)
    elif md_path.suffix == '.srt':
        index = DocumentIndex.from_srt_file(md_path)
    else:
        index = DocumentIndex.from_md_file(md_path)
    results, total = index.search(query, top_k=5)
    if not results:
        return f"No results found for query: '{query}'"
    lines = [f"Found {len(results)} of {total} matching nodes:"]
    for score, node in results:
        lines.append(f"\n[{score:.4f}] {node.header}")
        if node.body:
            lines.append(node.body[:500])
    return "\n".join(lines)


project_manager_agent = Agent(
    _main_model,
    instructions=PROJECT_MANAGER_INSTRUCTIONS,
    deps_type=SupportDependencies
)


@project_manager_agent.system_prompt
def add_home_dir(ctx: RunContext[SupportDependencies]) -> str:
    return home_dir_prompt(ctx.deps.home_dir)


SEARCH_DIRS = ["Downloads", "Documents", "PycharmProjects"]


@project_manager_agent.tool
def file_search(ctx: RunContext[SupportDependencies], filename: str) -> str:
    """Search for a file by name across Downloads, Documents and PycharmProjects under home_dir.
    If filename is an absolute path, checks existence directly without searching."""
    path = Path(filename)
    if path.is_absolute():
        return str(path) if path.exists() else f"File not found: {path}"
    matches = []
    for search_dir in SEARCH_DIRS:
        base = ctx.deps.home_dir / search_dir
        if base.is_dir():
            matches.extend(base.rglob(filename))
    if not matches:
        return f"File '{filename}' not found in {SEARCH_DIRS}"
    if len(matches) == 1:
        return str(matches[0])
    return "Multiple files found:\n" + "\n".join(str(p) for p in matches)


@project_manager_agent.tool
def file_fuzzy_search(ctx: RunContext[SupportDependencies], query: str) -> str:
    """Fuzzy search for a file by query across Downloads, Documents and PycharmProjects.
    Indexes all filenames with TF-IDF and returns top-10 matches. Use as fallback when file_search returns nothing."""
    nodes = []
    for search_dir in SEARCH_DIRS:
        base = ctx.deps.home_dir / search_dir
        if base.is_dir():
            for path in base.rglob("*"):
                if path.is_file():
                    normalized = path.stem.replace("_", " ").replace("-", " ")
                    nodes.append(DocumentNode(header=path.name, body=normalized, source=path))
    if not nodes:
        return "No files found in search directories."
    index = DocumentIndex(nodes)
    results, _ = index.search(query, top_k=10)
    if not results:
        return f"No files matching '{query}' found."
    lines = [f"Top {len(results)} fuzzy matches for '{query}':"]
    for score, node in results:
        lines.append(f"  [{score:.4f}] {node.source}")
    return "\n".join(lines)


@project_manager_agent.tool
async def generate_subtitles(_ctx: RunContext[SupportDependencies], video_path: str, language: str = "en") -> str:
    """Generate an SRT subtitles file from a video. language is a BCP-47 code e.g. 'en', 'ru'. Expects a full resolved path."""
    mp3_path = str(extract_audio_pipeline(video_path))
    srt_path = transcribe(mp3_path, language=language)
    return f"Subtitles saved to: {srt_path}"

@project_manager_agent.tool
async def search_file_content(ctx: RunContext[SupportDependencies], md_path: str, query: str) -> str:
    """Search the content of a markdown file or directory using the retrieval agent."""
    path = Path(md_path).expanduser().resolve()
    result = await retrieval_agent.run(query, deps=RetrievalDependencies(md_path=path), usage=ctx.usage)
    return result.output
