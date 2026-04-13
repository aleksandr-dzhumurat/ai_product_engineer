"""LangGraph agent: RAG question-answering with verify+revise loop.

Graph shape:

    START -> retrieve -> answer -> verify
                                     |
                         ok=true ----+----> END
                                     |
                         ok=false ---+----> revise_query -> retrieve -> answer -> verify (loop)

Loop is capped at MAX_ITERATIONS total answer/revise calls.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, List

from connections import BaseConnection, ChromaConnection
from dotenv import load_dotenv
from ingestion import EmbeddingModel, fetch_embeddings
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from agent import prompts

DOTENV_FILE = os.environ.get("DOTENV_FILE", ".env")

print(f'load_dotenv({DOTENV_FILE}): {load_dotenv(DOTENV_FILE)}')

MAX_ITERATIONS = 3

VLLM_BASE_URL = os.environ["LLM_BASE_URL"]
VLLM_MODEL = os.environ["LLM_MODEL"]
LLM_API_KEY = os.environ["NEBIUS_API_KEY"]

EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
EMBEDDING_SIZE = int(os.environ.get("EMBEDDING_SIZE", "768"))
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "documents")


@dataclass
class AgentState:
    question: str
    query: str = ""
    n_results: int = 5
    context_chunks: List[str] = field(default_factory=list)
    answer: str = ""
    verify_ok: bool = False
    verify_issue: str = ""
    iteration: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)


def llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=VLLM_MODEL,
        base_url=VLLM_BASE_URL,
        api_key=LLM_API_KEY,
        temperature=0.0,
    )


_db_client: BaseConnection | None = None


def _get_db_client() -> BaseConnection:
    global _db_client
    if _db_client is None:
        _db_client = ChromaConnection()
    return _db_client


def _get_embedding_model() -> EmbeddingModel:
    return EmbeddingModel(name=EMBEDDING_MODEL_NAME, embedding_size=EMBEDDING_SIZE)


# ---- Nodes ------------------------------------------------------------

def retrieve_node(state: AgentState) -> dict:
    """Retrieve relevant chunks from the vector DB."""
    search_query = state.query or state.question
    model = _get_embedding_model()
    embeddings = fetch_embeddings(
        [search_query], host=OLLAMA_HOST, model=model, timeout=60.0,
    )
    client = _get_db_client()
    results = client.query(
        collection_name=COLLECTION_NAME,
        query_embeddings=embeddings,
        n_results=state.n_results,
        include=["documents", "metadatas"],
    )
    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]

    print(f"  Retrieved {len(chunks)} chunks for query: '{search_query}'")
    sources = set()
    for meta in metadatas:
        source = meta.get("source_file_name") or meta.get("source", "unknown")
        sources.add(source)
    for src in sorted(sources):
        print(f"    - {src}")

    return {
        "context_chunks": chunks,
        "query": search_query,
    }


def answer_node(state: AgentState) -> dict:
    """Generate an answer using retrieved context."""
    context = "\n\n---\n\n".join(state.context_chunks)
    response = llm().invoke([
        ("system", prompts.ANSWER_SYSTEM),
        ("user", prompts.ANSWER_USER.format(
            context=context,
            question=state.question,
        )),
    ])
    answer_text = response.content.strip()
    return {
        "answer": answer_text,
        "iteration": state.iteration + 1,
        "history": state.history + [{"node": "answer", "answer": answer_text[:200]}],
    }


def _parse_verify_json(text: str) -> tuple[bool, str]:
    """Extract {"ok": bool, "issue": str} from an LLM reply."""
    stripped = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()
    for candidate in [stripped, (re.search(r"\{.*\}", stripped, re.DOTALL) or type("", (), {"group": lambda s, _: ""})()).group(0)]:
        try:
            data = json.loads(candidate)
            return bool(data.get("ok", False)), str(data.get("issue", ""))
        except (json.JSONDecodeError, AttributeError):
            continue
    ok = bool(re.search(r'"ok"\s*:\s*true', text, re.IGNORECASE))
    return ok, text[:200]


def verify_node(state: AgentState) -> dict:
    """Verify whether the answer adequately addresses the question."""
    if not state.answer:
        return {
            "verify_ok": False,
            "verify_issue": "No answer was generated",
            "history": state.history + [{"node": "verify", "ok": False, "issue": "No answer"}],
        }

    response = llm().invoke([
        ("system", prompts.VERIFY_SYSTEM),
        ("user", prompts.VERIFY_USER.format(
            question=state.question,
            answer=state.answer,
            context="\n".join(state.context_chunks[:3]),
        )),
    ])
    ok, issue = _parse_verify_json(response.content)
    return {
        "verify_ok": ok,
        "verify_issue": issue,
        "history": state.history + [{"node": "verify", "ok": ok, "issue": issue}],
    }


def revise_query_node(state: AgentState) -> dict:
    """Revise the search query to retrieve better context."""
    response = llm().invoke([
        ("system", prompts.REVISE_QUERY_SYSTEM),
        ("user", prompts.REVISE_QUERY_USER.format(
            question=state.question,
            previous_query=state.query,
            issue=state.verify_issue,
        )),
    ])
    raw = response.content.strip()
    lines = raw.splitlines()
    new_query = lines[0].strip()
    n_results = state.n_results
    for line in lines[1:]:
        match = re.match(r"MORE_CHUNKS\s*=\s*(\d+)", line.strip())
        if match:
            n_results = min(int(match.group(1)), 20)
            break
    return {
        "query": new_query,
        "n_results": n_results,
        "history": state.history + [{"node": "revise_query", "query": new_query, "n_results": n_results}],
    }


def route_after_verify(state: AgentState) -> str:
    if state.verify_ok or state.iteration >= MAX_ITERATIONS:
        return "end"
    return "revise"


def _finalize_node(_state: AgentState) -> dict:
    return {}


# ---- Graph wiring -----------------------------------------------------

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("answer", answer_node)
    g.add_node("verify", verify_node)
    g.add_node("revise_query", revise_query_node)
    g.add_node("finalize", _finalize_node)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "answer")
    g.add_edge("answer", "verify")
    g.add_conditional_edges(
        "verify",
        route_after_verify,
        {"revise": "revise_query", "end": "finalize"},
    )
    g.add_edge("revise_query", "retrieve")
    g.add_edge("finalize", END)
    return g.compile()


graph = build_graph()
