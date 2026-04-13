"""FastAPI wrapper exposing the RAG QA agent over HTTP.

Run:
    uv run uvicorn agent.server:app --host 0.0.0.0 --port 8001
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

load_dotenv()

from agent.graph import AgentState, graph  # noqa: E402

_lf_handler: Any = None
if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
    from langfuse.langchain import CallbackHandler

    _lf_handler = CallbackHandler()


app = FastAPI()


class AskRequest(BaseModel):
    question: str
    tags: dict[str, str] = {}


class AskResponse(BaseModel):
    answer: str
    iterations: int
    ok: bool
    sources: list[str] = []
    history: list[dict[str, Any]] = []


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    state = AgentState(question=req.question)
    config: dict[str, Any] = {
        "callbacks": [_lf_handler] if _lf_handler is not None else [],
        "metadata": req.tags,
    }
    try:
        final = await asyncio.to_thread(graph.invoke, state, config)
    except Exception as e:  # noqa: BLE001
        logger.exception("graph.invoke failed")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    return AskResponse(
        answer=final.get("answer", ""),
        iterations=final.get("iteration", 0),
        ok=final.get("verify_ok", False),
        history=final.get("history", []),
    )
