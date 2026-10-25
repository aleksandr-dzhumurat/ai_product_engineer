"""Nebius LLM client (OpenAI-compatible) for notebooks and src scripts.

Structured after src/scraping/llm_client.py. All chat completions go through
the Nebius API with NEBIUS_API_KEY; the base URL and default model can be
overridden via LLM_BASE_URL / LLM_MODEL (or NEBIUS_BASE_URL / NEBIUS_MODEL)
in .env.
"""
from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import backoff
import openai
from dotenv import load_dotenv
from openai import OpenAI

log = logging.getLogger(__name__)

# Anchored to this file's own location (src/llm.py -> repo root) so it always
# loads the project .env regardless of CWD. Doesn't override vars already
# exported in the shell.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class Config:
    NEBIUS_API_KEY: str = os.environ.get("NEBIUS_API_KEY", "")
    NEBIUS_BASE_URL: str = (
        os.environ.get("LLM_BASE_URL")
        or os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/")
    )
    NEBIUS_MODEL: str = (
        os.environ.get("LLM_MODEL")
        or os.environ.get("NEBIUS_MODEL", "google/gemma-3-27b-it")
    )


config = Config()
if not config.NEBIUS_API_KEY:
    log.warning("NEBIUS_API_KEY is not set - generate() calls will fail")


class RetryableAPIError(Exception):
    """Raised for transient API errors that can be retried (timeout, rate limit, etc)"""

class BlockedContentError(Exception):
    """Raised when content is blocked - should NOT be retried"""


@dataclass
class GenAIResponse:
    text: str
    total_tokens: int | None = None
    model: str | None = None
    generation_id: str | None = None


def generate_id(seed=None, limit=None):
    if seed is None:
        seed = int(time.time())
    res = str(
        hashlib
        .md5(str(seed).encode('utf-8'))
        .hexdigest()
    )
    if limit is not None:
        res = res[:limit]
    return res


g_client = None


def get_nebius_client() -> OpenAI:
    global g_client
    if g_client is None:
        g_client = OpenAI(
            base_url=config.NEBIUS_BASE_URL,
            api_key=config.NEBIUS_API_KEY,
        )
    return g_client


def validate_credentials() -> None:
    """Make one minimal Nebius call at startup so a bad/expired
    NEBIUS_API_KEY aborts immediately with a clear error, instead of only
    surfacing later inside a generate() call."""
    try:
        generate(get_nebius_client(), "ping", system_prompt="Reply with just: pong")
    except Exception as e:
        if _is_invalid_key_error(e):
            raise RuntimeError(f"NEBIUS_API_KEY rejected by Nebius ({config.NEBIUS_MODEL}): {e}") from e
        log.warning(
            "⚠️  Nebius credential check failed with a non-auth error (%s: %s) - continuing, this may be transient",
            type(e).__name__, e,
        )


def _is_invalid_key_error(exc) -> bool:
    return isinstance(exc, openai.AuthenticationError)


def _log_giveup(details):
    exc = details["exception"]
    if _is_invalid_key_error(exc):
        log.error("🔑 NEBIUS_API_KEY is invalid or rejected by Nebius — check the key in .env. Details: %s", exc)
    else:
        log.error("❌ All %d Nebius attempts failed: %s", details["tries"], exc)


@backoff.on_exception(
    backoff.expo,
    (RetryableAPIError, openai.APIError),
    max_tries=5,
    max_time=60,
    giveup=lambda e: isinstance(e, openai.APIError) and not isinstance(e, (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError)),
    on_giveup=_log_giveup,
)
def generate(api_client, user_prompt, system_prompt="You are a helpful assistant for business analytics", model=None):
    model = model or config.NEBIUS_MODEL
    response = api_client.chat.completions.create(
        model=model,
        temperature=0.3,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    if response is None:
        log.error("❌ API returned None response. Input text (first 300 chars): %s", user_prompt[:300])
        raise RetryableAPIError("API returned None response")

    choice = response.choices[0] if response.choices else None
    response_text = choice.message.content if choice is not None else None
    if response_text is None:
        finish_reason = getattr(choice, "finish_reason", "unknown") if choice is not None else "unknown"
        if finish_reason == "content_filter":
            log.warning(
                "⚠️  CONTENT BLOCKED by Nebius API. Input text (first 300 chars): %s",
                user_prompt[:300],
            )
            # Don't retry blocked content - it's permanent
            raise BlockedContentError(f"Content blocked: {finish_reason}")
        else:
            log.error("❌ No text in response, finish_reason=%s. Input text (first 300 chars): %s", finish_reason, user_prompt[:300])
            raise RetryableAPIError("API returned response without text")

    if response.usage:
        log.info(
            "🔢 tokens (model=%s): prompt=%d completion=%d total=%d",
            model, response.usage.prompt_tokens, response.usage.completion_tokens, response.usage.total_tokens,
        )

    return GenAIResponse(
        text=response_text,
        total_tokens=response.usage.total_tokens if response.usage else None,
        model=model,
        generation_id=generate_id(user_prompt),
    )


def prompt_generation(candidates):
    prompt = f"""
      Next rows below is an item_id reviews.
      {candidates}
      Utilize reviews to determine the best item_id.
      Avoid including actual reviews; rephrase them succinctly.
      Take in account number of reviews for same item_id and sentiment of review.
      Keep the recommendation under 50 words. Avoid starting with "Based on reviews"; opt for a more creative approach!
      Recommendation:
  """
    return prompt


def recs_generation(candidates):
    candidates = '\n'.join(["item_id: %s; review: %s" % (i['asin'], i['content']) for i in candidates])
    prompt = prompt_generation(candidates)
    return generate(get_nebius_client(), user_prompt=prompt)
