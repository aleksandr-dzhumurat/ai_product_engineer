"""Minimal Telegram sender: post one message to a chat/channel and log a
'tg_message_sent' event (with the returned tg_message_id) to events_log.jsonl,
so a future feature can reply into that message's thread
(reply_to_message_id=tg_message_id) without re-deriving the id some other way.

Reference: src/scraping/data/newsfeed/src/newsfeed/telegram/client.py and
bot.py - this project only needs the send + 429-retry mechanics from there,
not newsfeed's scraping/translation/comment-thread-polling machinery.

Usage::

    PYTHONPATH="$(pwd)" python src/scraping/tg_pusher.py --text "hello"
    PYTHONPATH="$(pwd)" python src/scraping/tg_pusher.py --text "hello" --chat-id @some_channel
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

from utils import DEFAULT_EVENTS_LOG_PATH, configure_logging, push_event

configure_logging()
logger = logging.getLogger(__name__)

# Anchored to this file's own directory (not CWD) - see llm_client.py for why.
load_dotenv(Path(__file__).with_name(".env"))

TG_BOT_TOKEN = os.environ.get("TG_BOT_TOKEN", "")
CHANNEL_ID = os.environ.get("CHANNEL_ID", "")

API_URL = "https://api.telegram.org/bot{token}/{method}"
MAX_RATE_LIMIT_RETRIES = 5

# Telegram caps a single message at 4096 characters; 4000 leaves headroom
# for HTML entities (e.g. "&" -> "&amp;") growing the text during escaping.
TELEGRAM_MAX_MESSAGE_LENGTH = 4000


class TelegramError(Exception):
    def __init__(self, method: str, data: dict):
        self.method = method
        self.data = data
        self.error_code = data.get("error_code")
        self.retry_after = data.get("parameters", {}).get("retry_after")
        super().__init__(f"{method} failed: {data}")


DEFAULT_HTTP_TIMEOUT = 20  # seconds; a getUpdates long-poll caller must pass a larger http_timeout than its poll "timeout" param, or the local socket read can time out a moment before Telegram's long-poll response arrives


def _call(bot_token: str, method: str, http_timeout: float = DEFAULT_HTTP_TIMEOUT, **params) -> dict:
    """Telegram's 429 responses come with an exact retry_after (seconds) -
    honoring that beats guessing with exponential backoff, since retrying
    even slightly early just gets 429'd again."""
    url = API_URL.format(token=bot_token, method=method)
    debug_params = {k: v for k, v in params.items() if k != "text"}
    logger.info(f"-> {method} {debug_params}")

    for attempt in range(1, MAX_RATE_LIMIT_RETRIES + 1):
        resp = requests.post(url, json=params, timeout=http_timeout)
        data = resp.json()
        if data.get("ok"):
            logger.info(f"<- {method} ok")
            return data["result"]

        error = TelegramError(method, data)
        logger.error(f"<- {method} FAILED error_code={error.error_code} description={data.get('description')!r}")
        if error.error_code != 429 or attempt == MAX_RATE_LIMIT_RETRIES:
            raise error

        wait = (error.retry_after or 2 ** attempt) + 1
        logger.warning(f"429 rate limited on {method}, retrying in {wait}s (attempt {attempt}/{MAX_RATE_LIMIT_RETRIES})")
        time.sleep(wait)

    raise TelegramError(method, data)


def send_message(bot_token: str, chat_id, html_text: str, reply_to_message_id: int | None = None) -> dict:
    params = {"chat_id": chat_id, "text": html_text, "parse_mode": "HTML"}
    if reply_to_message_id is not None:
        params["reply_to_message_id"] = reply_to_message_id
    return _call(bot_token, "sendMessage", **params)


def get_linked_discussion_group_id(bot_token: str, channel_id) -> int | None:
    chat = _call(bot_token, "getChat", chat_id=channel_id)
    return chat.get("linked_chat_id")


def wait_for_discussion_thread(bot_token: str, group_chat_id: int, channel_message_id: int, timeout: float = 60.0) -> int | None:
    """A channel post only gets Telegram "comments" if the channel has a
    linked discussion group: Telegram auto-forwards every channel post into
    that group as a plain message, and *that* forwarded copy's message_id
    (in the group) is what later comments must set as reply_to_message_id -
    replying to the channel post's own id does nothing, channels don't
    support replies at all. There's no direct "get thread for this post"
    call, so this polls getUpdates for the forward to show up.

    The forward isn't instant - newsfeed's own production logs (this
    function's ancestor, UpdatesPoller.wait_for_comment_thread) saw 30-46s
    in some cases, hence the generous default timeout. Returns None if the
    forward doesn't show up in time."""
    poll_timeout = 20
    deadline = time.monotonic() + timeout
    offset = None
    while time.monotonic() < deadline:
        params = {"timeout": poll_timeout, "allowed_updates": ["message"]}
        if offset is not None:
            params["offset"] = offset
        try:
            updates = _call(bot_token, "getUpdates", http_timeout=poll_timeout + 20, **params)
        except requests.exceptions.ReadTimeout:
            # Expected occasionally: Telegram's long-poll can still run past
            # our http_timeout margin under real network latency - self-healing.
            continue

        for update in updates:
            offset = update["update_id"] + 1
            message = update.get("message")
            if not message or message.get("chat", {}).get("id") != group_chat_id:
                continue
            origin = message.get("forward_origin")
            if origin and origin.get("type") == "channel":
                forwarded_id = origin.get("message_id")
            else:
                forwarded_id = message.get("forward_from_message_id")  # Bot API < 7.0 fallback
            if forwarded_id == channel_message_id:
                return message["message_id"]
    return None


_INLINE_LINK_RE = re.compile(r"\[([^\]]+)\]\((.+?)\)")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_ITALIC_RE = re.compile(r"\*(.+?)\*")
_CODE_RE = re.compile(r"`(.+?)`")


def escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _link_sub(m: re.Match) -> str:
    label, href = m.group(1), m.group(2)
    if href.startswith("#"):
        return label  # in-page anchors don't work inside a Telegram message
    return f'<a href="{href}">{label}</a>'


def inline_md_to_html(text: str) -> str:
    """Convert inline Markdown (links, bold, italic, code) to Telegram's
    supported HTML subset (https://core.telegram.org/bots/api#html-style -
    no headings/lists, just <b>/<i>/<a>/<code>). Escapes first so literal
    &/</> already in the text aren't mangled by the tag substitutions that
    follow - same approach as newsfeed/telegram/formatter.py's
    inline_md_to_html. Bold must be substituted before italic: both use
    literal "*", so matching **text** first (turning it into <b>text</b>,
    with no "*" characters left over) is what stops the italic pass from
    misreading a bold marker's asterisks as its own delimiters."""
    text = escape(text)
    text = _INLINE_LINK_RE.sub(_link_sub, text)
    text = _BOLD_RE.sub(r"<b>\1</b>", text)
    text = _ITALIC_RE.sub(r"<i>\1</i>", text)
    text = _CODE_RE.sub(r"<code>\1</code>", text)
    return text


def markdown_to_telegram_html(markdown_text: str) -> str:
    """Line-level Markdown -> Telegram HTML: '#'/'##' headings become bold
    lines (Telegram has no heading tags), '- ' bullets become '• ' (no
    <ul>/<li> either), everything else goes through inline_md_to_html() for
    bold/links/code."""
    lines = []
    for line in markdown_text.splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append("")
        elif stripped.startswith("#"):
            # .strip("*") handles "### **Sub-heading**" (heading text already
            # wrapped in ** on its own) - without it, inline_md_to_html would
            # bold it again inside the <b> this branch already adds, producing
            # invalid nested <b><b>...</b></b>.
            heading_text = stripped.lstrip("#").strip().strip("*")
            lines.append(f"<b>{inline_md_to_html(heading_text)}</b>")
        elif stripped.startswith("- "):
            lines.append("• " + inline_md_to_html(stripped[2:]))
        else:
            lines.append(inline_md_to_html(stripped))
    return "\n".join(lines)


_SECTION_SPLIT_RE = re.compile(r"\n(?=## )")


def split_digest_sections(markdown_text: str) -> tuple[str | None, list[str]]:
    """Split a DIGEST_SYSTEM_PROMPT-shaped digest into (featured_section,
    [topic_section, ...]) on '## ' heading boundaries. featured_section is
    None if the model didn't produce one. The H1 title line before the
    first '## ' is discarded - it isn't its own message."""
    parts = [p.strip() for p in _SECTION_SPLIT_RE.split(markdown_text) if p.strip()]
    sections = [p for p in parts if p.startswith("## ")]

    featured = None
    topics = []
    for section in sections:
        if section[3:].lstrip().lower().startswith("featured"):
            featured = section
        else:
            topics.append(section)
    return featured, topics


def _trim_to_fit(html_section: str, max_length: int = TELEGRAM_MAX_MESSAGE_LENGTH) -> str:
    """Keep the heading and as many leading lines as fit under Telegram's
    per-message length cap, dropping the rest - topics are already ordered
    by usefulness (DIGEST_SYSTEM_PROMPT's Ordering section), so whatever
    gets trimmed is the least useful tail of that topic, not a random cut."""
    if len(html_section) <= max_length:
        return html_section
    lines = html_section.split("\n")
    kept = [lines[0]]
    length = len(lines[0])
    for line in lines[1:]:
        candidate_length = length + 1 + len(line)
        if candidate_length > max_length:
            break
        kept.append(line)
        length = candidate_length
    return "\n".join(kept)


def send_digest(
    markdown_text: str,
    chat_id: str = CHANNEL_ID,
    events_path=DEFAULT_EVENTS_LOG_PATH,
    delay_seconds: float = 1.0,
    thread_wait_timeout: float = 60.0,
) -> list[int]:
    """Post a DIGEST_SYSTEM_PROMPT-shaped digest to chat_id (a channel):
    Featured first (or topic 1 if the model produced no Featured section),
    then every remaining topic as a genuine Telegram **comment** under that
    post - not a same-chat reply, which channels don't support at all.
    A comment is a reply, in the linked discussion group, to the channel
    post's auto-forwarded copy there (see wait_for_discussion_thread).
    A section too long for one message is trimmed, not split into more
    messages (see _trim_to_fit). Falls back to posting remaining sections as
    independent (uncommented) channel messages if chat_id has no linked
    discussion group, or the forward doesn't show up within
    thread_wait_timeout. Returns every tg_message_id sent, root first."""
    featured, topics = split_digest_sections(markdown_text)
    sections = ([featured] if featured else []) + topics
    if not sections:
        logger.warning("send_digest: no '## ' sections found in the digest, nothing to send")
        return []

    root_html = _trim_to_fit(markdown_to_telegram_html(sections[0]))
    root_id = push_message(root_html, chat_id=chat_id, events_path=events_path)
    message_ids = [root_id]
    logger.info(f"sent section 1/{len(sections)}, tg_message_id={root_id} (channel post)")

    remaining = sections[1:]
    if not remaining:
        return message_ids

    comment_chat_id = chat_id
    comment_reply_id = None
    group_id = get_linked_discussion_group_id(TG_BOT_TOKEN, chat_id)
    if group_id is None:
        logger.warning(f"{chat_id} has no linked discussion group - posting remaining sections as independent channel messages, not comments")
    else:
        thread_id = wait_for_discussion_thread(TG_BOT_TOKEN, group_id, root_id, timeout=thread_wait_timeout)
        if thread_id is None:
            logger.warning(f"comment thread for message {root_id} not found within {thread_wait_timeout}s - posting remaining sections as independent channel messages")
        else:
            comment_chat_id, comment_reply_id = group_id, thread_id

    for i, section in enumerate(remaining, 2):
        time.sleep(delay_seconds)
        html = _trim_to_fit(markdown_to_telegram_html(section))
        msg_id = push_message(html, chat_id=comment_chat_id, events_path=events_path, reply_to_message_id=comment_reply_id)
        message_ids.append(msg_id)
        logger.info(f"sent section {i}/{len(sections)}, tg_message_id={msg_id}" + (f" (comment on {root_id})" if comment_reply_id else " (independent, no comment thread)"))
    return message_ids


def push_message(
    text: str,
    chat_id: str = CHANNEL_ID,
    events_path=DEFAULT_EVENTS_LOG_PATH,
    reply_to_message_id: int | None = None,
) -> int:
    """Send one message and log a 'tg_message_sent' event carrying the
    returned tg_message_id - the natural, already-unique id Telegram gives
    the message, saved now so a future feature can build a comment thread
    under it later without re-deriving the id."""
    result = send_message(TG_BOT_TOKEN, chat_id, text, reply_to_message_id=reply_to_message_id)
    tg_message_id = result["message_id"]
    event_fields = {
        "worker_name": "tg_pusher",
        "tg_message_id": tg_message_id,
        "chat_id": chat_id,
        "text": text,
    }
    if reply_to_message_id is not None:
        event_fields["reply_to_message_id"] = reply_to_message_id
    push_event(events_path, "tg_message_sent", **event_fields)
    logger.info(f"sent message to {chat_id}, tg_message_id={tg_message_id}")
    return tg_message_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Send one message to Telegram and log its message_id.")
    parser.add_argument("--text", required=True, help="Message text (HTML parse mode).")
    parser.add_argument("--chat-id", default=CHANNEL_ID, help="Chat/channel id to send to (default: CHANNEL_ID from .env).")
    args = parser.parse_args()

    if not TG_BOT_TOKEN:
        raise SystemExit("TG_BOT_TOKEN is not set")
    if not args.chat_id:
        raise SystemExit("No chat id given and CHANNEL_ID is not set")

    push_message(args.text, chat_id=args.chat_id)


if __name__ == "__main__":
    main()
