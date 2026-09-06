"""Telegram bot with RAG-powered answers, image saving, and vision OCR.

Uses moonshotai/Kimi-K2.6 via Nebius API for text extraction from images.

Usage:
    TG_BOT_TOKEN="..." TG_BOT_DATA="/tmp/tg_images" DATA_DIR="$(pwd)" python src/rag/tg_bot.py
"""
from __future__ import annotations

import argparse
import base64
import logging
from datetime import datetime
from pathlib import Path

from agent import prompts
from agent.config import (
    COLLECTION_NAME,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    TG_BOT_DATA,
    TG_BOT_TOKEN,
    VISION_MODEL,
)
from agent.graph import AgentState, _get_db_client, graph
from openai import OpenAI
from telegram import ForceReply, Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

TOKEN = TG_BOT_TOKEN
DATA_DIR = Path(TG_BOT_DATA)

VISION_BASE_URL = LLM_BASE_URL
VISION_API_KEY = LLM_API_KEY

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def _ask_rag(question: str) -> dict:
    """Run the RAG graph and return the result dict."""
    state = AgentState(question=question)
    return graph.invoke(state)


def _ask_eng(question: str) -> dict:
    """Answer a TOEFL-style question directly via the LLM, no retrieval."""
    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": prompts.TOEFL_SYSTEM},
            {"role": "user", "content": prompts.TOEFL_USER.format(question=question)},
        ],
        temperature=0.0,
    )
    return {"answer": response.choices[0].message.content.strip()}


def _extract_text_from_image(file_path: Path) -> tuple[str, int, int]:
    """Extract text from an image using the configured vision model via Nebius API.

    Returns (extracted_text, input_tokens, output_tokens).
    """
    image_data = base64.b64encode(file_path.read_bytes()).decode("utf-8")
    client = OpenAI(base_url=VISION_BASE_URL, api_key=VISION_API_KEY)
    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                    },
                    {
                        "type": "text",
                        "text": (
                            "Transcribe the text in this image exactly as it appears, "
                            "preserving blanks/underscores. Be precise — do not guess or "
                            "fill in missing letters. Output ONLY the transcribed text, "
                            "as short and literal as possible, with no commentary, no "
                            "explanation, and no analysis."
                        ),
                    },
                ],
            }
        ],
        # Kimi-K2.6 is a reasoning model: it spends completion tokens on a
        # hidden "reasoning" pass before writing the final answer, and the
        # "be precise, don't guess" instruction makes it deliberate more
        # (observed ~4000 reasoning tokens on a dense screenshot), so this
        # needs generous headroom above that.
        max_tokens=16384,
    )
    input_tokens = 0
    output_tokens = 0
    usage = response.usage
    if usage:
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens
        logger.info(
            "Vision tokens — input: %d, output: %d, total: %d",
            input_tokens, output_tokens, input_tokens + output_tokens,
        )
    message = response.choices[0].message
    if message.content is None:
        logger.warning(
            "Vision response had no content (finish_reason=%s) — likely ran out of "
            "tokens mid-reasoning.",
            response.choices[0].finish_reason,
        )
        return "", input_tokens, output_tokens
    return message.content.strip(), input_tokens, output_tokens


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    mode = context.bot_data.get("mode", "rag")
    greeting = (
        "Send me a TOEFL-style question and I will tutor you through it."
        if mode == "eng"
        else "Send me a question about Nebius AI Cloud."
    )
    await update.message.reply_html(
        rf"Hi {user.mention_html()}! {greeting}",
        reply_markup=ForceReply(selective=True),
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    mode = context.bot_data.get("mode", "rag")
    if mode == "eng":
        await update.message.reply_text(
            "Send me a TOEFL-style question (multiple-choice, grammar, vocabulary, "
            "or writing prompt) and I will answer as a tutor.\n"
            "You can also send images of questions — I will extract the text and answer."
        )
    else:
        await update.message.reply_text(
            "Send me a text question and I will search the docs.\n"
            "You can also send images with text — I will extract it and search."
        )


def _run_question(mode: str, question: str) -> dict:
    """Answer a question according to the bot's mode."""
    if mode == "eng":
        return _ask_eng(question)
    return _ask_rag(question)


def _format_reply(mode: str, result: dict) -> str:
    from html import escape

    answer = result.get("answer", "No answer.")
    if mode == "eng":
        return escape(answer)

    iterations = result.get("iteration", 0)
    ok = result.get("verify_ok", False)
    return f"{escape(answer)}\n\n<i>iterations: {iterations}, verified: {ok}</i>"


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    question = update.message.text.strip()
    if not question:
        return

    user = update.effective_user
    mode = context.bot_data.get("mode", "rag")
    logger.info("Text from %s (%s) [mode=%s]: %s", user.username, user.id, mode, question[:120])

    await update.message.chat.send_action("typing")
    result = _run_question(mode, question)

    reply = _format_reply(mode, result)
    await update.message.reply_html(reply)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    photo = update.message.photo[-1]  # highest resolution

    user_dir = DATA_DIR / str(user.id)
    user_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{timestamp}_{photo.file_id[-8:]}.jpg"
    file_path = user_dir / file_name

    tg_file = await photo.get_file()
    await tg_file.download_to_drive(file_path)
    logger.info("Image from %s (%s) saved to %s", user.username, user.id, file_path)

    await update.message.chat.send_action("typing")

    # Extract text from image using vision LLM
    ocr_text, in_tok, out_tok = _extract_text_from_image(file_path)
    caption = (update.message.caption or "").strip()

    token_info = f"[vision tokens — input: {in_tok}, output: {out_tok}]"
    if ocr_text:
        logger.info("Vision extracted %d chars from %s", len(ocr_text), file_name)
        await update.message.reply_text(
            f"Image saved: {file_name}\nExtracted text:\n{ocr_text[:500]}\n\n{token_info}"
        )
    else:
        await update.message.reply_text(f"Image saved: {file_name}\nNo text detected. {token_info}")

    # Use caption if provided, otherwise use OCR text as the question
    question = caption or ocr_text
    if question:
        mode = context.bot_data.get("mode", "rag")
        await update.message.chat.send_action("typing")
        result = _run_question(mode, question)
        reply = _format_reply(mode, result)
        await update.message.reply_html(reply)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["rag", "eng"],
        default="rag",
        help="rag: answer from the Chroma-backed knowledge base (default). "
             "eng: TOEFL-style English tutor, no retrieval.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.mode == "rag":
        client = _get_db_client()
        try:
            client.get_collection(COLLECTION_NAME)
        except Exception as exc:
            raise SystemExit(
                f"Collection '{COLLECTION_NAME}' not found. "
                "Ingest documents first using src/rag/ingestion.py."
            ) from exc
        logger.info("Connected to collection: %s", COLLECTION_NAME)

    logger.info("Mode: %s", args.mode)
    logger.info("Image storage: %s", DATA_DIR)
    logger.info("Vision model: %s", VISION_MODEL)

    application = Application.builder().token(TOKEN).build()
    application.bot_data["mode"] = args.mode
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    logger.info("Bot started, polling...")
    application.run_polling()


if __name__ == "__main__":
    main()
