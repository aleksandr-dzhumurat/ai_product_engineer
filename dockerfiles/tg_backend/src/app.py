"""Telegram interface for the review search API (long polling)."""

import logging
import os

import httpx
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

logger = logging.getLogger(__name__)
SEARCH_URL = os.getenv("SEARCH_API_URL", "http://api:8000/search")
MESSAGE_LIMIT = 3500  # Also safe for non-BMP characters counted as UTF-16 units.


def split_messages(text, limit=MESSAGE_LIMIT):
    """Keep plain-text messages below Telegram's UTF-16 length limit."""
    chunks, current, size = [], [], 0
    for character in text:
        width = 2 if ord(character) > 0xFFFF else 1
        if size + width > limit:
            chunks.append("".join(current))
            current, size = [], 0
        current.append(character)
        size += width
    if current:
        chunks.append("".join(current))
    return chunks


async def dialog_router(human_query, client):
    response = await client.post(SEARCH_URL, json={"text": human_query, "num": 5})
    response.raise_for_status()
    results = response.json()
    if not isinstance(results, list) or any(
        not isinstance(item, dict)
        or not isinstance(item.get("content"), str)
        or not isinstance(item.get("asin"), str)
        for item in results
    ):
        raise ValueError("Invalid search API response")
    return results[:5]


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_message:
        await update.effective_message.reply_text(
            "Отправьте поисковый запрос, например dry cough. Я найду отзывы на товары. /help — помощь."
        )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_message:
        await update.effective_message.reply_text(
            "Введите слова для поиска по отзывам. Ответ содержит найденные тексты, "
            "а не медицинские рекомендации."
        )


async def bot_dialog(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.effective_message
    if message is None or not message.text:
        return
    query = message.text.strip()
    if not query or len(query) > 2000:
        await message.reply_text("Введите непустой запрос длиной до 2000 символов.")
        return
    try:
        results = await dialog_router(query, context.application.bot_data["search_client"])
    except (httpx.HTTPError, ValueError):
        # Do not log URLs, tokens, user messages or upstream response bodies.
        logger.warning("Search API request failed")
        await message.reply_text("Поиск временно недоступен. Попробуйте позже.")
        return
    if not results:
        await message.reply_text("Ничего не найдено. Попробуйте изменить запрос.")
        return
    answer = "\n\n".join(
        f"{number}. Товар: {item['asin']}\n{item['content']}"
        for number, item in enumerate(results, start=1)
    )
    for chunk in split_messages(answer):
        # No parse_mode: HTML in reviews remains text, not Telegram markup.
        await message.reply_text(chunk, parse_mode=None)


async def post_init(application):
    application.bot_data["search_client"] = httpx.AsyncClient(timeout=40)


async def post_shutdown(application):
    client = application.bot_data.get("search_client")
    if client:
        await client.aclose()


async def on_error(update, context):
    logger.error("Telegram update handling failed (%s)", type(context.error).__name__)


def main():
    token = os.getenv("TG_BOT_TOKEN")
    if not token:
        raise RuntimeError("Set TG_BOT_TOKEN in .env before starting the bot")
    logging.basicConfig(level=logging.INFO)
    # Telegram request URLs include the bot token.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    application = (
        Application.builder().token(token)
        .post_init(post_init).post_shutdown(post_shutdown).build()
    )
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot_dialog))
    application.add_error_handler(on_error)
    application.run_polling()


if __name__ == "__main__":
    main()
