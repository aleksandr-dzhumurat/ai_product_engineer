import os

import requests
from jinja2 import Template

from telegram import __version__ as TG_VER

try:
    from telegram import __version_info__
except ImportError:
    __version_info__ = (0, 0, 0, 0, 0)  # type: ignore[assignment]

if __version_info__ < (20, 0, 0, "alpha", 1):
    raise RuntimeError(
        f"This example is not compatible with your current PTB version {TG_VER}. To view the "
        f"{TG_VER} version of this example, "
        f"visit https://docs.python-telegram-bot.org/en/v{TG_VER}/examples.html"
    )
from telegram import ForceReply, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters


TOKEN = os.environ['TG_BOT_TOKEN']
SEARCH_URL = os.environ['SEARCH_API_URL']

def prepare_html(input_json_array: dict):
    """
        lines = [
            {'content': 'I get that awful tickly dry cough'},
        ]
    """
    template_str = """
    {% for line in lines %}
    <i>result:</i> {{ line['content'] }}
    {% endfor %}
    """
    template = Template(template_str)
    html_content = template.render(lines=input_json_array)

    return html_content

def dialog_router(human_query: str):
    import re
    answer = f'Hello, bro? Do you wanna search: {human_query}'
    payload = {
        "text": human_query
    }
    response = requests.post(SEARCH_URL, json=payload)
    answer = [{'content': re.sub(r"<.*?>", "", i['content'])} for i in response.json()]
    # make a response
    return {'final_answer': True, 'answer': answer}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_html(
        rf"Hi {user.mention_html()}! Use /help for help",
        reply_markup=ForceReply(selective=True),
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    response = [
        "Send me a phrase and I will perform a search for you!"
    ]
    for i in response:
        await update.message.reply_text(i)

async def bot_dialog(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_tg = update.effective_user
    user = {'user_id': user_tg.id, 'user_name': user_tg.username}
    print(user)
    bot_response = dialog_router(update.message.text)
    if bot_response['final_answer']:
        responce = prepare_html(bot_response['answer'])
        await update.message.reply_html(responce)
    else:
        await update.message.reply_text(bot_response['answer'])

def main() -> None:
    """Start the bot."""
    application = Application.builder().token(TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot_dialog))
    # Run the bot until the user presses Ctrl-C
    application.run_polling()


if __name__ == "__main__":
    main()
