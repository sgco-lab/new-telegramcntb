from modules.gpt import ask_gpt
from modules.loader import load_context
from telebot import TeleBot
import os

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN not set.")
bot = TeleBot(BOT_TOKEN)

context = load_context()

@bot.message_handler(func=lambda message: True)
def handle(message):
    user_input = message.text
    response = ask_gpt(user_input, context)
    bot.reply_to(message, response)

print("🤖 Bot is running...")
bot.infinity_polling()
