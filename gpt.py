import requests
import os
from utils.embedder import Embedder

GPT_API_KEY = os.getenv("GPT_API_KEY")
if not GPT_API_KEY:
    raise ValueError("GPT_API_KEY not set.")
GPT_API_URL = "https://api.pawan.krd/v1/chat/completions"

embedder = Embedder()

def ask_gpt(user_input, context=None):
    try:
        matched = embedder.search(user_input)
        smart_context = "\n".join(matched)
    except:
        smart_context = context or "شما یک ربات پشتیبانی هستید."

    headers = {
        "Authorization": f"Bearer {GPT_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": smart_context},
            {"role": "user", "content": user_input}
        ]
    }
    try:
        response = requests.post(GPT_API_URL, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return "❌ خطا در ارتباط با API"
    except Exception as e:
        return "❌ خطا در پاسخ‌گویی هوش مصنوعی"
