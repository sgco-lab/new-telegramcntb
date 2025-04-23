# modules/loader.py
def load_context():
    try:
        with open("data/context.txt", "r", encoding="utf-8") as f:
            return f.read()
    except:
        return "شما یک ربات پشتیبانی هوشمند هستید."
