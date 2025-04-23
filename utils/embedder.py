# utils/embedder.py
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os
import pickle

DATA_FILE = "data/context.txt"
EMBED_FILE = "embeddings/index.pkl"

class Embedder:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.passages = []

    def load_data(self):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            content = f.read()
            self.passages = [p.strip() for p in content.split("\n") if len(p.strip()) > 30]

    def build_index(self):
        self.load_data()
        X = self.vectorizer.fit_transform(self.passages)
        with open(EMBED_FILE, "wb") as f:
            pickle.dump((self.vectorizer, self.passages, X), f)

    def search(self, query, top_k=3):
        with open(EMBED_FILE, "rb") as f:
            vectorizer, passages, X = pickle.load(f)
        q_vec = vectorizer.transform([query])
        scores = (q_vec @ X.T).toarray()[0]
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [passages[i] for i in top_idx if scores[i] > 0.01]
