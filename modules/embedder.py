# modules/embedder.py
import os
from sentence_transformers import SentenceTransformer
import faiss
import pickle

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts):
    vectors = model.encode(texts, show_progress_bar=False)
    return vectors

def build_knowledge_base(texts, save_path="data/faiss_index"):
    vectors = embed_texts(texts)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    # Save FAISS index
    faiss.write_index(index, f"{save_path}.index")

    # Save text mapping
    with open(f"{save_path}.pkl", "wb") as f:
        pickle.dump(texts, f)

def load_knowledge_base(path="data/faiss_index"):
    index = faiss.read_index(f"{path}.index")
    with open(f"{path}.pkl", "rb") as f:
        texts = pickle.load(f)
    return index, texts

def search_knowledge(query, top_k=3):
    index, texts = load_knowledge_base()
    q_vector = embed_texts([query])
    D, I = index.search(q_vector, top_k)
    return [texts[i] for i in I[0]]
