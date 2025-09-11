# evaluate.py
import json, pickle, faiss
from sentence_transformers import SentenceTransformer
import numpy as np
from collections import Counter
import re

EMBED_MODEL = "all-MiniLM-L6-v2"
INDEX_PATH = "faiss.index"
META_PATH = "meta.pkl"

def normalize_text(t):
    t = t.lower().strip()
    t = re.sub(r"\s+", " ", t)
    return t

def precision_at_k(query, gold_doc_ids, k=4):
    embedder = SentenceTransformer(EMBED_MODEL)
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH,"rb") as f:
        meta = pickle.load(f)
    q_emb = embedder.encode([query], convert_to_numpy=True)
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    D,I = index.search(q_emb, k)
    hits = [meta[idx]["doc_id"] for idx in I[0]]
    return sum(1 for h in hits if h in gold_doc_ids)/k

# Example usage:
# dataset = [{"query":"What is IBM cloud strategy?", "gold":["ibm_report.pdf"]}, ...]
# compute mean precision@k
