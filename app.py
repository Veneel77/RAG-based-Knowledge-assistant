# app.py
import os
import pickle
import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from ingest_index import build_index
from dotenv import load_dotenv
load_dotenv()   # reads .env

HF_TOKEN = os.getenv("HF_TOKEN")

# from gpt4all import GPT4All   # if using gpt4all
# from llama_cpp import Llama  # if using llama-cpp-python
os.environ["STREAMLIT_HOME"] = os.path.join(os.getcwd(), ".streamlit")
os.makedirs(os.environ["STREAMLIT_HOME"], exist_ok=True)



# ----------------------------
# Config
# ----------------------------
EMBED_MODEL = "all-MiniLM-L6-v2"
INDEX_PATH = "faiss.index"
META_PATH = "meta.pkl"
PDFS = ["docs/AI1.pdf", "docs/aiimpact.pdf"]  # PDFs inside your repo
TOP_K = 4


# ----------------------------
# Load or build index
# ----------------------------
@st.cache_resource
def load_resources():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        st.warning("Index not found. Building index from PDFs...")
        build_index(PDFS, out_index=INDEX_PATH, out_meta=META_PATH)

    embedder = SentenceTransformer(EMBED_MODEL)
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)

    # Optional: load local LLM if you want
    # llm = GPT4All(model="gpt4all-l13b-snoozy.bin")
    # return embedder, index, meta, llm
    return embedder, index, meta


embedder, index, meta = load_resources()


# ----------------------------
# Retrieval
# ----------------------------
def retrieve_contexts(query, k=TOP_K):
    q_emb = embedder.encode([query], convert_to_numpy=True)
    q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-9)
    D, I = index.search(q_emb, k)
    hits = [meta[idx] for idx in I[0]]
    return hits


def make_prompt(question, contexts):
    prompt = "You are a helpful assistant. Use ONLY the provided contexts to answer and cite sources like (doc.pdf, page 2).\n\n"
    prompt += "CONTEXTS:\n"
    for c in contexts:
        prompt += f"---\nSource: {c['doc_id']} | page: {c['page']}\n{c['text']}\n"
    prompt += f"\nQUESTION: {question}\nANSWER (short, cite sources):"
    return prompt


# ----------------------------
# Streamlit UI
# ----------------------------
st.title("📚 RAG Knowledge Assistant")

query = st.text_input("Ask a question about the knowledge base")

if st.button("Ask") and query.strip():
    with st.spinner("Retrieving..."):
        contexts = retrieve_contexts(query)
    prompt = make_prompt(query, contexts)

    # Show retrieved contexts
    st.write("### Retrieved contexts")
    for c in contexts:
        preview = c["text"][:200].replace("\n", " ")
        st.markdown(f"- **{c['doc_id']}**, page {c['page']} — {preview}...")

    # Show answer
    st.write("### Answer")
    # If using GPT4All / Llama, uncomment below
    # resp = llm.generate(prompt, max_tokens=512)
    # st.write(resp)
    st.info("LLM generation disabled in this demo. Retrieved contexts shown above.")

    # Show sources
    st.write("### Sources used")
    for c in contexts:
        st.write(f"- {c['doc_id']} (page {c['page']})")
