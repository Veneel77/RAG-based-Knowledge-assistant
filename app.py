# app.py
import streamlit as st
import pickle, faiss, numpy as np
from sentence_transformers import SentenceTransformer
from gpt4all import GPT4All   # if using gpt4all
# from llama_cpp import Llama  # if using llama-cpp-python

EMBED_MODEL = "all-MiniLM-L6-v2"
INDEX_PATH = "faiss.index"
META_PATH = "meta.pkl"
TOP_K = 4

@st.cache_resource
def load_resources():
    # embeddings model
    embedder = SentenceTransformer(EMBED_MODEL)
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    # load local LLM (gpt4all or llama)
    llm = GPT4All(model="gpt4all-l13b-snoozy.bin")  # example; use the model you downloaded
    return embedder, index, meta, llm

embedder, index, meta, llm = load_resources()

def retrieve_contexts(query, k=TOP_K):
    q_emb = embedder.encode([query], convert_to_numpy=True)
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    D, I = index.search(q_emb, k)
    hits = []
    for idx in I[0]:
        hits.append(meta[idx])
    return hits

def make_prompt(question, contexts):
    prompt = "You are a helpful assistant. Use ONLY the provided contexts to answer and cite sources like (doc.pdf, page 2).\n\n"
    prompt += "CONTEXTS:\n"
    for i,c in enumerate(contexts):
        prompt += f"---\nSource: {c['doc_id']} | page: {c['page']}\n{c['text']}\n"
    prompt += f"\nQUESTION: {question}\nANSWER (short, cite sources):"
    return prompt

st.title("RAG Knowledge Assistant — Local/Free")
uploaded = st.file_uploader("Upload PDFs (optional) — upload then re-run ingest script to index", accept_multiple_files=True, type=["pdf"])
query = st.text_input("Ask a question about the uploaded docs / knowledge base")

if st.button("Ask") and query.strip():
    with st.spinner("Retrieving..."):
        contexts = retrieve_contexts(query)
    prompt = make_prompt(query, contexts)
    st.write("### Retrieved contexts")
    for c in contexts:
        st.markdown(f"- **{c['doc_id']}**, page {c['page']} — {c['text'][:200].replace('\\n',' ')}...")

    st.write("### Answer")
    # generate with gpt4all
    resp = llm.generate(prompt, max_tokens=512)  # API may vary by version - check docs
    st.write(resp)
    st.write("### Sources used (top K)")
    for c in contexts:
        st.write(f"- {c['doc_id']} (page {c['page']})")
