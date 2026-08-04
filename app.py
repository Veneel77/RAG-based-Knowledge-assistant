import os
import pickle
import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from ingest_index import build_index
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Nova AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------
# Config
# ---------------------------
EMBED_MODEL = "all-MiniLM-L6-v2"
INDEX_PATH = "faiss.index"
META_PATH = "meta.pkl"
# Automatically load every PDF inside docs/
PDFS = []

if not os.path.exists("docs"):
    os.makedirs("docs")

for file in os.listdir("docs"):
    if file.lower().endswith(".pdf"):
        PDFS.append(os.path.join("docs", file))
TOP_K = 4

# ---------------------------
# Load Resources
# ---------------------------
@st.cache_resource
def load_resources():

    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        build_index(PDFS, INDEX_PATH, META_PATH)

    embedder = SentenceTransformer(EMBED_MODEL)
    index = faiss.read_index(INDEX_PATH)

    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)

    return embedder, index, meta


embedder, index, meta = load_resources()

# ---------------------------
# Retrieval
# ---------------------------
def retrieve(query):

    emb = embedder.encode([query], convert_to_numpy=True)

    emb = emb / (
        np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
    )

    D, I = index.search(emb, TOP_K)

    return [meta[i] for i in I[0]]


# ---------------------------
# Session State
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:

    st.title("🤖 Nova AI")

    st.button("➕ New Chat")

    st.divider()

    st.subheader("Recent Chats")

    st.info("Chat history coming soon")

    st.divider()

    st.subheader("Documents")
    uploaded_file = st.file_uploader(
        "📄 Upload PDF",
        type=["pdf"]
        )


    for pdf in PDFS:
        st.success(os.path.basename(pdf))

    st.divider()

    st.toggle("Use RAG", value=True)

    st.selectbox(
        "Model",
        [
            "Gemini 2.5 Flash",
            "Llama 3",
            "GPT-4",
        ],
    )
# ---------------------------
# Handle Upload
# ---------------------------

if uploaded_file is not None:

    save_path = os.path.join("docs", uploaded_file.name)

    if os.path.exists(save_path):
        st.warning("This PDF already exists.")
    else:
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success("PDF uploaded successfully!")

        # Reload all PDFs from docs/
        pdfs = [
            os.path.join("docs", f)
            for f in os.listdir("docs")
            if f.lower().endswith(".pdf")
        ]

        build_index(
            pdfs,
            INDEX_PATH,
            META_PATH
        )

        st.cache_resource.clear()

        st.success("Knowledge Base Updated!")

        st.rerun()

# ---------------------------
# Main Window
# ---------------------------
st.title("🤖 Nova AI")

st.caption("Enterprise AI Assistant")

if len(st.session_state.messages) == 0:

    st.markdown(
        """
# 👋 Hello Veneel

How can I help you today?
"""
    )

# ---------------------------
# Existing Chat
# ---------------------------
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

# ---------------------------
# Chat Input
# ---------------------------
prompt = st.chat_input("Message Nova AI...")

if prompt:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):

        st.markdown(prompt)

    contexts = retrieve(prompt)
    st.write("DEBUG")
    st.write(contexts)


    context_text = ""

    for c in contexts:
        context_text += f"""
    Document: {c['doc_id']}
    Page: {c['page']}

    {c['text']}

    """

    prompt_for_llm = f"""
    You are an intelligent RAG assistant.

    Answer ONLY from the provided context.

    Context:

    {context_text}

    Question:

    {prompt}

    Answer:
    """

    try:

        response = model.generate_content(prompt_for_llm)

        answer = response.text

    except Exception as e:

        answer = f"""
    ⚠️ Gemini unavailable.

    Retrieved Context

    {context_text[:1500]}
    """

    with st.chat_message("assistant"):

        st.markdown(answer)

        with st.expander("📄 Sources"):

            for c in contexts:

                st.markdown(
                    f"""
**{c['doc_id']}**

Page **{c['page']}**

{c['text'][:300]}...
"""
                )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )