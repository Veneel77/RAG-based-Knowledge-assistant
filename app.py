import os
import pickle
import hashlib

import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import google.generativeai as genai
from dotenv import load_dotenv


# ============================================================
# 1. PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Nova AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. ENVIRONMENT / SECRETS
# ============================================================

load_dotenv()


def get_secret(name, default=None):
    """
    Get configuration from Streamlit Secrets first.
    Fall back to .env for local development.
    """

    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass

    return os.getenv(name, default)


GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
GEMINI_MODEL = get_secret(
    "GEMINI_MODEL",
    "gemini-2.5-flash"
)


# ============================================================
# 3. GEMINI CONFIGURATION
# ============================================================

if not GEMINI_API_KEY:
    st.error(
        "Gemini API key is not configured. "
        "Add GEMINI_API_KEY to Streamlit Secrets."
    )
    st.stop()


genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel(GEMINI_MODEL)


# ============================================================
# 4. APPLICATION CONFIG
# ============================================================

EMBED_MODEL = "all-MiniLM-L6-v2"

INDEX_PATH = "faiss.index"
META_PATH = "meta.pkl"

TOP_K = 5

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


# ============================================================
# 5. LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedder():

    return SentenceTransformer(EMBED_MODEL)


embedder = load_embedder()


# ============================================================
# 6. LOAD EXISTING KNOWLEDGE BASE
# ============================================================

@st.cache_resource
def load_knowledge_base():

    if not os.path.exists(INDEX_PATH):
        return None, []

    if not os.path.exists(META_PATH):
        return None, []

    index = faiss.read_index(INDEX_PATH)

    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)

    return index, metadata


knowledge_index, knowledge_metadata = load_knowledge_base()


# ============================================================
# 7. PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_pages(pdf_bytes):

    from io import BytesIO

    reader = PdfReader(BytesIO(pdf_bytes))

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):

        try:
            text = page.extract_text() or ""

        except Exception:
            text = ""

        if text.strip():

            pages.append(
                {
                    "page": page_number,
                    "text": text,
                }
            )

    return pages


# ============================================================
# 8. TEXT CHUNKING
# ============================================================

def chunk_text(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):

    chunks = []

    start = 0

    while start < len(text):

        end = min(
            len(text),
            start + chunk_size
        )

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# ============================================================
# 9. PROCESS USER-UPLOADED PDF
# ============================================================

def process_uploaded_pdf(pdf_bytes, filename):

    pages = extract_pdf_pages(pdf_bytes)

    if not pages:

        raise ValueError(
            "Could not extract readable text from this PDF."
        )

    all_chunks = []
    metadata = []

    for page_data in pages:

        page_number = page_data["page"]
        text = page_data["text"]

        chunks = chunk_text(text)

        for chunk_index, chunk in enumerate(chunks):

            all_chunks.append(chunk)

            metadata.append(
                {
                    "doc_id": filename,
                    "page": page_number,
                    "text": chunk,
                    "source": "uploaded",
                }
            )

    if not all_chunks:

        raise ValueError(
            "No readable text was found in the uploaded PDF."
        )

    # Generate embeddings
    embeddings = embedder.encode(
        all_chunks,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    embeddings = embeddings.astype("float32")

    # Normalize embeddings for cosine similarity
    embeddings = embeddings / (
        np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True
        )
        + 1e-9
    )

    dimension = embeddings.shape[1]

    # Create a NEW index only for this uploaded PDF
    uploaded_index = faiss.IndexFlatIP(dimension)

    uploaded_index.add(embeddings)

    return uploaded_index, metadata


# ============================================================
# 10. RETRIEVAL
# ============================================================

def retrieve(
    query,
    index,
    metadata,
    top_k=TOP_K
):

    if index is None or not metadata:

        return []

    # Embed user query
    query_embedding = embedder.encode(
        [query],
        convert_to_numpy=True,
        show_progress_bar=False
    )

    query_embedding = query_embedding.astype(
        "float32"
    )

    # Normalize query vector
    query_embedding = query_embedding / (
        np.linalg.norm(
            query_embedding,
            axis=1,
            keepdims=True
        )
        + 1e-9
    )

    k = min(
        top_k,
        index.ntotal,
        len(metadata)
    )

    if k <= 0:
        return []

    distances, indices = index.search(
        query_embedding,
        k
    )

    results = []

    for distance, index_position in zip(
        distances[0],
        indices[0]
    ):

        if index_position < 0:
            continue

        if index_position >= len(metadata):
            continue

        item = metadata[index_position].copy()

        item["similarity_score"] = float(
            distance
        )

        results.append(item)

    return results


# ============================================================
# 11. GEMINI RESPONSE GENERATION
# ============================================================

def generate_answer(
    query,
    contexts,
    conversation_history
):

    if not contexts:

        return (
            "I couldn't find relevant information "
            "in the available knowledge base."
        )

    context_parts = []

    for i, context in enumerate(contexts, start=1):

        context_parts.append(
            f"""
[Source {i}]
Document: {context.get("doc_id", "Unknown")}
Page: {context.get("page", "Unknown")}

{context.get("text", "")}
"""
        )

    context_text = "\n".join(context_parts)

    history_text = ""

    if conversation_history:

        history_parts = []

        for message in conversation_history[-6:]:

            history_parts.append(
                f'{message["role"].upper()}: '
                f'{message["content"]}'
            )

        history_text = "\n".join(history_parts)

    prompt = f"""
You are Nova AI, a document-grounded AI assistant.

Your task is to answer the user's question using the
provided retrieved context.

IMPORTANT RULES:

1. Use the retrieved context as the primary source of truth.
2. Do not invent facts that are not supported by the context.
3. If the answer cannot be found in the context, clearly say
   that the information is not available in the provided documents.
4. Give a direct and useful answer.
5. When appropriate, explain the answer in a structured way.
6. Do not mention internal implementation details unless the
   user asks about them.

CONVERSATION HISTORY:

{history_text}

RETRIEVED DOCUMENT CONTEXT:

{context_text}

USER QUESTION:

{query}

ANSWER:
"""

    response = model.generate_content(prompt)

    return response.text


# ============================================================
# 12. SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


if "uploaded_index" not in st.session_state:

    st.session_state.uploaded_index = None


if "uploaded_metadata" not in st.session_state:

    st.session_state.uploaded_metadata = []


if "uploaded_file_hash" not in st.session_state:

    st.session_state.uploaded_file_hash = None


if "uploaded_filename" not in st.session_state:

    st.session_state.uploaded_filename = None


# ============================================================
# 13. SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🤖 Nova AI")

    st.caption(
        "Enterprise RAG Knowledge Assistant"
    )

    st.divider()

    # --------------------------------------------------------
    # NEW CHAT
    # --------------------------------------------------------

    if st.button(
        "➕ New Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()

    st.divider()

    # --------------------------------------------------------
    # DOCUMENT UPLOAD
    # --------------------------------------------------------

    st.subheader("📄 Documents")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help=(
            "Upload a PDF to temporarily query "
            "that document."
        )
    )

    if uploaded_file is not None:

        pdf_bytes = uploaded_file.getvalue()

        file_hash = hashlib.md5(
            pdf_bytes
        ).hexdigest()

        # Process only if this is a NEW upload
        if (
            st.session_state.uploaded_file_hash
            != file_hash
        ):

            with st.spinner(
                "Processing your PDF..."
            ):

                try:

                    (
                        uploaded_index,
                        uploaded_metadata
                    ) = process_uploaded_pdf(
                        pdf_bytes,
                        uploaded_file.name
                    )

                    st.session_state.uploaded_index = (
                        uploaded_index
                    )

                    st.session_state.uploaded_metadata = (
                        uploaded_metadata
                    )

                    st.session_state.uploaded_file_hash = (
                        file_hash
                    )

                    st.session_state.uploaded_filename = (
                        uploaded_file.name
                    )

                    # Start a fresh conversation for
                    # the newly uploaded document
                    st.session_state.messages = []

                    st.success(
                        f"✅ {uploaded_file.name} processed"
                    )

                except Exception as e:

                    st.error(
                        f"PDF processing failed: {str(e)}"
                    )

        else:

            st.success(
                f"Using: "
                f"{st.session_state.uploaded_filename}"
            )

    # --------------------------------------------------------
    # KNOWLEDGE BASE STATUS
    # --------------------------------------------------------

    st.divider()

    st.subheader("🧠 Knowledge Source")

    if st.session_state.uploaded_index is not None:

        st.info(
            "📄 Uploaded PDF\n\n"
            f"**{st.session_state.uploaded_filename}**"
        )

        st.caption(
            "Queries are currently restricted to "
            "the uploaded PDF."
        )

        if st.button(
            "↩️ Use Knowledge Base",
            use_container_width=True
        ):

            st.session_state.uploaded_index = None
            st.session_state.uploaded_metadata = []
            st.session_state.uploaded_file_hash = None
            st.session_state.uploaded_filename = None
            st.session_state.messages = []

            st.rerun()

    else:

        if knowledge_index is not None:

            st.success(
                "🧠 Default Knowledge Base active"
            )

            st.caption(
                f"{knowledge_index.ntotal} indexed chunks"
            )

        else:

            st.warning(
                "Knowledge Base not found."
            )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    st.divider()

    st.subheader("⚙️ Configuration")

    st.selectbox(
        "Model",
        [GEMINI_MODEL],
        index=0
    )

    st.caption(
        "Powered by Gemini + FAISS + "
        "Sentence Transformers"
    )


# ============================================================
# 14. MAIN APPLICATION
# ============================================================

st.title("🤖 Nova AI")

st.caption(
    "Ask questions about your documents using "
    "Retrieval-Augmented Generation"
)


# ============================================================
# 15. WELCOME SCREEN
# ============================================================

if len(st.session_state.messages) == 0:

    if st.session_state.uploaded_index is not None:

        st.markdown(
            f"""
### 📄 Document loaded

**{st.session_state.uploaded_filename}**

Ask me anything about this document.
"""
        )

    else:

        st.markdown(
            """
### 👋 Hello!

I'm Nova AI.

You can either:

- 📚 Ask questions about the existing knowledge base
- 📄 Upload a PDF and ask questions specifically about it

Upload a PDF from the sidebar to switch to
document-specific RAG.
"""
        )


# ============================================================
# 16. DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        # Display sources saved with assistant message
        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            with st.expander(
                "📄 Sources"
            ):

                for source in message["sources"]:

                    st.markdown(
                        f"""
**{source.get("doc_id", "Unknown")}**

Page: **{source.get("page", "Unknown")}**

Similarity: **{source.get("similarity_score", 0):.4f}**

{source.get("text", "")[:400]}...
"""
                    )


# ============================================================
# 17. CHAT INPUT
# ============================================================

query = st.chat_input(
    "Message Nova AI..."
)


if query:

    # --------------------------------------------------------
    # USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message("user"):

        st.markdown(query)

    # --------------------------------------------------------
    # SELECT KNOWLEDGE SOURCE
    # --------------------------------------------------------

    if (
        st.session_state.uploaded_index
        is not None
    ):

        active_index = (
            st.session_state.uploaded_index
        )

        active_metadata = (
            st.session_state.uploaded_metadata
        )

        source_mode = (
            f"Uploaded PDF: "
            f"{st.session_state.uploaded_filename}"
        )

    else:

        active_index = knowledge_index
        active_metadata = knowledge_metadata

        source_mode = "Default Knowledge Base"

    # --------------------------------------------------------
    # RETRIEVE
    # --------------------------------------------------------

    contexts = retrieve(
        query,
        active_index,
        active_metadata,
        TOP_K
    )

    # --------------------------------------------------------
    # GENERATE ANSWER
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching documents and generating answer..."
        ):

            try:

                answer = generate_answer(
                    query=query,
                    contexts=contexts,
                    conversation_history=(
                        st.session_state.messages
                    )
                )

            except Exception as e:

                answer = (
                    "⚠️ I encountered an error "
                    f"while generating the answer:\n\n"
                    f"`{str(e)}`"
                )

        st.markdown(answer)

        # ----------------------------------------------------
        # SOURCES
        # ----------------------------------------------------

        if contexts:

            with st.expander(
                f"📄 Sources — {source_mode}"
            ):

                for context in contexts:

                    st.markdown(
                        f"""
**{context.get("doc_id", "Unknown")}**

Page: **{context.get("page", "Unknown")}**

Similarity: **{context.get("similarity_score", 0):.4f}**

{context.get("text", "")[:400]}...
"""
                    )

        else:

            st.warning(
                "No relevant document chunks were retrieved."
            )

    # --------------------------------------------------------
    # SAVE ASSISTANT MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": contexts
        }
    )