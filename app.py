import os
import pickle
import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from ingest_index import build_index
from dotenv import load_dotenv
import google.generativeai as genai

# ============================================================
# PAGE CONFIG
# This MUST be the first Streamlit command in the application.
# ============================================================

st.set_page_config(
    page_title="Nova AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# ENVIRONMENT / SECRETS
# Local:
#   GEMINI_API_KEY in .env
#
# Streamlit Cloud:
#   GEMINI_API_KEY in Settings -> Secrets
# ============================================================

load_dotenv()

try:
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
except Exception:
    GEMINI_API_KEY = None

if not GEMINI_API_KEY:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error(
        "GEMINI_API_KEY is not configured. "
        "Add it to Streamlit Secrets or your local .env file."
    )
    st.stop()

try:
    GEMINI_MODEL = st.secrets.get(
        "GEMINI_MODEL",
        os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    )
except Exception:
    GEMINI_MODEL = os.getenv(
        "GEMINI_MODEL",
        "gemini-2.5-flash",
    )

try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)
except Exception as e:
    st.error(f"Gemini initialization failed: {e}")
    st.stop()

# ============================================================
# CONFIGURATION
# ============================================================

EMBED_MODEL = "all-MiniLM-L6-v2"

INDEX_PATH = "faiss.index"
META_PATH = "meta.pkl"

# These are only used as the initial/demo knowledge base.
# Once a user uploads a PDF, ONLY that uploaded PDF is indexed.
DEFAULT_PDFS = [
    "docs/AI1.pdf",
    "docs/aiimpact.pdf",
]

TOP_K = 5

# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_pdf" not in st.session_state:
    st.session_state.uploaded_pdf = None

if "active_index" not in st.session_state:
    st.session_state.active_index = None

if "active_meta" not in st.session_state:
    st.session_state.active_meta = None

# ============================================================
# EMBEDDING MODEL
# Lazy loaded when the first query is made.
# This keeps application startup faster.
# ============================================================

@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL)

# ============================================================
# FAISS INDEX LOADING
# ============================================================

def load_index(
    index_path=INDEX_PATH,
    meta_path=META_PATH,
):
    """Load FAISS index and its metadata."""

    if not os.path.exists(index_path):
        return None, []

    if not os.path.exists(meta_path):
        return None, []

    try:
        index = faiss.read_index(index_path)

        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)

        return index, metadata

    except Exception as e:
        st.error(f"Could not load FAISS index: {e}")
        return None, []

# ============================================================
# DEFAULT KNOWLEDGE BASE
# ============================================================

def create_default_index():
    """
    Build the initial demo knowledge base only if no FAISS
    index exists yet.
    """

    valid_pdfs = [
        pdf
        for pdf in DEFAULT_PDFS
        if os.path.exists(pdf)
    ]

    if not valid_pdfs:
        return None, []

    try:
        build_index(
            valid_pdfs,
            out_index=INDEX_PATH,
            out_meta=META_PATH,
        )

        return load_index()

    except Exception as e:
        st.error(
            f"Could not create default knowledge base: {e}"
        )
        return None, []

# ============================================================
# INITIALIZE ACTIVE KNOWLEDGE BASE
# ============================================================

if st.session_state.active_index is None:

    index, metadata = load_index()

    if index is None:
        index, metadata = create_default_index()

    st.session_state.active_index = index
    st.session_state.active_meta = metadata

# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(query, k=TOP_K):
    """
    Convert the user's question into an embedding and retrieve
    the most similar chunks from FAISS.
    """

    index = st.session_state.active_index
    metadata = st.session_state.active_meta

    if index is None:
        return []

    if index.ntotal == 0:
        return []

    # Load embedding model only when needed.
    embedder = load_embedder()

    # Create query embedding.
    query_embedding = embedder.encode(
        [query],
        convert_to_numpy=True,
    )

    # Normalize because the ingestion pipeline uses normalized
    # embeddings and FAISS IndexFlatIP.
    query_embedding = (
        query_embedding
        /
        (
            np.linalg.norm(
                query_embedding,
                axis=1,
                keepdims=True,
            )
            + 1e-9
        )
    )

    query_embedding = query_embedding.astype("float32")

    k = min(k, index.ntotal)

    distances, indices = index.search(
        query_embedding,
        k,
    )

    results = []

    for idx, distance in zip(
        indices[0],
        distances[0],
    ):

        if idx < 0:
            continue

        if idx >= len(metadata):
            continue

        chunk = metadata[idx].copy()

        chunk["similarity"] = float(distance)

        results.append(chunk)

    return results

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🤖 Nova AI")

    # --------------------------------------------------------
    # NEW CHAT
    # --------------------------------------------------------

    if st.button(
        "➕ New Chat",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # --------------------------------------------------------
    # PDF UPLOAD
    # --------------------------------------------------------

    st.subheader("📄 Upload Document")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
    )

    if (
        uploaded_file is not None
        and
        st.session_state.uploaded_pdf
        != uploaded_file.name
    ):

        with st.spinner(
            "Processing your PDF..."
        ):

            os.makedirs(
                "uploads",
                exist_ok=True,
            )

            upload_path = os.path.join(
                "uploads",
                uploaded_file.name,
            )

            with open(
                upload_path,
                "wb",
            ) as f:
                f.write(
                    uploaded_file.getbuffer()
                )

            # IMPORTANT:
            # The user's uploaded PDF is the ONLY document
            # passed to build_index here.
            #
            # Therefore the old demo PDFs are not included
            # in this new user-specific index.
            build_index(
                [upload_path],
                out_index=INDEX_PATH,
                out_meta=META_PATH,
            )

            # Immediately load the newly-created index.
            new_index, new_metadata = load_index()

            st.session_state.active_index = new_index
            st.session_state.active_meta = new_metadata

            # Remember which document is currently active.
            st.session_state.uploaded_pdf = (
                uploaded_file.name
            )

            # Start a fresh conversation for the new PDF.
            st.session_state.messages = []

        st.success(
            f"{uploaded_file.name} is ready!"
        )

    # --------------------------------------------------------
    # CURRENT KNOWLEDGE BASE
    # --------------------------------------------------------

    st.divider()

    st.subheader("📚 Current Knowledge")

    if st.session_state.uploaded_pdf:

        st.success(
            f"📄 {st.session_state.uploaded_pdf}"
        )

        st.caption(
            "Questions are answered using this uploaded document."
        )

    else:

        st.info(
            "Using default knowledge base."
        )

        for pdf in DEFAULT_PDFS:

            if os.path.exists(pdf):

                st.caption(
                    f"📄 {os.path.basename(pdf)}"
                )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    st.divider()

    st.selectbox(
        "Model",
        ["Gemini 2.5 Flash"],
        index=0,
    )

    # --------------------------------------------------------
    # INDEX STATISTICS
    # --------------------------------------------------------

    if st.session_state.active_index is not None:

        st.caption(
            "Indexed chunks: "
            +
            str(
                st.session_state.active_index.ntotal
            )
        )

# ============================================================
# MAIN WINDOW
# ============================================================

st.title("🤖 Nova AI")

st.caption(
    "Chat with your documents using Retrieval-Augmented Generation"
)

# ============================================================
# WELCOME MESSAGE
# ============================================================

if len(st.session_state.messages) == 0:

    if st.session_state.uploaded_pdf:

        st.markdown(
            f"""
## 👋 You're ready!

I've processed:

**📄 {st.session_state.uploaded_pdf}**

Ask me anything about this document.
"""
        )

    else:

        st.markdown(
            """
## 👋 Welcome to Nova AI

Upload a PDF from the sidebar and ask questions about it.

You can also try the default knowledge base.
"""
        )

# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

# ============================================================
# CHAT INPUT
# ============================================================

user_query = st.chat_input(
    "Message Nova AI..."
)

# ============================================================
# QUERY PROCESSING
# ============================================================

if user_query:

    # --------------------------------------------------------
    # Save and display user message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_query,
        }
    )

    with st.chat_message("user"):

        st.markdown(user_query)

    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    with st.spinner(
        "Searching your document..."
    ):

        contexts = retrieve(
            user_query
        )

    # --------------------------------------------------------
    # NO RELEVANT CONTEXT
    # --------------------------------------------------------

    if not contexts:

        answer = (
            "I couldn't find relevant information "
            "in the current document."
        )

        with st.chat_message("assistant"):

            st.markdown(answer)

    # --------------------------------------------------------
    # RAG + GEMINI
    # --------------------------------------------------------

    else:

        context_parts = []

        for i, chunk in enumerate(
            contexts,
            start=1,
        ):

            context_parts.append(
                f"""
[Source {i}]

Document:
{chunk.get("doc_id", "Unknown")}

Page:
{chunk.get("page", "Unknown")}

Content:
{chunk.get("text", "")}
"""
            )

        context_text = "\n".join(
            context_parts
        )

        # ----------------------------------------------------
        # RAG PROMPT
        # ----------------------------------------------------

        prompt_for_llm = f"""
You are Nova AI, a document question-answering assistant.

Answer the user's question using ONLY the information
contained in the retrieved document context below.

RULES:

1. Ground your answer in the provided context.
2. Do not invent or hallucinate information.
3. If the answer is not present in the context, say:
   "I couldn't find this information in the uploaded document."
4. Give a clear and useful answer.
5. When possible, mention the source document and page.
6. Do not claim information unsupported by the context.

==============================
RETRIEVED DOCUMENT CONTEXT
==============================

{context_text}

==============================
USER QUESTION
==============================

{user_query}

==============================
ANSWER
==============================
"""

        # ----------------------------------------------------
        # GENERATE WITH GEMINI
        # ----------------------------------------------------

        with st.chat_message("assistant"):

            with st.spinner(
                "Thinking..."
            ):

                try:

                    response = (
                        model.generate_content(
                            prompt_for_llm
                        )
                    )

                    if (
                        response
                        and response.text
                    ):

                        answer = (
                            response.text.strip()
                        )

                    else:

                        answer = (
                            "I couldn't generate an answer."
                        )

                except Exception as e:

                    answer = (
                        "Gemini encountered an error.\n\n"
                        f"{str(e)}"
                    )

            st.markdown(answer)

            # ------------------------------------------------
            # SOURCES
            # ------------------------------------------------

            with st.expander(
                "📚 Sources used"
            ):

                for chunk in contexts:

                    st.markdown(
                        f"""
**{chunk.get("doc_id", "Unknown")}**

Page **{chunk.get("page", "Unknown")}**

Similarity: `{chunk.get("similarity", 0):.3f}`

{chunk.get("text", "")[:500]}...
"""
                    )

    # --------------------------------------------------------
    # SAVE ASSISTANT RESPONSE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )