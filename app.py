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
# ENVIRONMENT
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

if not GEMINI_API_KEY:

    st.error(
        "GEMINI_API_KEY is missing from .env"
    )

    st.stop()


genai.configure(
    api_key=GEMINI_API_KEY
)

model = genai.GenerativeModel(
    "gemini-2.5-flash"
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Nova AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CONFIG
# ============================================================

EMBED_MODEL = "all-MiniLM-L6-v2"

INDEX_PATH = "faiss.index"

META_PATH = "meta.pkl"

DEFAULT_PDFS = [
    "docs/AI1.pdf",
    "docs/aiimpact.pdf"
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
# ============================================================

@st.cache_resource
def load_embedder():

    return SentenceTransformer(
        EMBED_MODEL
    )


embedder = load_embedder()


# ============================================================
# LOAD INDEX
# ============================================================

def load_index(
    index_path=INDEX_PATH,
    meta_path=META_PATH
):

    if not os.path.exists(
        index_path
    ):

        return None, []


    if not os.path.exists(
        meta_path
    ):

        return None, []


    index = faiss.read_index(
        index_path
    )


    with open(
        meta_path,
        "rb"
    ) as f:

        metadata = pickle.load(
            f
        )


    return index, metadata


# ============================================================
# CREATE DEFAULT KNOWLEDGE BASE
# ============================================================

def create_default_index():

    valid_pdfs = []

    for pdf in DEFAULT_PDFS:

        if os.path.exists(pdf):

            valid_pdfs.append(pdf)


    if not valid_pdfs:

        return None, []


    build_index(
        valid_pdfs,
        out_index=INDEX_PATH,
        out_meta=META_PATH
    )


    return load_index()


# ============================================================
# INITIALIZE KNOWLEDGE BASE
# ============================================================

if (
    st.session_state.active_index
    is None
):

    index, metadata = load_index()


    if index is None:

        index, metadata = (
            create_default_index()
        )


    st.session_state.active_index = (
        index
    )

    st.session_state.active_meta = (
        metadata
    )


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(
    query,
    k=TOP_K
):

    index = (
        st.session_state.active_index
    )

    metadata = (
        st.session_state.active_meta
    )


    if index is None:

        return []


    if index.ntotal == 0:

        return []


    # --------------------------------------------------------
    # Query embedding
    # --------------------------------------------------------

    query_embedding = embedder.encode(
        [query],
        convert_to_numpy=True
    )


    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    query_embedding = (
        query_embedding
        /
        (
            np.linalg.norm(
                query_embedding,
                axis=1,
                keepdims=True
            )
            + 1e-9
        )
    )


    query_embedding = (
        query_embedding.astype(
            "float32"
        )
    )


    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    k = min(
        k,
        index.ntotal
    )


    distances, indices = (
        index.search(
            query_embedding,
            k
        )
    )


    results = []


    for idx, distance in zip(
        indices[0],
        distances[0]
    ):

        if idx < 0:

            continue


        if idx >= len(metadata):

            continue


        chunk = metadata[idx].copy()


        chunk["similarity"] = (
            float(distance)
        )


        results.append(
            chunk
        )


    return results


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title(
        "🤖 Nova AI"
    )


    # --------------------------------------------------------
    # New Chat
    # --------------------------------------------------------

    if st.button(
        "➕ New Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


    st.divider()


    # ========================================================
    # PDF UPLOAD
    # ========================================================

    st.subheader(
        "📄 Upload Document"
    )


    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"]
    )


    if uploaded_file is not None:

        # ----------------------------------------------------
        # Prevent repeated processing
        # ----------------------------------------------------

        if (
            st.session_state.uploaded_pdf
            != uploaded_file.name
        ):

            with st.spinner(
                "Processing your PDF..."
            ):

                # --------------------------------------------
                # Save uploaded PDF
                # --------------------------------------------

                os.makedirs(
                    "uploads",
                    exist_ok=True
                )


                upload_path = os.path.join(
                    "uploads",
                    uploaded_file.name
                )


                with open(
                    upload_path,
                    "wb"
                ) as f:

                    f.write(
                        uploaded_file.getbuffer()
                    )


                # --------------------------------------------
                # IMPORTANT
                # ONLY USER PDF IS INDEXED
                # --------------------------------------------

                build_index(
                    [upload_path],

                    out_index=INDEX_PATH,

                    out_meta=META_PATH
                )


                # --------------------------------------------
                # Load new index
                # --------------------------------------------

                new_index, new_metadata = (
                    load_index()
                )


                st.session_state.active_index = (
                    new_index
                )

                st.session_state.active_meta = (
                    new_metadata
                )


                st.session_state.uploaded_pdf = (
                    uploaded_file.name
                )


                # --------------------------------------------
                # Clear previous conversation
                # --------------------------------------------

                st.session_state.messages = []


            st.success(
                f"{uploaded_file.name} is ready!"
            )


    # ========================================================
    # ACTIVE DOCUMENT
    # ========================================================

    st.divider()

    st.subheader(
        "📚 Current Knowledge"
    )


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


    # ========================================================
    # MODEL
    # ========================================================

    st.divider()

    st.selectbox(
        "Model",
        [
            "Gemini 2.5 Flash"
        ]
    )


    # ========================================================
    # STATS
    # ========================================================

    if (
        st.session_state.active_index
        is not None
    ):

        st.caption(
            "Indexed chunks: "
            +
            str(
                st.session_state
                .active_index
                .ntotal
            )
        )


# ============================================================
# MAIN HEADER
# ============================================================

st.title(
    "🤖 Nova AI"
)

st.caption(
    "Chat with your documents using Retrieval-Augmented Generation"
)


# ============================================================
# WELCOME
# ============================================================

if len(
    st.session_state.messages
) == 0:

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

for message in (
    st.session_state.messages
):

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
# PROCESS USER QUERY
# ============================================================

if user_query:

    # --------------------------------------------------------
    # USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_query
        }
    )


    with st.chat_message(
        "user"
    ):

        st.markdown(
            user_query
        )


    # --------------------------------------------------------
    # RETRIEVE
    # --------------------------------------------------------

    contexts = retrieve(
        user_query
    )


    # --------------------------------------------------------
    # NO CONTEXT
    # --------------------------------------------------------

    if not contexts:

        answer = (
            "I couldn't find relevant information "
            "in the current document."
        )

        with st.chat_message(
            "assistant"
        ):

            st.markdown(
                answer
            )


    else:

        # ----------------------------------------------------
        # BUILD CONTEXT
        # ----------------------------------------------------

        context_text = ""


        for i, chunk in enumerate(
            contexts,
            start=1
        ):

            context_text += f"""

[Source {i}]

Document:
{chunk.get("doc_id", "Unknown")}

Page:
{chunk.get("page", "Unknown")}

Content:
{chunk.get("text", "")}

"""


        # ----------------------------------------------------
        # GEMINI PROMPT
        # ----------------------------------------------------

        prompt_for_llm = f"""

You are Nova AI, a document question-answering assistant.

Your job is to answer the user's question using ONLY
the information contained in the retrieved document context.

RULES:

1. Ground your answer in the provided context.
2. Do not invent information.
3. If the answer is not present in the context, say:
   "I couldn't find this information in the uploaded document."
4. Give a clear and useful answer.
5. When possible, mention the source document and page.
6. Do not claim information that is not supported by the context.

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
        # GEMINI
        # ----------------------------------------------------

        with st.chat_message(
            "assistant"
        ):

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


            st.markdown(
                answer
            )


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
    # SAVE ASSISTANT MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )