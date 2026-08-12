# ============================================================
# NOVA AI
# Retrieval-Augmented Generation Knowledge Assistant
# ============================================================

import os
import pickle
import hashlib
import tempfile
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

from ingest_index import build_index


# ============================================================
# PAGE CONFIG
# IMPORTANT: This must be the FIRST Streamlit command.
# ============================================================

st.set_page_config(
    page_title="Nova AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

def get_secret(name, default=None):
    """
    Read a value from Streamlit Secrets first.
    Fall back to environment variables for local development.
    """

    try:
        value = st.secrets.get(name)

        if value:
            return value

    except Exception:
        pass

    return os.getenv(name, default)


GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
GEMINI_MODEL = get_secret(
    "GEMINI_MODEL",
    "gemini-3.6-flash"
)


if not GEMINI_API_KEY:
    st.error(
        "❌ Gemini API key is not configured.\n\n"
        "For Streamlit Cloud, add GEMINI_API_KEY "
        "under App Settings → Secrets.\n\n"
        "For local development, add GEMINI_API_KEY "
        "to your .env file."
    )
    st.stop()


# ============================================================
# INITIALIZE GEMINI
# ============================================================

try:

    genai.configure(
        api_key=GEMINI_API_KEY
    )

    model = genai.GenerativeModel(
        GEMINI_MODEL
    )

except Exception as e:

    st.error(
        "❌ Gemini initialization failed."
    )

    st.code(
        str(e)
    )

    st.stop()


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

EMBED_MODEL = "all-MiniLM-L6-v2"

DEFAULT_INDEX_PATH = "faiss.index"
DEFAULT_META_PATH = "meta.pkl"

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


if "uploaded_file_hash" not in st.session_state:

    st.session_state.uploaded_file_hash = None


if "uploaded_index" not in st.session_state:

    st.session_state.uploaded_index = None


if "uploaded_metadata" not in st.session_state:

    st.session_state.uploaded_metadata = []


if "default_index" not in st.session_state:

    st.session_state.default_index = None


if "default_metadata" not in st.session_state:

    st.session_state.default_metadata = []


# ============================================================
# EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedder():

    return SentenceTransformer(
        EMBED_MODEL
    )


# ============================================================
# LOAD FAISS INDEX
# ============================================================

def load_index(
    index_path,
    meta_path
):

    if not os.path.exists(index_path):

        return None, []


    if not os.path.exists(meta_path):

        return None, []


    try:

        index = faiss.read_index(
            index_path
        )

        with open(
            meta_path,
            "rb"
        ) as f:

            metadata = pickle.load(f)


        return index, metadata


    except Exception as e:

        st.error(
            "❌ Could not load FAISS knowledge base."
        )

        st.code(
            str(e)
        )

        return None, []


# ============================================================
# LOAD DEFAULT KNOWLEDGE BASE
# ============================================================

def load_default_knowledge_base():

    index, metadata = load_index(
        DEFAULT_INDEX_PATH,
        DEFAULT_META_PATH
    )


    if index is not None:

        return index, metadata


    # If FAISS files are missing, build them
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
            out_index=DEFAULT_INDEX_PATH,
            out_meta=DEFAULT_META_PATH
        )


        return load_index(
            DEFAULT_INDEX_PATH,
            DEFAULT_META_PATH
        )


    except Exception as e:

        st.error(
            "❌ Could not create the default knowledge base."
        )

        st.code(
            str(e)
        )

        return None, []


# ============================================================
# INITIALIZE DEFAULT KNOWLEDGE BASE
# ============================================================

if st.session_state.default_index is None:

    (
        st.session_state.default_index,
        st.session_state.default_metadata
    ) = load_default_knowledge_base()


# ============================================================
# GET ACTIVE KNOWLEDGE BASE
# ============================================================

def get_active_knowledge_base():

    # Uploaded PDF takes priority
    if (
        st.session_state.uploaded_index
        is not None
    ):

        return (
            st.session_state.uploaded_index,
            st.session_state.uploaded_metadata,
            "uploaded"
        )


    # Otherwise use default KB
    return (
        st.session_state.default_index,
        st.session_state.default_metadata,
        "default"
    )


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(
    query,
    k=TOP_K
):

    (
        index,
        metadata,
        source_type
    ) = get_active_knowledge_base()


    if index is None:

        return []


    if index.ntotal == 0:

        return []


    if not metadata:

        return []


    try:

        embedder = load_embedder()


        # ----------------------------------------------------
        # Create query embedding
        # ----------------------------------------------------

        query_embedding = embedder.encode(
            [query],
            convert_to_numpy=True
        )


        # ----------------------------------------------------
        # Normalize embedding
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Search FAISS
        # ----------------------------------------------------

        k = min(
            k,
            index.ntotal
        )


        distances, indices = index.search(
            query_embedding,
            k
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


            chunk["similarity"] = float(
                distance
            )


            results.append(
                chunk
            )


        return results


    except Exception as e:

        st.error(
            "❌ Retrieval failed."
        )

        st.code(
            str(e)
        )

        return []


# ============================================================
# PROCESS UPLOADED PDF
# ============================================================

def process_uploaded_pdf(
    pdf_bytes,
    filename
):

    """
    Creates a temporary FAISS index for the uploaded PDF.

    IMPORTANT:
    This does NOT overwrite:
        faiss.index
        meta.pkl

    Therefore the default knowledge base remains safe.
    """

    # Create a temporary directory
    temp_dir = tempfile.mkdtemp(
        prefix="nova_upload_"
    )


    upload_path = os.path.join(
        temp_dir,
        filename
    )


    uploaded_index_path = os.path.join(
        temp_dir,
        "uploaded.faiss"
    )


    uploaded_meta_path = os.path.join(
        temp_dir,
        "uploaded_meta.pkl"
    )


    # Save uploaded PDF
    with open(
        upload_path,
        "wb"
    ) as f:

        f.write(
            pdf_bytes
        )


    # Build ONLY uploaded document index
    build_index(
        [upload_path],
        out_index=uploaded_index_path,
        out_meta=uploaded_meta_path
    )


    # Load temporary index
    index, metadata = load_index(
        uploaded_index_path,
        uploaded_meta_path
    )


    if index is None:

        raise RuntimeError(
            "Uploaded PDF index could not be created."
        )


    return index, metadata


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title(
        "🤖 Nova AI"
    )

    st.caption(
        "RAG Knowledge Assistant"
    )


    # ========================================================
    # NEW CHAT
    # ========================================================

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
        type=["pdf"],
        help=(
            "Upload a PDF to temporarily "
            "use it as the knowledge source."
        )
    )


    if uploaded_file is not None:

        pdf_bytes = uploaded_file.getvalue()


        # Create hash so same PDF is not processed repeatedly
        file_hash = hashlib.md5(
            pdf_bytes
        ).hexdigest()


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


                    # --------------------------------------------
                    # Store uploaded KB ONLY in session state
                    # --------------------------------------------

                    st.session_state.uploaded_index = (
                        uploaded_index
                    )


                    st.session_state.uploaded_metadata = (
                        uploaded_metadata
                    )


                    st.session_state.uploaded_file_hash = (
                        file_hash
                    )


                    st.session_state.uploaded_pdf = (
                        uploaded_file.name
                    )


                    # Fresh chat for new document
                    st.session_state.messages = []


                    st.success(
                        f"✅ {uploaded_file.name} is ready!"
                    )


                except Exception as e:

                    st.error(
                        "❌ PDF processing failed."
                    )

                    st.code(
                        str(e)
                    )


    # ========================================================
    # CLEAR UPLOADED PDF
    # ========================================================

    if st.session_state.uploaded_pdf:

        if st.button(
            "🔄 Return to Default Knowledge",
            use_container_width=True
        ):

            st.session_state.uploaded_pdf = None

            st.session_state.uploaded_file_hash = None

            st.session_state.uploaded_index = None

            st.session_state.uploaded_metadata = []

            st.session_state.messages = []

            st.rerun()


    st.divider()


    # ========================================================
    # CURRENT KNOWLEDGE SOURCE
    # ========================================================

    st.subheader(
        "📚 Current Knowledge"
    )


    if st.session_state.uploaded_pdf:

        st.success(
            f"📄 {st.session_state.uploaded_pdf}"
        )

        st.caption(
            "Nova AI is currently answering "
            "from your uploaded PDF."
        )

    else:

        st.info(
            "📚 Using default knowledge base."
        )

        for pdf in DEFAULT_PDFS:

            if os.path.exists(pdf):

                st.caption(
                    f"📄 {os.path.basename(pdf)}"
                )


    st.divider()


    # ========================================================
    # MODEL
    # ========================================================

    st.subheader(
        "🧠 AI Model"
    )


    st.caption(
        GEMINI_MODEL
    )


    # ========================================================
    # INDEX STATS
    # ========================================================

    (
        active_index,
        active_metadata,
        active_type
    ) = get_active_knowledge_base()


    if active_index is not None:

        st.caption(
            f"Indexed chunks: {active_index.ntotal}"
        )


# ============================================================
# MAIN HEADER
# ============================================================

st.title(
    "🤖 Nova AI"
)

st.caption(
    "Ask questions using Retrieval-Augmented Generation"
)


# ============================================================
# WELCOME MESSAGE
# ============================================================

if len(
    st.session_state.messages
) == 0:

    if st.session_state.uploaded_pdf:

        st.markdown(
            f"""
### 👋 Document loaded

**📄 {st.session_state.uploaded_pdf}**

Nova AI is currently using this PDF
as its knowledge source.

Ask me anything about this document.
"""
        )

    else:

        st.markdown(
            """
### 👋 Hello!

I'm **Nova AI**.

You can either:

- 📚 Ask questions using the existing knowledge base
- 📄 Upload a PDF and ask questions specifically about that document

When you upload a PDF, Nova AI temporarily switches to that document.

Your default knowledge base is **NOT replaced**.
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
    "Ask Nova AI..."
)


# ============================================================
# PROCESS USER QUESTION
# ============================================================

if user_query:

    # --------------------------------------------------------
    # Save user message
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
    # Retrieve relevant documents
    # --------------------------------------------------------

    with st.spinner(
        "🔎 Searching knowledge base..."
    ):

        contexts = retrieve(
            user_query
        )


    # --------------------------------------------------------
    # NO CONTEXT
    # --------------------------------------------------------

    if not contexts:

        answer = (
            "I couldn't find relevant information "
            "in the current knowledge source."
        )


        with st.chat_message(
            "assistant"
        ):

            st.warning(
                answer
            )


    else:

        # ----------------------------------------------------
        # Build RAG context
        # ----------------------------------------------------

        context_parts = []


        for i, chunk in enumerate(
            contexts,
            start=1
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
        # Determine source
        # ----------------------------------------------------

        if st.session_state.uploaded_pdf:

            source_instruction = (
                "The user uploaded a PDF. "
                "Answer using the retrieved content "
                "from that uploaded PDF."
            )

        else:

            source_instruction = (
                "No PDF was uploaded. "
                "Answer using the retrieved content "
                "from the default knowledge base."
            )


        # ----------------------------------------------------
        # Gemini prompt
        # ----------------------------------------------------

        prompt_for_llm = f"""
You are Nova AI, a reliable Retrieval-Augmented
Generation knowledge assistant.

{source_instruction}

Your task is to answer the user's question using
the retrieved document context below.

IMPORTANT RULES:

1. Ground your answer in the retrieved context.
2. Do not invent facts that contradict the context.
3. If the retrieved context contains the answer,
   explain it clearly and completely.
4. If the context does not contain enough information,
   explicitly say that the retrieved documents do not
   contain enough information.
5. You may use your general knowledge ONLY when the
   question is a basic/general question and the retrieved
   documents clearly do not contain the answer.
6. When using information from the documents, mention
   the relevant source document/page when useful.
7. Do not claim that something is present in a document
   when it is not.
8. Give a useful, natural answer rather than simply
   repeating the retrieved text.
9. Use bullet points or sections when they improve clarity.

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
        # Generate Gemini response
        # ----------------------------------------------------

        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "🧠 Thinking..."
            ):

                try:

                    response = model.generate_content(
                        prompt_for_llm
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
                            "Gemini returned an empty response."
                        )


                except Exception as e:

                    # ====================================================
                    # IMPORTANT DEBUGGING SECTION
                    # ====================================================

                    error_message = str(e)

                    print(
                        f"Gemini generation error: {error_message}"
                    )


                    # --------------------------------------------
                    # Quota / rate limit
                    # --------------------------------------------

                    if (
                        "429" in error_message
                        or "quota" in error_message.lower()
                        or "resource exhausted" in error_message.lower()
                    ):

                        answer = (
                            "⚠️ **Gemini quota/rate limit reached.**\n\n"
                            "Your RAG retrieval is working, but Gemini "
                            "cannot generate the answer right now.\n\n"
                            "Please wait for the quota to reset or use "
                            "another available Gemini API key/project."
                        )


                    # --------------------------------------------
                    # Authentication
                    # --------------------------------------------

                    elif (
                        "401" in error_message
                        or "403" in error_message
                        or "authentication" in error_message.lower()
                        or "api key" in error_message.lower()
                        or "permission" in error_message.lower()
                    ):

                        answer = (
                            "⚠️ **Gemini authentication failed.**\n\n"
                            "Please verify that the correct "
                            "`GEMINI_API_KEY` is configured in "
                            "Streamlit Secrets."
                        )


                    # --------------------------------------------
                    # Model error
                    # --------------------------------------------

                    elif (
                        "model" in error_message.lower()
                        and (
                            "not found" in error_message.lower()
                            or "invalid" in error_message.lower()
                        )
                    ):

                        answer = (
                            "⚠️ **Gemini model configuration error.**\n\n"
                            f"Configured model: `{GEMINI_MODEL}`"
                        )


                    # --------------------------------------------
                    # Other errors
                    # --------------------------------------------

                    else:

                        answer = (
                            "⚠️ **Gemini generation failed.**\n\n"
                            "The retrieved documents were found "
                            "successfully, but Gemini returned an "
                            "unexpected error.\n\n"
                            f"Technical error:\n`{error_message}`"
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

                    doc_name = chunk.get(
                        "doc_id",
                        chunk.get(
                            "document",
                            "Unknown"
                        )
                    )


                    page = chunk.get(
                        "page",
                        "Unknown"
                    )


                    similarity = float(
                        chunk.get(
                            "similarity",
                            chunk.get(
                                "similarity_score",
                                0.0
                            )
                        )
                    )


                    text = chunk.get(
                        "text",
                        chunk.get(
                            "content",
                            ""
                        )
                    )


                    st.markdown(
                        f"""
**{doc_name}**

Page: **{page}**

Similarity: **{similarity:.3f}**

{text[:700]}...
"""
                    )


    # --------------------------------------------------------
    # Save assistant response
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )