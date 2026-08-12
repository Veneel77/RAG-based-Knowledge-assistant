import os
import pickle
import hashlib
from io import BytesIO

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
    Read configuration from Streamlit Secrets first.
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
        "⚠️ Gemini API key is not configured.\n\n"
        "Please add GEMINI_API_KEY to Streamlit Secrets."
    )

    st.stop()


try:

    genai.configure(
        api_key=GEMINI_API_KEY
    )

    model = genai.GenerativeModel(
        GEMINI_MODEL
    )

except Exception as e:

    st.error(
        "⚠️ Gemini initialization failed.\n\n"
        "Please verify your Gemini API key and model configuration."
    )

    print(f"Gemini initialization error: {e}")

    st.stop()


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

    return SentenceTransformer(
        EMBED_MODEL
    )


embedder = load_embedder()


# ============================================================
# 6. LOAD DEFAULT KNOWLEDGE BASE
# ============================================================

@st.cache_resource
def load_knowledge_base():

    if not os.path.exists(INDEX_PATH):

        return None, []

    if not os.path.exists(META_PATH):

        return None, []

    try:

        index = faiss.read_index(
            INDEX_PATH
        )

        with open(
            META_PATH,
            "rb"
        ) as f:

            metadata = pickle.load(f)

        return index, metadata

    except Exception as e:

        print(
            f"Knowledge base loading error: {e}"
        )

        return None, []


knowledge_index, knowledge_metadata = (
    load_knowledge_base()
)


# ============================================================
# 7. PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_pages(pdf_bytes):

    reader = PdfReader(
        BytesIO(pdf_bytes)
    )

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

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

    step = chunk_size - overlap

    while start < len(text):

        end = min(
            len(text),
            start + chunk_size
        )

        chunk = text[start:end].strip()

        if chunk:

            chunks.append(
                chunk
            )

        start += step

    return chunks


# ============================================================
# 9. PROCESS USER-UPLOADED PDF
# ============================================================

def process_uploaded_pdf(
    pdf_bytes,
    filename
):

    pages = extract_pdf_pages(
        pdf_bytes
    )

    if not pages:

        raise ValueError(
            "Could not extract readable text from this PDF."
        )

    all_chunks = []

    metadata = []

    for page_data in pages:

        page_number = page_data["page"]

        text = page_data["text"]

        chunks = chunk_text(
            text
        )

        for chunk in chunks:

            all_chunks.append(
                chunk
            )

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

    # --------------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------------

    embeddings = embedder.encode(
        all_chunks,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    embeddings = embeddings.astype(
        "float32"
    )

    # --------------------------------------------------------
    # Normalize for cosine similarity
    # --------------------------------------------------------

    embeddings = embeddings / (
        np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True
        )
        + 1e-9
    )

    dimension = embeddings.shape[1]

    # --------------------------------------------------------
    # Create temporary uploaded-document index
    # --------------------------------------------------------

    uploaded_index = faiss.IndexFlatIP(
        dimension
    )

    uploaded_index.add(
        embeddings
    )

    return (
        uploaded_index,
        metadata
    )


# ============================================================
# 10. RETRIEVAL
# ============================================================

def retrieve(
    query,
    index,
    metadata,
    top_k=TOP_K
):

    if index is None:

        return []

    if not metadata:

        return []

    if index.ntotal == 0:

        return []

    # --------------------------------------------------------
    # Query embedding
    # --------------------------------------------------------

    query_embedding = embedder.encode(
        [query],
        convert_to_numpy=True,
        show_progress_bar=False
    )

    query_embedding = query_embedding.astype(
        "float32"
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    query_embedding = query_embedding / (
        np.linalg.norm(
            query_embedding,
            axis=1,
            keepdims=True
        )
        + 1e-9
    )

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

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

        item = metadata[
            index_position
        ].copy()

        item[
            "similarity_score"
        ] = float(distance)

        results.append(
            item
        )

    return results


# ============================================================
# 11. GEMINI RESPONSE GENERATION
# ============================================================

def generate_answer(
    query,
    contexts,
    conversation_history
):

    # --------------------------------------------------------
    # No retrieved information
    # --------------------------------------------------------

    if not contexts:

        return (
            "I couldn't find relevant information "
            "in the current knowledge source."
        )


    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    context_parts = []

    for i, context in enumerate(
        contexts,
        start=1
    ):

        context_parts.append(
            f"""
[Source {i}]

Document:
{context.get("doc_id", "Unknown")}

Page:
{context.get("page", "Unknown")}

Content:
{context.get("text", "")}
"""
        )

    context_text = "\n".join(
        context_parts
    )


    # --------------------------------------------------------
    # Conversation history
    # --------------------------------------------------------

    history_text = ""

    if conversation_history:

        history_parts = []

        for message in conversation_history[-6:]:

            history_parts.append(
                f'{message["role"].upper()}: '
                f'{message["content"]}'
            )

        history_text = "\n".join(
            history_parts
        )


    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = f"""
You are Nova AI, a reliable Retrieval-Augmented Generation
assistant.

Your job is to answer the user's question using the
retrieved document context provided below.

IMPORTANT RULES:

1. Use the retrieved context as the primary source of truth.

2. Do NOT invent facts that are not supported by the
   retrieved context.

3. If the retrieved context does not contain enough
   information to answer the question, clearly say that
   the information was not found in the current knowledge
   source.

4. Give a clear, direct and useful answer.

5. Explain concepts in simple language when appropriate.

6. Use bullet points or sections when they improve clarity.

7. Do not mention internal implementation details unless
   the user asks about them.

8. If the user asks a question about an uploaded document,
   stay focused on that uploaded document.

9. Do not mix information from another document or
   knowledge source.

10. Never pretend that information exists in the document
    when it does not.

==============================
CONVERSATION HISTORY
==============================

{history_text}

==============================
RETRIEVED DOCUMENT CONTEXT
==============================

{context_text}

==============================
USER QUESTION
==============================

{query}

==============================
ANSWER
==============================
"""

    # --------------------------------------------------------
    # Gemini generation
    # --------------------------------------------------------

    try:

        response = model.generate_content(
            prompt
        )

        if response and response.text:

            return response.text.strip()

        return (
            "I couldn't generate an answer from "
            "the retrieved information."
        )

    except Exception as e:

        error_message = str(e).lower()

        # ----------------------------------------------------
        # QUOTA / RATE LIMIT
        # ----------------------------------------------------

        if (
            "429" in error_message
            or "quota" in error_message
            or "rate limit" in error_message
            or "resource exhausted" in error_message
        ):

            return (
                "⚠️ **Gemini free-tier limit reached.**\n\n"
                "Nova AI successfully searched the document, "
                "but Gemini is temporarily unavailable because "
                "the current free-tier generation quota has "
                "been reached.\n\n"
                "Please try again after the quota resets."
            )

        # ----------------------------------------------------
        # API KEY / AUTHENTICATION
        # ----------------------------------------------------

        if (
            "api key" in error_message
            or "authentication" in error_message
            or "permission" in error_message
            or "403" in error_message
        ):

            return (
                "⚠️ **Gemini authentication failed.**\n\n"
                "Please verify the Gemini API key configured "
                "in Streamlit Secrets."
            )

        # ----------------------------------------------------
        # GENERIC ERROR
        # ----------------------------------------------------

        print(
            f"Gemini generation error: {e}"
        )

        return (
            "⚠️ I encountered a temporary error while "
            "generating the answer. Please try again."
        )


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

    st.title(
        "🤖 Nova AI"
    )

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
    # PDF UPLOAD
    # --------------------------------------------------------

    st.subheader(
        "📄 Documents"
    )

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


        # ----------------------------------------------------
        # Process only NEW PDF
        # ----------------------------------------------------

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


                    # ----------------------------------------
                    # Store ONLY in session state
                    # ----------------------------------------

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


                    # ----------------------------------------
                    # Fresh conversation
                    # ----------------------------------------

                    st.session_state.messages = []


                    st.success(
                        f"✅ {uploaded_file.name} processed"
                    )


                except Exception as e:

                    st.error(
                        "PDF processing failed."
                    )

                    print(
                        f"PDF processing error: {e}"
                    )

        else:

            st.success(
                f"Using: "
                f"{st.session_state.uploaded_filename}"
            )


    # --------------------------------------------------------
    # KNOWLEDGE SOURCE
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "🧠 Knowledge Source"
    )


    if (
        st.session_state.uploaded_index
        is not None
    ):

        st.info(
            "📄 Uploaded PDF\n\n"
            f"**{st.session_state.uploaded_filename}**"
        )

        st.caption(
            "Nova AI is currently answering "
            "ONLY from this uploaded PDF."
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

        if (
            knowledge_index is not None
            and knowledge_index.ntotal > 0
        ):

            st.success(
                "🧠 Default Knowledge Base active"
            )

            st.caption(
                f"{knowledge_index.ntotal} indexed chunks"
            )

        else:

            st.warning(
                "⚠️ Default Knowledge Base not found."
            )


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "⚙️ Configuration"
    )

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

st.title(
    "🤖 Nova AI"
)

st.caption(
    "Ask questions about your documents using "
    "Retrieval-Augmented Generation"
)


# ============================================================
# 15. WELCOME SCREEN
# ============================================================

if len(
    st.session_state.messages
) == 0:

    if (
        st.session_state.uploaded_index
        is not None
    ):

        st.markdown(
            f"""
### 📄 Document loaded

**{st.session_state.uploaded_filename}**

Nova AI is currently using this PDF as its
knowledge source.

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
- 📄 Upload a PDF and ask questions specifically
  about that document

When you upload a PDF, Nova AI temporarily switches
to that document.

Your default knowledge base is NOT replaced.
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


        # ----------------------------------------------------
        # Display sources
        # ----------------------------------------------------

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

Similarity: **{float(source.get("similarity_score", 0)):.4f}**

{source.get("text", "")[:400]}...
"""
                    )


# ============================================================
# 17. CHAT INPUT
# ============================================================

query = st.chat_input(
    "Message Nova AI..."
)


# ============================================================
# 18. PROCESS QUERY
# ============================================================

if query:

    # --------------------------------------------------------
    # Save user message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )


    with st.chat_message(
        "user"
    ):

        st.markdown(
            query
        )


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
            "Uploaded PDF: "
            f"{st.session_state.uploaded_filename}"
        )

    else:

        active_index = knowledge_index

        active_metadata = knowledge_metadata

        source_mode = (
            "Default Knowledge Base"
        )


    # --------------------------------------------------------
    # RETRIEVE
    # --------------------------------------------------------

    with st.spinner(
        "Searching knowledge..."
    ):

        contexts = retrieve(
            query,
            active_index,
            active_metadata,
            TOP_K
        )


    # --------------------------------------------------------
    # GENERATE ANSWER
    # --------------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Generating answer..."
        ):

            answer = generate_answer(
                query=query,
                contexts=contexts,
                conversation_history=(
                    st.session_state.messages
                )
            )


        st.markdown(
            answer
        )


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

Similarity: **{float(context.get("similarity_score", 0)):.4f}**

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