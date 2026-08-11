# ingest_index.py

import os
import pickle
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader


# ============================================================
# CONFIG
# ============================================================

EMBED_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 500

CHUNK_OVERLAP = 50

BATCH_SIZE = 16


# ============================================================
# CHUNK TEXT
# ============================================================

def chunk_text(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):
    """
    Split text into overlapping chunks.
    """

    if not text:
        return []

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
# EXTRACT PDF PAGES
# ============================================================

def pdf_to_pages(pdf_path):
    """
    Extract text page-by-page from a PDF.
    """

    reader = PdfReader(pdf_path)

    pages = []

    for i, page in enumerate(reader.pages):

        try:

            text = page.extract_text() or ""

            pages.append(
                (
                    i + 1,
                    text
                )
            )

        except Exception as e:

            print(
                f"Warning: Could not read page {i + 1}: {e}"
            )

            pages.append(
                (
                    i + 1,
                    ""
                )
            )

    return pages


# ============================================================
# BUILD FAISS INDEX
# ============================================================

def build_index(
    pdf_paths,
    out_index="faiss.index",
    out_meta="meta.pkl",
    embed_model=EMBED_MODEL
):
    """
    Build a FAISS index from the supplied PDFs.

    IMPORTANT:
    Only the PDFs passed through pdf_paths are processed.
    """

    if not pdf_paths:

        raise ValueError(
            "No PDF files were provided."
        )

    print("=" * 60)

    print("Building FAISS knowledge base")

    print("Documents:")

    for pdf in pdf_paths:

        print(
            f"  - {pdf}"
        )

    print("=" * 60)


    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    embedder = SentenceTransformer(
        embed_model
    )

    dimension = (
        embedder
        .get_sentence_embedding_dimension()
    )


    # --------------------------------------------------------
    # FAISS
    # --------------------------------------------------------

    # Inner Product + normalized vectors
    # = cosine similarity

    index = faiss.IndexFlatIP(
        dimension
    )

    metadata = []


    # ========================================================
    # PROCESS EACH PDF
    # ========================================================

    for pdf in pdf_paths:

        print(
            f"\nProcessing {pdf} ..."
        )

        pages = pdf_to_pages(
            pdf
        )


        all_chunks = []

        chunk_metadata = []


        # ----------------------------------------------------
        # Page → chunks
        # ----------------------------------------------------

        for page_number, text in pages:

            chunks = chunk_text(
                text
            )

            for chunk_index, chunk in enumerate(
                chunks
            ):

                all_chunks.append(
                    chunk
                )

                chunk_metadata.append(
                    {
                        "doc_id": os.path.basename(
                            pdf
                        ),

                        "page": page_number,

                        "chunk_index": chunk_index,

                        "text": chunk
                    }
                )


        print(
            f"Created {len(all_chunks)} chunks"
        )


        # ----------------------------------------------------
        # Generate embeddings in batches
        # ----------------------------------------------------

        for i in tqdm(
            range(
                0,
                len(all_chunks),
                BATCH_SIZE
            ),
            desc=f"Embedding {os.path.basename(pdf)}"
        ):

            batch = all_chunks[
                i:i + BATCH_SIZE
            ]


            if not batch:

                continue


            embeddings = embedder.encode(
                batch,
                batch_size=BATCH_SIZE,
                convert_to_numpy=True,
                show_progress_bar=False
            )


            # ------------------------------------------------
            # Normalize vectors
            # ------------------------------------------------

            embeddings = embeddings / (
                np.linalg.norm(
                    embeddings,
                    axis=1,
                    keepdims=True
                ) + 1e-9
            )


            embeddings = embeddings.astype(
                "float32"
            )


            # ------------------------------------------------
            # Add vectors
            # ------------------------------------------------

            index.add(
                embeddings
            )


            metadata.extend(
                chunk_metadata[
                    i:i + BATCH_SIZE
                ]
            )


    # ========================================================
    # SAVE INDEX
    # ========================================================

    faiss.write_index(
        index,
        out_index
    )


    # ========================================================
    # SAVE METADATA
    # ========================================================

    with open(
        out_meta,
        "wb"
    ) as f:

        pickle.dump(
            metadata,
            f
        )


    print("\n" + "=" * 60)

    print("FAISS index successfully created.")

    print(
        f"Total vectors: {index.ntotal}"
    )

    print(
        f"Total metadata entries: {len(metadata)}"
    )

    print(
        f"Index: {out_index}"
    )

    print(
        f"Metadata: {out_meta}"
    )

    print("=" * 60)


# ============================================================
# COMMAND LINE USAGE
# ============================================================

if __name__ == "__main__":

    import sys


    if len(sys.argv) < 2:

        print(
            "Usage:"
        )

        print(
            "python ingest_index.py file1.pdf file2.pdf ..."
        )

        sys.exit(1)


    pdfs = sys.argv[1:]


    build_index(
        pdfs
    )