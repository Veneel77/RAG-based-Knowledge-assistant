# ingest_index.py
import os
import pickle
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

# ----------------------------
# Config
# ----------------------------
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500        # max characters per chunk
CHUNK_OVERLAP = 50      # overlap for context
BATCH_SIZE = 16         # how many chunks per embedding batch


# ----------------------------
# Split text into chunks
# ----------------------------
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# ----------------------------
# Extract text from PDF safely
# ----------------------------
def pdf_to_pages(pdf_path):
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
            pages.append((i + 1, text))
        except Exception:
            pages.append((i + 1, ""))
    return pages


# ----------------------------
# Build FAISS index
# ----------------------------
def build_index(
    pdf_paths,
    out_index="faiss.index",
    out_meta="meta.pkl",
    embed_model=EMBED_MODEL,
):
    embedder = SentenceTransformer(embed_model)
    dim = embedder.get_sentence_embedding_dimension()
    index = faiss.IndexFlatIP(dim)
    metadata = []

    for pdf in pdf_paths:
        print(f"Processing {pdf} ...")
        pages = pdf_to_pages(pdf)

        all_chunks = []
        chunk_meta = []
        for page_num, text in pages:
            for chunk in chunk_text(text):
                all_chunks.append(chunk)
                chunk_meta.append({"doc_id": os.path.basename(pdf),
                                   "page": page_num,
                                   "text": chunk})

        # Encode in mini-batches to avoid RAM spikes
        for i in tqdm(range(0, len(all_chunks), BATCH_SIZE), desc=f"Embedding {pdf}"):
            batch = all_chunks[i:i + BATCH_SIZE]
            X = embedder.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            # Normalize for cosine similarity
            X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
            index.add(X)
            metadata.extend(chunk_meta[i:i + BATCH_SIZE])

    faiss.write_index(index, out_index)
    with open(out_meta, "wb") as f:
        pickle.dump(metadata, f)
    print(f"Index built: {out_index}, metadata saved: {out_meta}")


# ----------------------------
# CLI usage
# ----------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ingest_index.py file1.pdf file2.pdf ...")
        sys.exit(1)

    pdfs = sys.argv[1:]
    build_index(pdfs)
