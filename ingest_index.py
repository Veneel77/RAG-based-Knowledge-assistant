import os, pickle
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import pdfplumber   # more reliable than pypdf

EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def pdf_to_pages(pdf_path):
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({
                    "doc_id": os.path.basename(pdf_path),
                    "page": i+1,
                    "text": text
                })
    return pages

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + size, L)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
        if start < 0: start = 0
        if start >= L: break
    return chunks

def build_index(pdf_paths, out_index="faiss.index", out_meta="meta.pkl"):
    model = SentenceTransformer(EMBED_MODEL)
    metadatas = []

    for pdf in pdf_paths:
        pages = pdf_to_pages(pdf)
        for p in pages:
            chunks = chunk_text(p["text"])
            for c in chunks:
                metadatas.append({"doc_id": p["doc_id"], "page": p["page"], "text": c})

    texts = [m["text"] for m in metadatas if m["text"].strip()]
    if not texts:
        print("❌ No text extracted from the PDFs. Try another extractor or check if PDF is image-based.")
        return

    print("Encoding", len(texts), "chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # Normalize embeddings
    if embeddings.ndim == 1:   # avoid crash if single embedding
        embeddings = embeddings.reshape(1, -1)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, out_index)
    with open(out_meta, "wb") as f:
        pickle.dump(metadatas, f)

    print(f"✅ Saved index to {out_index}, metadata to {out_meta}")

if __name__ == "__main__":
    import sys
    pdfs = sys.argv[1:]
    if not pdfs:
        print("Usage: python ingest_index.py file1.pdf file2.pdf ...")
    else:
        build_index(pdfs)
