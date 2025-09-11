# ingest_index.py
import os, pickle
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from tqdm import tqdm

# Config
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 100
BATCH_SIZE = 16

def pdf_to_pages(pdf_path):
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"doc_id": os.path.basename(pdf_path), "page": i+1, "text": text})
    return pages

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks, start, L = [], 0, len(text)
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
    dim = model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatIP(dim)
    metadatas = []

    for pdf in pdf_paths:
        print(f"\n📄 Processing {pdf}...")
        pages = pdf_to_pages(pdf)
        for p in tqdm(pages, desc=f"Pages in {pdf}"):
            chunks = chunk_text(p["text"])
            for i in range(0, len(chunks), BATCH_SIZE):
                batch = chunks[i:i+BATCH_SIZE]
                embeddings = model.encode(batch, convert_to_numpy=True)
                # normalize for cosine similarity
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / np.where(norms==0, 1, norms)
                index.add(embeddings)
                for c in batch:
                    metadatas.append({"doc_id": p["doc_id"], "page": p["page"], "text": c})

    # Save index + metadata
    faiss.write_index(index, out_index)
    with open(out_meta, "wb") as f:
        pickle.dump(metadatas, f)
    print(f"\n✅ Index built and saved to {out_index}, metadata to {out_meta}")

if __name__ == "__main__":
    import sys
    pdfs = sys.argv[1:]
    if not pdfs:
        print("Usage: python ingest_index.py file1.pdf file2.pdf ...")
    else:
        build_index(pdfs)
