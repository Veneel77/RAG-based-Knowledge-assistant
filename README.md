# RAG-Based Knowledge Assistant

A Retrieval-Augmented Generation (RAG) chatbot that answers questions from custom documents (PDFs, research papers, job descriptions, reports).

## Features
- Upload & index multiple PDFs
- Vector search using **FAISS + Hugging Face embeddings**
- Interactive Q&A via **Streamlit**
- Source citations for transparency

## Tech Stack
- **Embeddings:** sentence-transformers/all-MiniLM-L6-v2  
- **Vector DB:** FAISS  
- **Framework:** Streamlit  
- **Deployment:** Streamlit Cloud (free hosting)

## Setup
```bash
pip install -r requirements.txt
python ingest_index.py docs/AI1.pdf docs/aiimpact.pdf
streamlit run app.py

## Future Work

Integrate open/free LLMs (GPT4All / Llama.cpp)

Enable cloud APIs (Hugging Face Inference / OpenAI) for better text generation

Add evaluation metrics (precision/recall on retrieval)

