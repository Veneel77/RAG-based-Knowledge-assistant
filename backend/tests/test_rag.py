"""
Tests for RAG pipeline components
"""
import pytest
from backend.app.rag.chunker import TextChunker
from backend.app.rag.embedder import Embedder


def test_text_chunker():
    """Test text chunking"""
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    
    text = "This is a test document. " * 50  # Long text
    chunks = chunker.split_text(text)
    
    assert len(chunks) > 0
    assert all(len(chunk) <= 100 + 50 for chunk in chunks)  # Allow some flexibility


def test_empty_text_chunking():
    """Test chunking empty text"""
    chunker = TextChunker()
    chunks = chunker.split_text("")
    
    assert chunks == []


def test_embedder_initialization():
    """Test embedder initialization"""
    embedder = Embedder()
    
    assert embedder.model is not None
    assert embedder.dimension > 0


def test_single_embedding():
    """Test generating single embedding"""
    embedder = Embedder()
    text = "This is a test sentence."
    
    embedding = embedder.embed_text(text)
    
    assert embedding is not None
    assert len(embedding) == embedder.dimension


def test_batch_embeddings():
    """Test generating batch embeddings"""
    embedder = Embedder()
    texts = ["First sentence.", "Second sentence.", "Third sentence."]
    
    embeddings = embedder.embed_texts(texts)
    
    assert embeddings is not None
    assert len(embeddings) == len(texts)
    assert len(embeddings[0]) == embedder.dimension

