"""
Retriever
Orchestrates the retrieval process using embeddings and vector store
"""
from typing import List, Dict
from backend.app.rag.embedder import Embedder
from backend.app.rag.vector_store import VectorStore
from backend.app.config import get_settings
from backend.app.utils.logger import setup_logger

settings = get_settings()
logger = setup_logger(__name__)


class Retriever:
    """Retrieve relevant document chunks for a query"""
    
    def __init__(self):
        """Initialize retriever with embedder and vector store"""
        logger.info("Initializing Retriever")
        self.embedder = Embedder()
        self.vector_store = VectorStore(dimension=self.embedder.dimension)
        logger.info("Retriever initialized successfully")
    
    def retrieve(self, query: str, k: int = None) -> List[Dict]:
        """
        Retrieve top-k most relevant chunks for a query
        
        Args:
            query: User query
            k: Number of results to return
        
        Returns:
            List of retrieved chunks with metadata and scores
        """
        k = k or settings.TOP_K_RESULTS
        
        try:
            # Generate query embedding
            logger.info(f"Retrieving top-{k} results for query: {query[:100]}...")
            query_embedding = self.embedder.embed_text(query)
            
            # Search vector store
            results = self.vector_store.search(query_embedding, k=k)
            
            # Format results
            retrieved_chunks = []
            for metadata, distance in results:
                chunk_data = {
                    'content': metadata.get('content', ''),
                    'document_id': metadata.get('document_id'),
                    'chunk_index': metadata.get('chunk_index'),
                    'distance': distance,
                    'similarity_score': 1 / (1 + distance)  # Convert distance to similarity
                }
                retrieved_chunks.append(chunk_data)
            
            logger.info(f"Retrieved {len(retrieved_chunks)} chunks")
            return retrieved_chunks
        
        except Exception as e:
            logger.error(f"Error during retrieval: {str(e)}")
            raise
    
    def get_context(self, query: str, k: int = None) -> str:
        """
        Get formatted context string from retrieved chunks
        
        Args:
            query: User query
            k: Number of chunks to retrieve
        
        Returns:
            Formatted context string
        """
        chunks = self.retrieve(query, k=k)
        
        if not chunks:
            return ""
        
        # Format context
        context_parts = []
        for idx, chunk in enumerate(chunks, 1):
            context_parts.append(f"[Source {idx}]\n{chunk['content']}\n")
        
        context = "\n".join(context_parts)
        logger.info(f"Generated context of {len(context)} characters")
        
        return context

