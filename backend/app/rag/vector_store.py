"""
Vector store using FAISS
Manages vector storage and similarity search
"""
import os
import pickle
from typing import List, Tuple, Optional
import numpy as np
import faiss
from backend.app.config import get_settings
from backend.app.utils.logger import setup_logger

settings = get_settings()
logger = setup_logger(__name__)


class VectorStore:
    """FAISS-based vector store for similarity search"""
    
    def __init__(self, dimension: int = None):
        """
        Initialize vector store
        
        Args:
            dimension: Dimension of embedding vectors
        """
        self.dimension = dimension or settings.VECTOR_DIMENSION
        self.index_path = settings.FAISS_INDEX_PATH
        self.metadata_path = f"{self.index_path}_metadata.pkl"
        
        # Initialize index
        self.index = None
        self.metadata = []  # Store document metadata
        
        # Try to load existing index
        if self.load():
            logger.info(f"Loaded existing FAISS index with {self.index.ntotal} vectors")
        else:
            self._create_new_index()
            logger.info(f"Created new FAISS index with dimension {self.dimension}")
    
    def _create_new_index(self):
        """Create a new FAISS index"""
        # Using IndexFlatL2 for exact search (good for smaller datasets)
        # For larger datasets, consider IndexIVFFlat or IndexHNSWFlat
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []
    
    def add_vectors(
        self,
        vectors: np.ndarray,
        metadata_list: List[dict]
    ):
        """
        Add vectors to the index
        
        Args:
            vectors: Numpy array of vectors to add
            metadata_list: List of metadata dictionaries for each vector
        """
        if len(vectors) != len(metadata_list):
            raise ValueError("Number of vectors and metadata entries must match")
        
        # Ensure vectors are float32
        vectors = vectors.astype('float32')
        
        # Add to index
        self.index.add(vectors)
        self.metadata.extend(metadata_list)
        
        logger.info(f"Added {len(vectors)} vectors to index. Total: {self.index.ntotal}")
    
    def search(
        self,
        query_vector: np.ndarray,
        k: int = None
    ) -> List[Tuple[dict, float]]:
        """
        Search for similar vectors
        
        Args:
            query_vector: Query vector
            k: Number of results to return
        
        Returns:
            List of (metadata, distance) tuples
        """
        k = k or settings.TOP_K_RESULTS
        
        if self.index.ntotal == 0:
            logger.warning("Index is empty, no results to return")
            return []
        
        # Ensure query is float32 and 2D
        query_vector = query_vector.astype('float32').reshape(1, -1)
        
        # Search
        k = min(k, self.index.ntotal)  # Can't return more than we have
        distances, indices = self.index.search(query_vector, k)
        
        # Prepare results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.metadata):
                results.append((self.metadata[idx], float(distance)))
        
        logger.info(f"Found {len(results)} results for query")
        return results
    
    def save(self) -> bool:
        """
        Save index and metadata to disk
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, self.index_path)
            
            # Save metadata
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
            
            logger.info(f"Saved index with {self.index.ntotal} vectors to {self.index_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            return False
    
    def load(self) -> bool:
        """
        Load index and metadata from disk
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(self.index_path) or not os.path.exists(self.metadata_path):
                return False
            
            # Load FAISS index
            self.index = faiss.read_index(self.index_path)
            
            # Load metadata
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            
            return True
        
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            return False
    
    def clear(self):
        """Clear the index and metadata"""
        self._create_new_index()
        logger.info("Cleared vector store")
    
    def remove_by_document_id(self, document_id: int):
        """
        Remove all vectors for a specific document
        Note: FAISS doesn't support deletion, so we rebuild the index
        
        Args:
            document_id: ID of document to remove
        """
        # Filter out metadata for the document
        new_metadata = [m for m in self.metadata if m.get('document_id') != document_id]
        
        if len(new_metadata) == len(self.metadata):
            logger.warning(f"No vectors found for document_id {document_id}")
            return
        
        # Get indices to keep
        indices_to_keep = [i for i, m in enumerate(self.metadata) if m.get('document_id') != document_id]
        
        # Rebuild index
        if indices_to_keep:
            # Extract vectors we want to keep
            old_vectors = []
            for idx in indices_to_keep:
                vector = self.index.reconstruct(idx)
                old_vectors.append(vector)
            
            # Create new index
            self._create_new_index()
            
            # Add back the vectors we kept
            if old_vectors:
                vectors = np.array(old_vectors)
                self.add_vectors(vectors, new_metadata)
        else:
            # No vectors left, clear everything
            self.clear()
        
        logger.info(f"Removed vectors for document_id {document_id}")

