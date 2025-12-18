"""
Text chunker
Splits text into overlapping chunks for better retrieval
"""
from typing import List
from backend.app.config import get_settings
from backend.app.utils.logger import setup_logger

settings = get_settings()
logger = setup_logger(__name__)


class TextChunker:
    """Split text into overlapping chunks for RAG"""
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        """
        Initialize text chunker
        
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        
        logger.info(f"TextChunker initialized: chunk_size={self.chunk_size}, overlap={self.chunk_overlap}")
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Input text to split
        
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            # Get chunk
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence/paragraph boundary for better context
            if end < text_length:
                # Look for best breaking points (prioritize paragraphs, then sentences)
                last_double_newline = chunk.rfind('\n\n')  # Paragraph break
                last_period = chunk.rfind('. ')  # Sentence with space after
                last_newline = chunk.rfind('\n')
                
                # Choose the best break point
                if last_double_newline > self.chunk_size * 0.4:
                    last_break = last_double_newline
                elif last_period > self.chunk_size * 0.4:
                    last_break = last_period + 1  # Include the period
                elif last_newline > self.chunk_size * 0.4:
                    last_break = last_newline
                else:
                    last_break = -1
                
                if last_break > 0:
                    chunk = chunk[:last_break + 1]
                    end = start + last_break + 1
            
            chunks.append(chunk.strip())
            
            # Move to next chunk with overlap
            start = end - self.chunk_overlap
            
            # Prevent infinite loop
            if start <= chunks.__len__() - 1:
                start = end
        
        logger.info(f"Split text into {len(chunks)} chunks")
        return chunks
    
    def split_text_with_metadata(self, text: str, document_id: int) -> List[dict]:
        """
        Split text into chunks with metadata
        
        Args:
            text: Input text to split
            document_id: ID of the source document
        
        Returns:
            List of dictionaries with chunk content and metadata
        """
        chunks = self.split_text(text)
        
        chunks_with_metadata = []
        for idx, chunk in enumerate(chunks):
            chunks_with_metadata.append({
                'content': chunk,
                'chunk_index': idx,
                'document_id': document_id,
                'chunk_size': len(chunk)
            })
        
        return chunks_with_metadata

