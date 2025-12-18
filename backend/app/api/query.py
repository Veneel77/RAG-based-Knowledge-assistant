"""
Query API endpoints
Handles RAG queries and response generation
"""
import time
import json
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel

from backend.app.models.database import User, QueryHistory, Document, get_db
from backend.app.auth.auth import get_current_active_user
from backend.app.rag.retriever import Retriever
from backend.app.rag.generator import Generator
from backend.app.config import get_settings
from backend.app.utils.logger import setup_logger

settings = get_settings()
logger = setup_logger(__name__)
router = APIRouter(prefix="/query", tags=["query"])

# Initialize RAG components (will be created per request)
# retriever = Retriever()
generator = Generator()


class QueryRequest(BaseModel):
    """Request model for query"""
    query: str
    k: Optional[int] = None  # Number of results to retrieve


class SourceInfo(BaseModel):
    """Source information in response"""
    document_id: int
    document_name: str
    chunk_index: int
    content: str
    similarity_score: float


class QueryResponse(BaseModel):
    """Response model for query"""
    query: str
    response: str
    sources: List[SourceInfo]
    processing_time: int  # in milliseconds


@router.post("/", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Query documents using RAG
    
    Process:
    1. Retrieve relevant chunks
    2. Get document context
    3. Generate response using Gemini
    4. Save to history
    5. Return response with sources
    """
    start_time = time.time()
    
    try:
        if not request.query or len(request.query.strip()) < 3:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query must be at least 3 characters long"
            )
        
        logger.info(f"Processing query from user {current_user.username}: {request.query[:100]}")
        
        # Create a fresh retriever to ensure we have the latest index
        retriever = Retriever(reload_index=True)
        
        # Retrieve relevant chunks
        k = request.k or settings.TOP_K_RESULTS
        retrieved_chunks = retriever.retrieve(request.query, k=k)
        
        if not retrieved_chunks:
            logger.warning("No relevant chunks found")
            response_text = "I couldn't find any relevant information in the uploaded documents to answer your question. Please make sure you've uploaded relevant documents."
            sources = []
        else:
            # Get context from retrieved chunks
            context_parts = []
            for idx, chunk in enumerate(retrieved_chunks, 1):
                context_parts.append(f"[Source {idx}]\n{chunk['content']}\n")
            context = "\n".join(context_parts)
            
            # Get conversation history (last 3 exchanges)
            history = db.query(QueryHistory).filter(
                QueryHistory.user_id == current_user.id
            ).order_by(QueryHistory.timestamp.desc()).limit(3).all()
            
            conversation_history = [
                {'query': h.query, 'response': h.response}
                for h in reversed(history)
            ]
            
            # Generate response
            response_text = generator.generate_response(
                query=request.query,
                context=context,
                conversation_history=conversation_history
            )
            
            # Prepare sources
            sources = []
            for chunk in retrieved_chunks:
                document = db.query(Document).filter(
                    Document.id == chunk['document_id']
                ).first()
                
                if document:
                    sources.append(SourceInfo(
                        document_id=document.id,
                        document_name=document.original_filename,
                        chunk_index=chunk['chunk_index'],
                        content=chunk['content'][:200] + "...",  # Truncate for response
                        similarity_score=chunk['similarity_score']
                    ))
        
        # Calculate processing time
        processing_time = int((time.time() - start_time) * 1000)
        
        # Save to history
        history_entry = QueryHistory(
            user_id=current_user.id,
            query=request.query,
            response=response_text,
            sources=json.dumps([s.dict() for s in sources]),
            processing_time=processing_time
        )
        db.add(history_entry)
        db.commit()
        
        logger.info(f"Query processed successfully in {processing_time}ms")
        
        return QueryResponse(
            query=request.query,
            response=response_text,
            sources=sources,
            processing_time=processing_time
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """Check if query service is healthy"""
    try:
        # Check if Gemini API is accessible
        is_healthy = generator.check_api_key()
        
        # Create retriever to check vector store
        temp_retriever = Retriever(reload_index=True)
        
        return {
            "status": "healthy" if is_healthy else "degraded",
            "gemini_api": "connected" if is_healthy else "error",
            "retriever": "ready",
            "vector_store_size": temp_retriever.vector_store.index.ntotal
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

