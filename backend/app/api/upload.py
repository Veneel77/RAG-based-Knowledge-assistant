"""
Document upload API endpoints
Handles file uploads and document ingestion
"""
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import List
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel

from backend.app.models.database import User, Document, DocumentChunk, get_db
from backend.app.auth.auth import get_current_active_user
from backend.app.rag.loader import DocumentLoader
from backend.app.rag.chunker import TextChunker
from backend.app.rag.embedder import Embedder
from backend.app.rag.vector_store import VectorStore
from backend.app.config import get_settings
from backend.app.utils.logger import setup_logger

settings = get_settings()
logger = setup_logger(__name__)
router = APIRouter(prefix="/upload", tags=["upload"])

# Initialize RAG components (singleton pattern)
embedder = Embedder()
vector_store = VectorStore(dimension=embedder.dimension)
chunker = TextChunker()


class UploadResponse(BaseModel):
    """Response model for document upload"""
    message: str
    document_id: int
    filename: str
    chunk_count: int
    file_size: int


class DocumentInfo(BaseModel):
    """Document information model"""
    id: int
    filename: str
    file_type: str
    file_size: int
    chunk_count: int
    uploaded_at: datetime
    processed: bool


@router.post("/", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Upload and process a document
    
    Process:
    1. Validate file
    2. Save to disk
    3. Extract text
    4. Chunk text
    5. Generate embeddings
    6. Store in vector database
    7. Save metadata to SQL database
    """
    try:
        # Validate file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File type {file_ext} not allowed. Allowed types: {settings.ALLOWED_EXTENSIONS}"
            )
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{current_user.id}_{timestamp}_{file.filename}"
        file_path = settings.UPLOAD_DIR_PATH / unique_filename
        
        # Save file
        logger.info(f"Saving file: {file.filename} for user: {current_user.username}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        file_size = file_path.stat().st_size
        
        # Validate file size
        if file_size > settings.MAX_FILE_SIZE:
            os.remove(file_path)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File size exceeds maximum allowed size of {settings.MAX_FILE_SIZE} bytes"
            )
        
        # Create document record
        document = Document(
            filename=unique_filename,
            original_filename=file.filename,
            file_path=str(file_path),
            file_type=file_ext,
            file_size=file_size,
            user_id=current_user.id,
            processed=False
        )
        db.add(document)
        db.commit()
        db.refresh(document)
        
        logger.info(f"Document record created with ID: {document.id}")
        
        # Load and process document
        try:
            # Extract text
            logger.info(f"Extracting text from document: {document.id}")
            text = DocumentLoader.load_document(str(file_path))
            
            if not text or len(text.strip()) < 10:
                raise ValueError("Document appears to be empty or too short")
            
            # Chunk text
            logger.info(f"Chunking text from document: {document.id}")
            chunks = chunker.split_text(text)
            
            if not chunks:
                raise ValueError("No chunks generated from document")
            
            # Save chunks to database
            chunk_records = []
            for idx, chunk_text in enumerate(chunks):
                chunk = DocumentChunk(
                    document_id=document.id,
                    chunk_index=idx,
                    content=chunk_text
                )
                chunk_records.append(chunk)
            
            db.bulk_save_objects(chunk_records)
            db.commit()
            
            logger.info(f"Saved {len(chunks)} chunks to database")
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(chunks)} chunks")
            embeddings = embedder.embed_texts(chunks)
            
            # Prepare metadata for vector store
            metadata_list = [
                {
                    'content': chunk_text,
                    'document_id': document.id,
                    'chunk_index': idx,
                    'user_id': current_user.id
                }
                for idx, chunk_text in enumerate(chunks)
            ]
            
            # Add to vector store
            logger.info(f"Adding embeddings to vector store")
            vector_store.add_vectors(embeddings, metadata_list)
            vector_store.save()
            
            # Update document record
            document.chunk_count = len(chunks)
            document.processed = True
            db.commit()
            
            logger.info(f"Document {document.id} processed successfully")
            
            return UploadResponse(
                message="Document uploaded and processed successfully",
                document_id=document.id,
                filename=file.filename,
                chunk_count=len(chunks),
                file_size=file_size
            )
        
        except Exception as e:
            # Clean up on error
            logger.error(f"Error processing document: {str(e)}")
            db.delete(document)
            db.commit()
            if file_path.exists():
                os.remove(file_path)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error processing document: {str(e)}"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in upload: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )


@router.get("/documents", response_model=List[DocumentInfo])
async def list_documents(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get list of all documents for current user"""
    documents = db.query(Document).filter(Document.user_id == current_user.id).all()
    
    return [
        DocumentInfo(
            id=doc.id,
            filename=doc.original_filename,
            file_type=doc.file_type,
            file_size=doc.file_size,
            chunk_count=doc.chunk_count,
            uploaded_at=doc.uploaded_at,
            processed=doc.processed
        )
        for doc in documents
    ]


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete a document"""
    document = db.query(Document).filter(
        Document.id == document_id,
        Document.user_id == current_user.id
    ).first()
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    try:
        # Remove from vector store
        vector_store.remove_by_document_id(document_id)
        vector_store.save()
        
        # Delete file
        if os.path.exists(document.file_path):
            os.remove(document.file_path)
        
        # Delete from database (cascades to chunks)
        db.delete(document)
        db.commit()
        
        logger.info(f"Deleted document {document_id}")
        
        return {"message": "Document deleted successfully"}
    
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting document: {str(e)}"
        )

