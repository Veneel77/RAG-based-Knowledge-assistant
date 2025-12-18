"""
History API endpoints
Handles query history management
"""
from typing import List
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel

from backend.app.models.database import User, QueryHistory, get_db
from backend.app.auth.auth import get_current_active_user
from backend.app.utils.logger import setup_logger

logger = setup_logger(__name__)
router = APIRouter(prefix="/history", tags=["history"])


class HistoryItem(BaseModel):
    """History item model"""
    id: int
    query: str
    response: str
    timestamp: datetime
    processing_time: int


class HistoryResponse(BaseModel):
    """Response model for history listing"""
    total: int
    items: List[HistoryItem]


@router.get("/", response_model=HistoryResponse)
async def get_history(
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get query history for current user
    
    Args:
        limit: Maximum number of items to return
        offset: Number of items to skip
    """
    try:
        # Get total count
        total = db.query(QueryHistory).filter(
            QueryHistory.user_id == current_user.id
        ).count()
        
        # Get history items
        history = db.query(QueryHistory).filter(
            QueryHistory.user_id == current_user.id
        ).order_by(QueryHistory.timestamp.desc()).offset(offset).limit(limit).all()
        
        items = [
            HistoryItem(
                id=h.id,
                query=h.query,
                response=h.response,
                timestamp=h.timestamp,
                processing_time=h.processing_time or 0
            )
            for h in history
        ]
        
        return HistoryResponse(total=total, items=items)
    
    except Exception as e:
        logger.error(f"Error fetching history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching history: {str(e)}"
        )


@router.delete("/{history_id}")
async def delete_history_item(
    history_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete a specific history item"""
    history_item = db.query(QueryHistory).filter(
        QueryHistory.id == history_id,
        QueryHistory.user_id == current_user.id
    ).first()
    
    if not history_item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="History item not found"
        )
    
    db.delete(history_item)
    db.commit()
    
    return {"message": "History item deleted successfully"}


@router.delete("/")
async def clear_history(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Clear all history for current user"""
    deleted_count = db.query(QueryHistory).filter(
        QueryHistory.user_id == current_user.id
    ).delete()
    
    db.commit()
    
    return {
        "message": "History cleared successfully",
        "deleted_count": deleted_count
    }

