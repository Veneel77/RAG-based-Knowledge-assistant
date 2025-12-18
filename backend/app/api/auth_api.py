"""
Authentication API endpoints
Handles user registration and login
"""
from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr

from backend.app.models.database import User, get_db
from backend.app.auth.jwt import create_access_token, get_password_hash, verify_password
from backend.app.auth.auth import get_user_by_username, get_user_by_email, get_current_active_user
from backend.app.config import get_settings
from backend.app.utils.logger import setup_logger

settings = get_settings()
logger = setup_logger(__name__)
router = APIRouter(prefix="/auth", tags=["authentication"])


class UserRegistration(BaseModel):
    """User registration request"""
    username: str
    email: EmailStr
    password: str


class UserLogin(BaseModel):
    """User login request"""
    username: str
    password: str


class Token(BaseModel):
    """Token response"""
    access_token: str
    token_type: str


class UserProfile(BaseModel):
    """User profile response"""
    id: int
    username: str
    email: str
    is_active: bool


@router.post("/register", response_model=Token)
async def register(
    user_data: UserRegistration,
    db: Session = Depends(get_db)
):
    """
    Register a new user
    
    Args:
        user_data: User registration data
    
    Returns:
        Access token
    """
    try:
        # Validate username length
        if len(user_data.username) < 3:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username must be at least 3 characters long"
            )
        
        # Validate password length
        if len(user_data.password) < 6:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password must be at least 6 characters long"
            )
        
        # Check if username exists
        existing_user = get_user_by_username(db, user_data.username)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered"
            )
        
        # Check if email exists
        existing_email = get_user_by_email(db, user_data.email)
        if existing_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Create new user
        hashed_password = get_password_hash(user_data.password)
        new_user = User(
            username=user_data.username,
            email=user_data.email,
            hashed_password=hashed_password,
            is_active=True
        )
        
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        logger.info(f"New user registered: {new_user.username}")
        
        # Create access token
        access_token = create_access_token(
            data={"user_id": new_user.id, "username": new_user.username}
        )
        
        return Token(access_token=access_token, token_type="bearer")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during registration: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}"
        )


@router.post("/login", response_model=Token)
async def login(
    login_data: UserLogin,
    db: Session = Depends(get_db)
):
    """
    Login user
    
    Args:
        login_data: User login credentials
    
    Returns:
        Access token
    """
    try:
        # Get user
        user = get_user_by_username(db, login_data.username)
        
        # Verify user exists and password is correct
        if not user or not verify_password(login_data.password, user.hashed_password):
            logger.warning(f"Failed login attempt for username: {login_data.username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password"
            )
        
        # Check if user is active
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is inactive"
            )
        
        logger.info(f"User logged in: {user.username}")
        
        # Create access token
        access_token = create_access_token(
            data={"user_id": user.id, "username": user.username}
        )
        
        return Token(access_token=access_token, token_type="bearer")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during login: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login failed: {str(e)}"
        )


@router.get("/me", response_model=UserProfile)
async def get_current_user_profile(
    current_user: User = Depends(get_current_active_user)
):
    """Get current user profile"""
    return UserProfile(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        is_active=current_user.is_active
    )

