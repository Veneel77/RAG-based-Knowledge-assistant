"""
Tests for authentication system
"""
import pytest
from backend.app.auth.jwt import (
    get_password_hash,
    verify_password,
    create_access_token,
    verify_token
)


def test_password_hashing():
    """Test password hashing and verification"""
    password = "test_password_123"
    hashed = get_password_hash(password)
    
    assert hashed != password
    assert verify_password(password, hashed) is True
    assert verify_password("wrong_password", hashed) is False


def test_jwt_token_creation():
    """Test JWT token creation"""
    data = {"user_id": 1, "username": "testuser"}
    token = create_access_token(data)
    
    assert token is not None
    assert isinstance(token, str)
    assert len(token) > 0


def test_jwt_token_verification():
    """Test JWT token verification"""
    data = {"user_id": 1, "username": "testuser"}
    token = create_access_token(data)
    
    payload = verify_token(token)
    
    assert payload is not None
    assert payload["user_id"] == 1
    assert payload["username"] == "testuser"


def test_invalid_token():
    """Test invalid token verification"""
    invalid_token = "invalid.token.here"
    payload = verify_token(invalid_token)
    
    assert payload is None

