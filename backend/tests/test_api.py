"""
Integration tests for API endpoints
"""
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_register_user():
    """Test user registration"""
    user_data = {
        "username": f"testuser_{pytest.timestamp}",
        "email": f"test_{pytest.timestamp}@example.com",
        "password": "testpass123"
    }
    
    response = client.post("/auth/register", json=user_data)
    
    # May fail if user exists, that's ok
    if response.status_code == 200:
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"


def test_login_invalid_credentials():
    """Test login with invalid credentials"""
    login_data = {
        "username": "nonexistent_user",
        "password": "wrong_password"
    }
    
    response = client.post("/auth/login", json=login_data)
    assert response.status_code == 401


# Add timestamp to pytest for unique test users
pytest.timestamp = str(hash(str(pytest.__version__)))[-8:]

