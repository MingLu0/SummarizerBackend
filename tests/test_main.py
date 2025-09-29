"""
Tests for main FastAPI application.
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app


class TestMainApp:
    """Test main FastAPI application."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Text Summarizer API"
        assert data["version"] == "1.0.0"
        assert data["docs"] == "/docs"
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "text-summarizer-api"
        assert data["version"] == "1.0.0"
    
    def test_docs_endpoint(self, client):
        """Test that docs endpoint is accessible."""
        response = client.get("/docs")
        assert response.status_code == 200
    
    def test_redoc_endpoint(self, client):
        """Test that redoc endpoint is accessible."""
        response = client.get("/redoc")
        assert response.status_code == 200
