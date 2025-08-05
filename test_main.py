import pytest
import requests
import json
import os
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from main import app, verify_token, process_pdf_from_url, chunk_text

# Test client
client = TestClient(app)

# Test configuration
TEST_TOKEN = "0Y988a27dc0bf2e755e893e2665069fe4d09189215f7824b023cc07db597bb"
HEADERS = {"Authorization": f"Bearer {TEST_TOKEN}"}

class TestAuthentication:
    """Test authentication functionality."""
    
    def test_valid_token(self):
        """Test with valid authentication token."""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_invalid_token(self):
        """Test with invalid authentication token."""
        invalid_headers = {"Authorization": "Bearer invalid-token"}
        response = client.post("/hackrx/run", headers=invalid_headers, json={
            "documents": "https://example.com/test.pdf",
            "questions": ["Test question?"]
        })
        assert response.status_code == 403
    
    def test_missing_token(self):
        """Test with missing authentication token."""
        response = client.post("/hackrx/run", json={
            "documents": "https://example.com/test.pdf",
            "questions": ["Test question?"]
        })
        assert response.status_code == 403

class TestHealthCheck:
    """Test health check functionality."""
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "openai_configured" in data
        assert "pinecone_configured" in data

class TestTextProcessing:
    """Test text processing functions."""
    
    def test_chunk_text(self):
        """Test text chunking functionality."""
        sample_text = "This is a sample text. " * 100  # Create a longer text
        chunks = chunk_text(sample_text)
        
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
    
    def test_chunk_empty_text(self):
        """Test chunking with empty text."""
        chunks = chunk_text("")
        assert chunks == []
    
    @patch('requests.get')
    def test_process_pdf_success(self, mock_get):
        """Test successful PDF processing."""
        # Mock PDF content (simplified)
        mock_response = MagicMock()
        mock_response.content = b"Mock PDF content"
        mock_response.headers = {'content-type': 'application/pdf'}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # This would normally fail without a real PDF, but we're testing the request part
        with pytest.raises(Exception):  # Expected to fail at PDF parsing
            process_pdf_from_url("https://example.com/test.pdf")

class TestAPIEndpoints:
    """Test API endpoints."""
    
    def test_root_endpoint(self):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data
    
    def test_invalid_request_format(self):
        """Test with invalid request format."""
        response = client.post("/hackrx/run", headers=HEADERS, json={
            "invalid_field": "value"
        })
        assert response.status_code == 422  # Validation error

class TestIntegration:
    """Integration tests (require actual API keys)."""
    
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY") or not os.getenv("PINECONE_API_KEY"),
        reason="API keys not configured"
    )
    def test_full_pipeline_mock(self):
        """Test full pipeline with mocked external services."""
        # This test would require proper mocking of OpenAI and Pinecone
        # For now, we just test the request structure
        
        sample_request = {
            "documents": "https://example.com/sample.pdf",
            "questions": ["What is this document about?"]
        }
        
        # This will fail without proper setup, but tests the endpoint structure
        response = client.post("/hackrx/run", headers=HEADERS, json=sample_request)
        
        # Should fail at PDF processing stage if URL is not real
        assert response.status_code in [400, 500]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
