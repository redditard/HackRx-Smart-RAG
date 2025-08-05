#!/usr/bin/env python3
"""
Test script for the LLM Document Processing System (Gemini)
"""

import requests
import json

# Configuration
BASE_URL = "http://127.0.0.1:8000"
AUTH_TOKEN = "0Y988a27dc0bf2e755e893e2665069fe4d09189215f7824b023cc07db597bb"

def test_health_check():
    """Test the health check endpoint."""
    print("Testing health check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Health check status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_main_endpoint():
    """Test the main processing endpoint."""
    print("Testing main endpoint...")
    
    # Sample request data
    sample_request = {
        "documents": "https://example.com/sample-document.pdf",  # Replace with actual PDF URL
        "questions": [
            "What is the coverage for knee surgery?",
            "What are the policy terms for a 46-year-old male?",
            "Is there coverage for surgery in Pune?"
        ]
    }
    
    headers = {
        "Authorization": f"Bearer {AUTH_TOKEN}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/hackrx/run",
            json=sample_request,
            headers=headers,
            timeout=120  # Allow 2 minutes for processing
        )
        
        print(f"Main endpoint status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("Success! Answers received:")
            for i, answer in enumerate(result["answers"], 1):
                print(f"\nQuestion {i}: {sample_request['questions'][i-1]}")
                print(f"Answer: {answer}")
        else:
            print(f"Error response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("Request timed out - this is normal for the first request as it processes the document")
    except Exception as e:
        print(f"Error: {e}")
    
    print()

def test_authentication():
    """Test authentication failure."""
    print("Testing authentication failure...")
    
    sample_request = {
        "documents": "https://example.com/sample-document.pdf",
        "questions": ["Test question"]
    }
    
    # Wrong token
    headers = {
        "Authorization": "Bearer wrong-token",
        "Content-Type": "application/json"
    }
    
    response = requests.post(f"{BASE_URL}/hackrx/run", json=sample_request, headers=headers)
    print(f"Auth failure status: {response.status_code} (should be 403)")
    print(f"Response: {response.json()}")
    print()

if __name__ == "__main__":
    print("=" * 50)
    print("LLM Document Processing System (Gemini) - Test Script")
    print("=" * 50)
    print()
    
    # Test health check
    test_health_check()
    
    # Test authentication
    test_authentication()
    
    # Test main functionality
    print("Note: For the main endpoint test, you need to:")
    print("1. Have valid API keys in your .env file (Gemini + Pinecone)")
    print("2. Replace the sample PDF URL with a real one")
    print("3. Start the server with: uvicorn main:app --reload")
    print()
    
    # Uncomment the line below to test the main endpoint
    # test_main_endpoint()
