#!/usr/bin/env python3
"""
Example client for the LLM Document Processing System

This script demonstrates how to use the API with a real example.
"""

import requests
import json
import time

class DocumentProcessor:
    def __init__(self, base_url="http://127.0.0.1:8000", token=None):
        self.base_url = base_url
        self.token = token or "0Y988a27dc0bf2e755e893e2665069fe4d09189215f7824b023cc07db597bb"
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
    
    def health_check(self):
        """Check if the service is healthy."""
        try:
            response = requests.get(f"{self.base_url}/health")
            return response.status_code == 200, response.json()
        except Exception as e:
            return False, str(e)
    
    def process_document(self, document_url, questions, timeout=120):
        """Process a document with questions."""
        request_data = {
            "documents": document_url,
            "questions": questions
        }
        
        try:
            print(f"Processing document: {document_url}")
            print(f"Questions: {len(questions)}")
            print("This may take a while for the first request...")
            
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/hackrx/run",
                json=request_data,
                headers=self.headers,
                timeout=timeout
            )
            
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                return True, result, processing_time
            else:
                return False, response.json(), processing_time
                
        except requests.exceptions.Timeout:
            return False, "Request timed out", timeout
        except Exception as e:
            return False, str(e), 0

def main():
    """Main example function."""
    print("=" * 60)
    print("LLM Document Processing System - Example Client")
    print("=" * 60)
    
    # Initialize client
    client = DocumentProcessor()
    
    # Health check
    print("\n1. Checking service health...")
    healthy, health_data = client.health_check()
    if healthy:
        print("✓ Service is healthy")
        print(f"  OpenAI configured: {health_data.get('openai_configured', False)}")
        print(f"  Pinecone configured: {health_data.get('pinecone_configured', False)}")
    else:
        print("✗ Service is not healthy")
        print(f"  Error: {health_data}")
        return
    
    # Example document and questions
    print("\n2. Processing example document...")
    
    # You can replace this with any publicly accessible PDF URL
    document_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    
    questions = [
        "What is this document about?",
        "What are the main topics covered?",
        "Is there any specific policy information mentioned?"
    ]
    
    # Process the document
    success, result, processing_time = client.process_document(document_url, questions)
    
    if success:
        print(f"✓ Processing completed in {processing_time:.2f} seconds")
        print("\nResults:")
        print("-" * 40)
        
        answers = result.get("answers", [])
        for i, (question, answer) in enumerate(zip(questions, answers), 1):
            print(f"\nQ{i}: {question}")
            print(f"A{i}: {answer}")
    else:
        print(f"✗ Processing failed: {result}")
    
    print("\n" + "=" * 60)
    print("Example completed!")
    print("\nTo use with your own documents:")
    print("1. Replace document_url with your PDF URL")
    print("2. Modify questions as needed")
    print("3. Ensure your .env file has valid API keys")

if __name__ == "__main__":
    main()
