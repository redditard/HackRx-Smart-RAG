# LLM Document Processing System (Gemini)

This is an intelligent query-retrieval system that processes natural language queries and retrieves relevant information from large unstructured documents using Google Gemini and free embedding models.

## Features

- PDF document ingestion from URLs
- Semantic text chunking and embedding using SentenceTransformers
- Vector storage using Pinecone
- Intelligent query retrieval using RAG (Retrieval-Augmented Generation)
- Natural language answering using Google Gemini

## Setup Instructions

1. **Clone and navigate to the project:**
   ```bash
   cd intelligent_retrieval_system
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/macOS
   # or
   .\venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   - Copy `.env.example` to `.env`
   - Fill in your API keys:
     - `GEMINI_API_KEY`: Your Google AI API key
     - `PINECONE_API_KEY`: Your Pinecone API key
     - `PINECONE_ENVIRONMENT`: Your Pinecone environment (e.g., "us-east-1-aws")
     - `PINECONE_INDEX_NAME`: Name for your Pinecone index

5. **Run the application:**
   ```bash
   uvicorn main:app --reload
   ```

6. **Access the API:**
   - API documentation: http://127.0.0.1:8000/docs
   - Main endpoint: POST /hackrx/run

## API Usage

### Authentication
All requests require a Bearer token in the Authorization header:
```
Authorization: Bearer 0Y988a27dc0bf2e755e893e2665069fe4d09189215f7824b023cc07db597bb
```

### Request Format
```json
{
  "documents": "https://example.com/document.pdf",
  "questions": [
    "What is the coverage for knee surgery?",
    "What are the policy terms?"
  ]
}
```

### Response Format
```json
{
  "answers": [
    "Knee surgery is covered under the policy with specific conditions...",
    "The policy terms include..."
  ]
}
```

## Sample Query
- **Input:** "46M, knee surgery, Pune, 3-month policy"
- **Expected Output:** "Yes, knee surgery is covered under the policy."

## Architecture

1. **Document Ingestion:** Downloads and parses PDF documents
2. **Text Chunking:** Splits documents into semantic chunks
3. **Embedding:** Converts text chunks to vector embeddings using SentenceTransformers
4. **Vector Storage:** Stores embeddings in Pinecone
5. **Retrieval:** Finds relevant chunks for each query
6. **Generation:** Uses Google Gemini to generate answers based on retrieved context
