# LLM Document Processing System

This is an intelligent query-retrieval system that processes natural language queries and retrieves relevant information from large unstructured documents using **Google Gemini** or **Local LLM models** with free embedding models.

## Features

- PDF document ingestion from URLs
- Semantic text chunking and embedding using SentenceTransformers
- Vector storage using Pinecone
- Intelligent query retrieval using RAG (Retrieval-Augmented Generation)
- **Dual LLM Support**: Choose between Google Gemini or Local LLM (e.g., Ollama)
- Natural language answering with configurable models

## Model Options

### 🤖 Google Gemini
- Uses Google's Gemini-1.5-flash model
- Requires GEMINI_API_KEY
- Cloud-based, fast responses
- High-quality natural language generation

### 🏠 Local LLM
- Supports local LLM servers with OpenAI-compatible APIs (LM Studio, Ollama, etc.)
- No external API required
- Complete privacy and control
- Configurable endpoint and model

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
     - `GEMINI_API_KEY`: Your Google AI API key (for Gemini model)
     - `PINECONE_API_KEY`: Your Pinecone API key
     - `PINECONE_ENVIRONMENT`: Your Pinecone environment (e.g., "us-east-1-aws")
     - `PINECONE_INDEX_NAME`: Name for your Pinecone index
     - `LOCAL_LLM_ENDPOINT`: Your LM Studio endpoint (e.g., "http://localhost:1234/v1/chat/completions")
     - `LOCAL_LLM_MODEL`: Your LM Studio model name (e.g., "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF")

5. **Run the application:**

   **Option 1: Using the startup script (Recommended)**
   ```bash
   # For Google Gemini
   python run_server.py --model gemini
   
   # For Local LLM (e.g., Ollama)
   python run_server.py --model local
   
   # With custom host/port
   python run_server.py --model gemini --host 0.0.0.0 --port 8080 --reload
   ```

   **Option 2: Direct uvicorn command**
   ```bash
   # For Gemini (default)
   uvicorn main:app --reload
   
   # For Local LLM (set environment variable first)
   DEFAULT_MODEL_TYPE=local uvicorn main:app --reload
   ```

   **Option 3: Direct Python execution**
   ```bash
   # For Gemini
   python main.py --model gemini
   
   # For Local LLM
   python main.py --model local
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
