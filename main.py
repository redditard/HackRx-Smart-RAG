from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl
from typing import List
import os
import io
import logging
from dotenv import load_dotenv
import requests
import pypdf
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="LLM Document Processing System (Gemini)",
    description="Intelligent query-retrieval system for processing documents using Google Gemini",
    version="1.0.0"
)

# Security setup
security = HTTPBearer()
EXPECTED_TOKEN = "0Y988a27dc0bf2e755e893e2665069fe4d09189215f7824b023cc07db597bb"

# API Keys and configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hackrx-documents")

# Initialize clients
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    # Initialize Gemini model
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
else:
    logger.warning("GEMINI_API_KEY not found in environment variables")
    gemini_model = None

# Initialize embedding model (using sentence-transformers for free embeddings)
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    EMBEDDING_DIMENSION = 384  # dimension for all-MiniLM-L6-v2
    logger.info("Embedding model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load embedding model: {e}")
    embedding_model = None

# Initialize Pinecone client
pc = None
if PINECONE_API_KEY:
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        logger.info("Pinecone client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone client: {e}")
else:
    logger.warning("PINECONE_API_KEY not found in environment variables")

# Pydantic models
class QueryRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# Authentication dependency
async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.scheme != "Bearer" or credentials.credentials != EXPECTED_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid or missing authentication token")
    return credentials.credentials

# Helper functions
def process_pdf_from_url(url: str) -> str:
    """Download and extract text from a PDF URL."""
    try:
        logger.info(f"Downloading PDF from: {url}")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Check if the content is actually a PDF
        content_type = response.headers.get('content-type', '')
        if 'pdf' not in content_type.lower():
            logger.warning(f"Content type is {content_type}, proceeding anyway")
        
        # Read PDF content
        pdf_content = io.BytesIO(response.content)
        pdf_reader = pypdf.PdfReader(pdf_content)
        
        # Extract text from all pages
        full_text = ""
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                text = page.extract_text()
                full_text += f"\n\nPage {page_num + 1}:\n{text}"
            except Exception as e:
                logger.warning(f"Error extracting text from page {page_num + 1}: {e}")
        
        logger.info(f"Successfully extracted {len(full_text)} characters from PDF")
        return full_text
        
    except requests.RequestException as e:
        logger.error(f"Error downloading PDF: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to download PDF: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to process PDF: {str(e)}")

def chunk_text(text: str) -> List[str]:
    """Split text into chunks for embedding."""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(text)
        logger.info(f"Created {len(chunks)} text chunks")
        return chunks
    except Exception as e:
        logger.error(f"Error chunking text: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to chunk text: {str(e)}")

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a list of texts using SentenceTransformers."""
    try:
        if embedding_model is None:
            raise Exception("Embedding model not initialized")
        
        embeddings = embedding_model.encode(texts, convert_to_tensor=False)
        # Convert numpy arrays to lists
        embeddings_list = [embedding.tolist() for embedding in embeddings]
        logger.info(f"Generated embeddings for {len(texts)} texts")
        return embeddings_list
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate embeddings: {str(e)}")

def embed_and_store(chunks: List[str], document_id: str):
    """Generate embeddings and store in Pinecone."""
    try:
        # Check if Pinecone client is initialized
        if pc is None:
            raise Exception("Pinecone client not initialized")
        
        # Check if index exists, create if not
        if PINECONE_INDEX_NAME not in pc.list_indexes().names():
            logger.info(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=EMBEDDING_DIMENSION,  # all-MiniLM-L6-v2 dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        
        # Connect to index
        index = pc.Index(PINECONE_INDEX_NAME)
        
        # Generate embeddings
        embeddings = get_embeddings(chunks)
        
        # Prepare data for upsert
        vectors_to_upsert = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vector_id = f"{document_id}-chunk-{i}"
            metadata = {
                'text': chunk,
                'doc_url': document_id,
                'chunk_index': i
            }
            vectors_to_upsert.append((vector_id, embedding, metadata))
        
        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            index.upsert(vectors=batch)
        
        logger.info(f"Successfully stored {len(chunks)} chunks in Pinecone")
        
    except Exception as e:
        logger.error(f"Error storing in Pinecone: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to store embeddings: {str(e)}")

def retrieve_relevant_chunks(question: str, top_k: int = 5) -> str:
    """Retrieve relevant text chunks for a question."""
    try:
        # Check if Pinecone client is initialized
        if pc is None:
            raise Exception("Pinecone client not initialized")
        
        # Connect to index
        index = pc.Index(PINECONE_INDEX_NAME)
        
        # Generate embedding for the question
        question_embedding = get_embeddings([question])[0]
        
        # Query Pinecone
        query_response = index.query(
            vector=question_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # Extract relevant text chunks
        relevant_chunks = []
        for match in query_response['matches']:
            if 'metadata' in match and 'text' in match['metadata']:
                chunk_text = match['metadata']['text']
                score = match['score']
                relevant_chunks.append(f"[Relevance: {score:.3f}] {chunk_text}")
        
        context = "\n\n".join(relevant_chunks)
        logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks for question")
        return context
        
    except Exception as e:
        logger.error(f"Error retrieving chunks: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve relevant information: {str(e)}")

def generate_answer(context: str, question: str) -> str:
    """Generate an answer using Gemini based on the context."""
    try:
        if gemini_model is None:
            raise Exception("Gemini model not initialized")
        
        prompt = f"""You are a helpful assistant for answering questions based on a provided document.
Your task is to answer the user's question accurately and concisely based ONLY on the context below.
Do not use any external knowledge. If the information is not present in the context, you must state "The provided document does not contain information on this topic."

Be specific and reference relevant details from the document when possible.

CONTEXT:
---
{context}
---

QUESTION: {question}

ANSWER:"""

        response = gemini_model.generate_content(prompt)
        
        if response.text:
            answer = response.text.strip()
        else:
            answer = "I was unable to generate an answer based on the provided context."
        
        logger.info(f"Generated answer for question: {question[:50]}...")
        return answer
        
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {str(e)}")

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with basic information."""
    return {
        "message": "LLM Document Processing System (Gemini)",
        "version": "1.0.0",
        "endpoints": {
            "main": "/hackrx/run",
            "docs": "/docs"
        }
    }

@app.post("/hackrx/run", response_model=QueryResponse, dependencies=[Depends(verify_token)])
async def run_query(request: QueryRequest):
    """Main endpoint for processing document queries."""
    try:
        logger.info(f"Processing request with {len(request.questions)} questions")
        
        # Step 1: Download and extract text from PDF
        full_text = process_pdf_from_url(str(request.documents))
        
        if not full_text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the PDF")
        
        # Step 2: Chunk the text
        chunks = chunk_text(full_text)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No text chunks could be created")
        
        # Step 3: Generate embeddings and store in Pinecone
        document_id = str(request.documents).replace("://", "_").replace("/", "_")
        embed_and_store(chunks, document_id)
        
        # Step 4: Process each question
        final_answers = []
        for i, question in enumerate(request.questions):
            logger.info(f"Processing question {i+1}/{len(request.questions)}: {question}")
            
            # Retrieve relevant context
            context = retrieve_relevant_chunks(question)
            
            if not context.strip():
                answer = "No relevant information found in the document for this question."
            else:
                # Generate answer using LLM
                answer = generate_answer(context, question)
            
            final_answers.append(answer)
        
        logger.info(f"Successfully processed all {len(request.questions)} questions")
        return QueryResponse(answers=final_answers)
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing request: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "gemini_configured": bool(GEMINI_API_KEY),
        "pinecone_configured": bool(PINECONE_API_KEY),
        "embedding_model_loaded": embedding_model is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
