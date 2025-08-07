from fastapi import FastAPI, HTTPException, Depends, Security, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Dict, Any
import os
import io
import logging
import argparse
import sys
import hashlib
import json
import asyncio
from datetime import datetime, timedelta
from dotenv import load_dotenv
import requests
import pypdf
import pdfplumber
import pandas as pd
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
import aiofiles
import aiohttp
from aiohttp import TCPConnector
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import concurrent.futures
import multiprocessing

# Load environment variables
load_dotenv()

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='LLM Document Processing System')
    parser.add_argument('--model', choices=['gemini', 'local', 'together'], 
                       default=os.getenv('DEFAULT_MODEL_TYPE', 'gemini'),
                       help='Choose the LLM model: gemini, local, or together (default: gemini)')
    return parser.parse_args()

# Initialize based on command line args (if called directly)
args = None
if __name__ == "__main__":
    args = parse_args()
    MODEL_TYPE = args.model
else:
    # When imported as a module, use environment variable or default
    MODEL_TYPE = os.getenv('DEFAULT_MODEL_TYPE', 'gemini')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f"Starting server with {MODEL_TYPE.upper()} model")

# Initialize FastAPI app
app = FastAPI(
    title=f"LLM Document Processing System ({MODEL_TYPE.title()})",
    description=f"Intelligent query-retrieval system using {MODEL_TYPE.title()} for document processing",
    version="1.0.0"
)

# Security setup
security = HTTPBearer()
EXPECTED_TOKEN = "09988a27dc0bf3ef755e893e2e6650693e4009189215f7824b023cc07db59b1b"

# API Keys and configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hackrx-documents")

# Local LLM configuration
LOCAL_LLM_ENDPOINT = os.getenv("LOCAL_LLM_ENDPOINT", "http://localhost:1234/v1/chat/completions")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF")
LOCAL_LLM_TIMEOUT = int(os.getenv("LOCAL_LLM_TIMEOUT", "120"))

# Together AI configuration
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
TOGETHER_MODEL = os.getenv("TOGETHER_MODEL", "meta-llama/Llama-3-8b-chat-hf")
TOGETHER_TIMEOUT = int(os.getenv("TOGETHER_TIMEOUT", "120"))

# Caching configuration
CACHE_DIR = os.getenv("CACHE_DIR", "./cache")
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
CACHE_TTL_HOURS = int(os.getenv("CACHE_TTL_HOURS", "24"))

# Performance configuration
BATCH_SIZE = int(os.getenv("PINECONE_BATCH_SIZE", "50"))  # Reduced for better memory usage
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "3"))  # Reduced for stability
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "25"))  # Optimized embedding batch size
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "3"))  # Reduced for faster retrieval
CONNECTION_POOL_SIZE = int(os.getenv("CONNECTION_POOL_SIZE", "100"))
CONNECTION_POOL_PER_HOST = int(os.getenv("CONNECTION_POOL_PER_HOST", "30"))

# Global HTTP session for connection pooling
global_session = None

async def get_http_session():
    """Get or create a global HTTP session with optimized connection pooling."""
    global global_session
    if global_session is None or global_session.closed:
        connector = TCPConnector(
            limit=CONNECTION_POOL_SIZE,
            limit_per_host=CONNECTION_POOL_PER_HOST,
            ttl_dns_cache=300,
            use_dns_cache=True,
            enable_cleanup_closed=True,
            keepalive_timeout=30
        )
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        global_session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        )
    return global_session

# Create cache directory if it doesn't exist
if CACHE_ENABLED:
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(f"{CACHE_DIR}/documents", exist_ok=True)
    os.makedirs(f"{CACHE_DIR}/embeddings", exist_ok=True)

# Initialize clients based on model type
gemini_model = None
local_llm_available = False

if MODEL_TYPE == 'gemini':
    if GEMINI_API_KEY:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            gemini_model = genai.GenerativeModel('gemini-2.5-flash-lite')
            logger.info("Gemini model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {e}")
    else:
        logger.warning("GEMINI_API_KEY not found in environment variables")
elif MODEL_TYPE == 'local':
    # Test local LLM availability (LM Studio - OpenAI compatible)
    try:
        # For LM Studio, check the models endpoint
        models_endpoint = LOCAL_LLM_ENDPOINT.replace('/chat/completions', '/models')
        response = requests.get(models_endpoint, timeout=5)
        if response.status_code == 200:
            local_llm_available = True
            logger.info(f"LM Studio endpoint available at {LOCAL_LLM_ENDPOINT}")
            # Try to get available models
            try:
                models_data = response.json()
                if 'data' in models_data and len(models_data['data']) > 0:
                    available_models = [model['id'] for model in models_data['data']]
                    logger.info(f"Available models: {available_models}")
                else:
                    logger.warning("No models loaded in LM Studio")
            except:
                logger.info("Connected to LM Studio but couldn't parse models list")
        else:
            logger.error(f"LM Studio endpoint returned status {response.status_code}")
    except Exception as e:
        logger.error(f"Failed to connect to LM Studio at {LOCAL_LLM_ENDPOINT}: {e}")
        logger.warning("LM Studio not available, falling back to Gemini if configured")
        # Fallback to Gemini if local LLM is not available
        if GEMINI_API_KEY:
            try:
                genai.configure(api_key=GEMINI_API_KEY)
                gemini_model = genai.GenerativeModel('gemini-2.5-flash-lite')
                MODEL_TYPE = 'gemini'
                logger.info("Fallback: Gemini model initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize fallback Gemini model: {e}")

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
def get_document_hash(url: str) -> str:
    """Generate a hash for the document URL for caching."""
    return hashlib.md5(url.encode()).hexdigest()

def get_cache_file_path(doc_hash: str, cache_type: str) -> str:
    """Get the cache file path for a document."""
    return os.path.join(CACHE_DIR, cache_type, f"{doc_hash}.json")

async def is_cache_valid(cache_file: str) -> bool:
    """Check if cache file exists and is still valid."""
    if not CACHE_ENABLED or not os.path.exists(cache_file):
        return False
    
    try:
        stat = os.stat(cache_file)
        cache_time = datetime.fromtimestamp(stat.st_mtime)
        expiry_time = cache_time + timedelta(hours=CACHE_TTL_HOURS)
        return datetime.now() < expiry_time
    except Exception:
        return False

async def load_from_cache(cache_file: str) -> Optional[Any]:
    """Load data from cache file."""
    try:
        async with aiofiles.open(cache_file, 'r') as f:
            content = await f.read()
            return json.loads(content)
    except Exception as e:
        logger.warning(f"Failed to load cache from {cache_file}: {e}")
        return None

async def save_to_cache(cache_file: str, data: Any) -> None:
    """Save data to cache file."""
    try:
        async with aiofiles.open(cache_file, 'w') as f:
            await f.write(json.dumps(data, default=str))
        logger.info(f"Cached data to {cache_file}")
    except Exception as e:
        logger.warning(f"Failed to cache data to {cache_file}: {e}")

async def smart_cache_check(url: str) -> tuple[bool, Optional[str], Optional[List[str]], Optional[str]]:
    """Smart cache check that returns document text, chunks, and embeddings if available."""
    doc_hash = get_document_hash(url)
    
    # Check document cache
    doc_cache_file = get_cache_file_path(doc_hash, "documents")
    text_cached = False
    cached_text = None
    
    if await is_cache_valid(doc_cache_file):
        cached_doc = await load_from_cache(doc_cache_file)
        if cached_doc and 'text' in cached_doc:
            cached_text = cached_doc['text']
            text_cached = True
    
    # Check if chunks are already processed and stored
    chunks_cache_file = get_cache_file_path(doc_hash, "chunks")
    cached_chunks = None
    
    if await is_cache_valid(chunks_cache_file):
        cached_chunk_data = await load_from_cache(chunks_cache_file)
        if cached_chunk_data and 'chunks' in cached_chunk_data:
            cached_chunks = cached_chunk_data['chunks']
    
    return text_cached, cached_text, cached_chunks, doc_hash

async def cache_chunks(doc_hash: str, chunks: List[str]) -> None:
    """Cache text chunks for faster processing."""
    if CACHE_ENABLED:
        cache_file = get_cache_file_path(doc_hash, "chunks")
        cache_data = {
            'doc_hash': doc_hash,
            'chunks': chunks,
            'chunk_count': len(chunks),
            'created_at': datetime.now().isoformat()
        }
        await save_to_cache(cache_file, cache_data)

async def process_pdf_from_url_async(url: str) -> str:
    """Download and extract text from a PDF URL with optimized caching and connection pooling."""
    try:
        doc_hash = get_document_hash(url)
        cache_file = get_cache_file_path(doc_hash, "documents")
        
        # Check cache first
        if await is_cache_valid(cache_file):
            logger.info(f"Loading PDF text from cache: {doc_hash}")
            cached_data = await load_from_cache(cache_file)
            if cached_data and 'text' in cached_data:
                return cached_data['text']
        
        logger.info(f"Downloading PDF from: {url}")
        
        # Use connection pooling for better performance
        session = await get_http_session()
        async with session.get(url) as response:
            response.raise_for_status()
            
            # Check if the content is actually a PDF
            content_type = response.headers.get('content-type', '')
            if 'pdf' not in content_type.lower():
                logger.warning(f"Content type is {content_type}, proceeding anyway")
            
            # Read PDF content
            pdf_data = await response.read()
            
            # Process PDF in executor to avoid blocking
            full_text = await asyncio.get_event_loop().run_in_executor(
                None, 
                _extract_pdf_text, 
                pdf_data
            )
            
            logger.info(f"Successfully extracted {len(full_text)} characters from PDF")
            
            # Cache the result
            if CACHE_ENABLED:
                cache_data = {
                    'url': url,
                    'text': full_text,
                    'extracted_at': datetime.now().isoformat(),
                    'char_count': len(full_text),
                    'doc_hash': doc_hash
                }
                await save_to_cache(cache_file, cache_data)
            
            return full_text
        
    except aiohttp.ClientError as e:
        logger.error(f"Error downloading PDF: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to download PDF: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to process PDF: {str(e)}")

def _extract_pdf_text(pdf_data: bytes) -> str:
    """Extract text from PDF data with enhanced table handling."""
    try:
        pdf_content = io.BytesIO(pdf_data)
        full_text = ""
        
        # Method 1: Use pdfplumber for better table extraction
        try:
            import pdfplumber
            with pdfplumber.open(pdf_content) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        # Extract regular text
                        text = page.extract_text() or ""
                        
                        # Extract tables separately
                        tables = page.extract_tables()
                        table_text = ""
                        
                        if tables:
                            for table_num, table in enumerate(tables):
                                if table:
                                    table_text += f"\n\n--- Table {table_num + 1} on Page {page_num + 1} ---\n"
                                    
                                    # Convert table to formatted text
                                    for row_num, row in enumerate(table):
                                        if row:  # Skip empty rows
                                            # Clean and join row data
                                            cleaned_row = [str(cell).strip() if cell else "" for cell in row]
                                            if any(cleaned_row):  # Only add non-empty rows
                                                if row_num == 0:  # Header row
                                                    table_text += " | ".join(cleaned_row) + "\n"
                                                    table_text += "-" * len(" | ".join(cleaned_row)) + "\n"
                                                else:
                                                    table_text += " | ".join(cleaned_row) + "\n"
                        
                        # Combine text and tables
                        page_content = f"\n\nPage {page_num + 1}:\n{text}"
                        if table_text:
                            page_content += f"\n{table_text}"
                        
                        full_text += page_content
                        
                    except Exception as e:
                        logger.warning(f"pdfplumber error on page {page_num + 1}: {e}")
                        
        except ImportError:
            logger.warning("pdfplumber not available, falling back to pypdf")
            
        # Method 2: Fallback to pypdf if pdfplumber fails or isn't available
        if not full_text.strip():
            pdf_content.seek(0)  # Reset stream position
            pdf_reader = pypdf.PdfReader(pdf_content)
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    text = page.extract_text()
                    full_text += f"\n\nPage {page_num + 1}:\n{text}"
                except Exception as e:
                    logger.warning(f"pypdf error on page {page_num + 1}: {e}")
        
        # Method 3: Enhanced text cleaning for better table recognition
        if full_text:
            # Clean up common table formatting issues
            lines = full_text.split('\n')
            cleaned_lines = []
            
            for line in lines:
                # Detect potential table rows (lines with multiple spaces/tabs)
                if '   ' in line or '\t' in line:
                    # Convert multiple spaces to pipe separators for better parsing
                    import re
                    line = re.sub(r'\s{3,}', ' | ', line)
                
                cleaned_lines.append(line)
            
            full_text = '\n'.join(cleaned_lines)
        
        return full_text
        
    except Exception as e:
        logger.error(f"Error in enhanced PDF text extraction: {e}")
        # Final fallback - basic extraction
        try:
            pdf_content.seek(0)
            pdf_reader = pypdf.PdfReader(pdf_content)
            fallback_text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    text = page.extract_text()
                    fallback_text += f"\n\nPage {page_num + 1}:\n{text}"
                except:
                    pass
            return fallback_text
        except:
            raise Exception("Failed to extract any text from PDF")

def process_pdf_from_url(url: str) -> str:
    """Synchronous wrapper for async PDF processing."""
    return asyncio.run(process_pdf_from_url_async(url))

def chunk_text(text: str) -> List[str]:
    """Split text into chunks with enhanced table preservation."""
    try:
        # Enhanced chunking strategy for tables and structured data
        
        # Step 1: Identify and preserve table sections
        lines = text.split('\n')
        table_sections = []
        regular_text = []
        current_table = []
        in_table = False
        
        for i, line in enumerate(lines):
            # Detect table patterns (lines with multiple | or consistent spacing)
            is_table_line = (
                '|' in line and line.count('|') >= 2 or
                (line.strip().startswith('-') and len(line.strip()) > 10) or
                (in_table and line.strip() and ('|' in line or line.count('  ') >= 2))
            )
            
            if is_table_line:
                if not in_table:
                    # Starting a new table
                    if current_table:
                        table_sections.append('\n'.join(current_table))
                        current_table = []
                    in_table = True
                current_table.append(line)
            else:
                if in_table and current_table:
                    # End of table
                    table_sections.append('\n'.join(current_table))
                    current_table = []
                    in_table = False
                
                if not in_table:
                    regular_text.append(line)
        
        # Don't forget the last table if file ends with it
        if current_table:
            table_sections.append('\n'.join(current_table))
        
        # Step 2: Chunk regular text normally
        regular_text_str = '\n'.join(regular_text)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Using config value
            chunk_overlap=150,  # Using config value
            separators=["\n\n", "\n", " ", ""]
        )
        regular_chunks = text_splitter.split_text(regular_text_str) if regular_text_str.strip() else []
        
        # Step 3: Handle table sections - keep tables intact if possible
        table_chunks = []
        for table_section in table_sections:
            if len(table_section) <= 1200:  # Keep small tables intact
                table_chunks.append(f"TABLE SECTION:\n{table_section}")
            else:
                # Split large tables by logical breaks (header + rows)
                table_lines = table_section.split('\n')
                header_lines = []
                current_chunk_lines = []
                
                for line in table_lines:
                    if line.strip().startswith('-') and len(line.strip()) > 10:
                        # This is likely a separator, keep with header
                        if not current_chunk_lines:
                            header_lines.append(line)
                        else:
                            current_chunk_lines.append(line)
                            if len('\n'.join(header_lines + current_chunk_lines)) > 800:
                                table_chunks.append(f"TABLE SECTION:\n" + '\n'.join(header_lines + current_chunk_lines))
                                current_chunk_lines = []
                    else:
                        current_chunk_lines.append(line)
                        if len('\n'.join(header_lines + current_chunk_lines)) > 800:
                            table_chunks.append(f"TABLE SECTION:\n" + '\n'.join(header_lines + current_chunk_lines))
                            current_chunk_lines = []
                
                if current_chunk_lines:
                    table_chunks.append(f"TABLE SECTION:\n" + '\n'.join(header_lines + current_chunk_lines))
        
        # Combine all chunks
        all_chunks = regular_chunks + table_chunks
        
        logger.info(f"Created {len(all_chunks)} text chunks ({len(regular_chunks)} regular, {len(table_chunks)} table)")
        return all_chunks
        
    except Exception as e:
        logger.error(f"Error chunking text: {e}")
        # Fallback to simple chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(text)
        logger.info(f"Created {len(chunks)} text chunks (fallback)")
        return chunks

async def get_embeddings_async(texts: List[str], doc_hash: str = None) -> List[List[float]]:
    """Generate embeddings with parallel processing and optimized caching."""
    try:
        if embedding_model is None:
            raise Exception("Embedding model not initialized")
        
        # Check cache if doc_hash is provided
        if CACHE_ENABLED and doc_hash:
            cache_file = get_cache_file_path(doc_hash, "embeddings")
            if await is_cache_valid(cache_file):
                logger.info(f"Loading embeddings from cache: {doc_hash}")
                cached_data = await load_from_cache(cache_file)
                if cached_data and 'embeddings' in cached_data and len(cached_data['embeddings']) == len(texts):
                    return cached_data['embeddings']
        
        # Optimized parallel embedding generation
        optimal_batch_size = min(EMBEDDING_BATCH_SIZE, len(texts))
        all_embeddings = []
        
        def generate_batch_embeddings(batch_texts):
            """Generate embeddings for a batch of texts."""
            return embedding_model.encode(
                batch_texts, 
                convert_to_tensor=False, 
                show_progress_bar=False,
                batch_size=16,  # Internal batch size for sentence transformers
                normalize_embeddings=True  # Normalize for better similarity search
            )
        
        # Use ThreadPoolExecutor for embedding generation (ProcessPoolExecutor can't access global objects)
        max_workers = min(multiprocessing.cpu_count(), 4)  # Limit to prevent resource exhaustion
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create batches
            batches = [texts[i:i + optimal_batch_size] for i in range(0, len(texts), optimal_batch_size)]
            
            # Submit all batches for parallel processing
            future_to_batch = {
                executor.submit(generate_batch_embeddings, batch): i 
                for i, batch in enumerate(batches)
            }
            
            # Collect results in order
            batch_results = [None] * len(batches)
            
            for future in concurrent.futures.as_completed(future_to_batch):
                batch_index = future_to_batch[future]
                try:
                    embeddings = future.result()
                    batch_results[batch_index] = [embedding.tolist() for embedding in embeddings]
                    logger.info(f"Completed embedding batch {batch_index + 1}/{len(batches)}")
                except Exception as e:
                    logger.error(f"Error in embedding batch {batch_index}: {e}")
                    raise
            
            # Combine results in correct order
            for batch_embeddings in batch_results:
                if batch_embeddings:
                    all_embeddings.extend(batch_embeddings)
        
        logger.info(f"Generated embeddings for {len(texts)} texts using {max_workers} workers")
        
        # Cache the result if doc_hash is provided
        if CACHE_ENABLED and doc_hash:
            cache_data = {
                'doc_hash': doc_hash,
                'embeddings': all_embeddings,
                'text_count': len(texts),
                'generated_at': datetime.now().isoformat(),
                'model_info': 'all-MiniLM-L6-v2'
            }
            await save_to_cache(cache_file, cache_data)
        
        return all_embeddings
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate embeddings: {str(e)}")

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Synchronous wrapper for async embeddings generation."""
    return asyncio.run(get_embeddings_async(texts))

async def embed_and_store_async(chunks: List[str], document_id: str) -> None:
    """Generate embeddings and store in Pinecone with optimized async batch processing."""
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
            # Wait for index to be ready
            await asyncio.sleep(10)
        
        # Connect to index
        index = pc.Index(PINECONE_INDEX_NAME)
        
        # Check if document already exists in Pinecone
        doc_hash = get_document_hash(document_id)
        test_vector_id = f"{document_id}-chunk-0"
        
        try:
            fetch_response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: index.fetch(ids=[test_vector_id])
            )
            if fetch_response['vectors']:
                logger.info("Document already embedded in Pinecone, skipping embedding step")
                return
        except Exception:
            logger.info("Document not found in Pinecone, proceeding with embedding")
        
        # Generate embeddings with parallel processing
        embeddings = await get_embeddings_async(chunks, doc_hash)
        
        # Prepare data for optimized batch upsert
        vectors_to_upsert = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vector_id = f"{document_id}-chunk-{i}"
            metadata = {
                'text': chunk[:1000],  # Limit metadata size
                'doc_url': document_id,
                'doc_hash': doc_hash,
                'chunk_index': i,
                'created_at': datetime.now().isoformat()
            }
            vectors_to_upsert.append((vector_id, embedding, metadata))
        
        # Optimized batch upsert with smart concurrency
        optimal_batch_size = min(BATCH_SIZE, 50)  # Smaller batches for stability
        total_batches = (len(vectors_to_upsert) - 1) // optimal_batch_size + 1
        logger.info(f"Upserting {len(vectors_to_upsert)} vectors in {total_batches} batches")
        
        # Create semaphore with reduced concurrency for Pinecone stability
        semaphore = asyncio.Semaphore(min(MAX_CONCURRENT_REQUESTS, 2))
        
        async def upsert_batch_optimized(batch_data, batch_num):
            async with semaphore:
                try:
                    # Exponential backoff delay to prevent rate limiting
                    delay = min(0.1 * (2 ** (batch_num % 4)), 2.0)
                    await asyncio.sleep(delay)
                    
                    # Upsert with retry logic
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            await asyncio.get_event_loop().run_in_executor(
                                None, 
                                lambda: index.upsert(vectors=batch_data)
                            )
                            logger.info(f"✓ Batch {batch_num + 1}/{total_batches} completed")
                            break
                        except Exception as e:
                            if attempt < max_retries - 1:
                                wait_time = 2 ** attempt
                                logger.warning(f"Batch {batch_num + 1} failed, retrying in {wait_time}s: {e}")
                                await asyncio.sleep(wait_time)
                            else:
                                raise
                                
                except Exception as e:
                    logger.error(f"Error in batch {batch_num + 1}: {e}")
                    raise
        
        # Process batches with optimized concurrency
        tasks = []
        for i in range(0, len(vectors_to_upsert), optimal_batch_size):
            batch = vectors_to_upsert[i:i + optimal_batch_size]
            batch_num = i // optimal_batch_size
            task = upsert_batch_optimized(batch, batch_num)
            tasks.append(task)
        
        # Wait for all batches to complete
        await asyncio.gather(*tasks)
        
        logger.info(f"✅ Successfully stored {len(chunks)} chunks in Pinecone with optimized batching")
        
    except Exception as e:
        logger.error(f"Error storing in Pinecone: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to store embeddings: {str(e)}")

def embed_and_store(chunks: List[str], document_id: str):
    """Synchronous wrapper for async embed and store."""
    return asyncio.run(embed_and_store_async(chunks, document_id))

async def retrieve_relevant_chunks_async(question: str, top_k: int = None, document_id: str = None) -> str:
    """Retrieve relevant text chunks with optimized async processing and better relevance filtering."""
    try:
        # Use configured top_k or parameter (increased to 20 for maximum accuracy and comprehensive coverage)
        actual_top_k = top_k if top_k is not None else 20  # Further increased from 15
        
        # Check if Pinecone client is initialized
        if pc is None:
            raise Exception("Pinecone client not initialized")
        
        # Connect to index
        index = pc.Index(PINECONE_INDEX_NAME)
        
        # Generate embedding for the question with optimized processing
        question_embedding = await get_embeddings_async([question])
        question_vector = question_embedding[0]
        
        # Prepare metadata filter to prevent context contamination
        metadata_filter = None
        if document_id:
            doc_hash = get_document_hash(document_id)
            metadata_filter = {"doc_hash": {"$eq": doc_hash}}
            logger.info(f"Using metadata filter for document: {doc_hash}")
        
        # Query Pinecone with optimized parameters and metadata filtering
        query_response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: index.query(
                vector=question_vector,
                top_k=actual_top_k,
                include_metadata=True,
                include_values=False,  # Don't return vectors to save bandwidth
                filter=metadata_filter  # Prevent context contamination between documents
            )
        )
        
        # Extract and rank relevant text chunks with improved filtering
        relevant_chunks = []
        for match in query_response['matches']:
            if 'metadata' in match and 'text' in match['metadata']:
                chunk_text = match['metadata']['text']
                score = match['score']
                # Lowered threshold for better coverage of policy information
                if score > 0.25:  # Even more inclusive threshold for comprehensive coverage
                    # Enhanced formatting for table data
                    if '|' in chunk_text and '-' in chunk_text:
                        # This looks like table data, format it better
                        relevant_chunks.append(f"[Table Data - Relevance: {score:.3f}]\n{chunk_text}")
                    else:
                        relevant_chunks.append(f"[Relevance: {score:.3f}] {chunk_text}")
        
        if not relevant_chunks:
            # If no chunks meet the threshold, take the top 8 anyway for policy questions
            logger.warning(f"No chunks above threshold, using top matches")
            for match in query_response['matches'][:8]:  # Increased from 5
                if 'metadata' in match and 'text' in match['metadata']:
                    chunk_text = match['metadata']['text']
                    score = match['score']
                    relevant_chunks.append(f"[Low Relevance: {score:.3f}] {chunk_text}")
        
        if not relevant_chunks:
            return "No relevant information found in the document."
        
        context = "\n\n".join(relevant_chunks)
        logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks for question")
        return context
        
    except Exception as e:
        logger.error(f"Error retrieving chunks: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve relevant information: {str(e)}")

def retrieve_relevant_chunks(question: str, top_k: int = 20, document_id: str = None) -> str:
    """Synchronous wrapper for async retrieval."""
    return asyncio.run(retrieve_relevant_chunks_async(question, top_k, document_id))

async def call_local_llm_async(prompt: str) -> str:
    """Call LM Studio (OpenAI-compatible) endpoint asynchronously."""
    try:
        # LM Studio uses OpenAI-compatible API format
        payload = {
            "model": LOCAL_LLM_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": 1000,
            "stream": False
        }
        
        headers = {
            "Content-Type": "application/json",
        }
        
        session = await get_http_session()
        async with session.post(
            LOCAL_LLM_ENDPOINT,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=LOCAL_LLM_TIMEOUT),
            headers=headers
        ) as response:
            
            if response.status == 200:
                result = await response.json()
                # Extract response from OpenAI-compatible format
                if 'choices' in result and len(result['choices']) > 0:
                    content = result['choices'][0]['message']['content']
                    return content.strip()
                else:
                    logger.error("Invalid response format from LM Studio")
                    return None
            else:
                error_text = await response.text()
                logger.error(f"LM Studio returned status {response.status}: {error_text}")
                return None
        
    except Exception as e:
        logger.error(f"Error calling LM Studio: {e}")
        return None

async def call_together_ai_async(prompt: str) -> str:
    """Call Together AI endpoint asynchronously."""
    try:
        if not TOGETHER_API_KEY:
            raise Exception("Together AI API key not configured")
        
        payload = {
            "model": TOGETHER_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 0.7,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "stop": ["</s>", "[INST]", "[/INST]"]
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {TOGETHER_API_KEY}"
        }
        
        session = await get_http_session()
        async with session.post(
            "https://api.together.xyz/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=TOGETHER_TIMEOUT),
            headers=headers
        ) as response:
            
            if response.status == 200:
                result = await response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    content = result['choices'][0]['message']['content']
                    return content.strip()
                else:
                    logger.error("Invalid response format from Together AI")
                    return None
            else:
                error_text = await response.text()
                logger.error(f"Together AI returned status {response.status}: {error_text}")
                return None
        
    except Exception as e:
        logger.error(f"Error calling Together AI: {e}")
        return None

def call_local_llm(prompt: str) -> str:
    """Synchronous wrapper for async LM Studio call."""
    return asyncio.run(call_local_llm_async(prompt))

async def generate_answer_async(context: str, question: str) -> str:
    """Generate an answer using the selected LLM with enhanced prompting for policy/insurance questions."""
    try:
        # Enhanced prompt for insurance/policy documents
        prompt = f"""Answer policy questions directly and briefly. When asked about specific plans, use the general policy terms if plan-specific details aren't available. Start with "Yes" or "No" for yes/no questions.

Context: {context}

Question: {question}

Answer briefly in 1 sentence with key details only (amounts, conditions). Don't over-explain:"""

        if MODEL_TYPE == 'gemini' and gemini_model is not None:
            # Use Gemini (run in executor as it's not async)
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: gemini_model.generate_content(prompt)
            )
            if response.text:
                answer = response.text.strip()
            else:
                answer = "I was unable to generate an answer based on the provided context."
                
        elif MODEL_TYPE == 'local' and local_llm_available:
            # Use local LLM (LM Studio) asynchronously
            answer = await call_local_llm_async(prompt)
            if answer is None:
                answer = "I was unable to generate an answer using LM Studio."
                
        elif MODEL_TYPE == 'together' and TOGETHER_API_KEY:
            # Use Together AI asynchronously
            answer = await call_together_ai_async(prompt)
            if answer is None:
                answer = "I was unable to generate an answer using Together AI."
        else:
            # Fallback
            if gemini_model is not None:
                logger.info("Using Gemini as fallback")
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: gemini_model.generate_content(prompt)
                )
                if response.text:
                    answer = response.text.strip()
                else:
                    answer = "I was unable to generate an answer based on the provided context."
            else:
                raise Exception("No LLM model available")
        
        logger.info(f"Generated answer using {MODEL_TYPE} model for question: {question[:50]}...")
        return answer
        
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {str(e)}")

def generate_answer(context: str, question: str) -> str:
    """Synchronous wrapper for async answer generation."""
    return asyncio.run(generate_answer_async(context, question))

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with basic information."""
    return {
        "message": f"LLM Document Processing System ({MODEL_TYPE.title()})",
        "version": "1.0.0",
        "model_type": MODEL_TYPE,
        "endpoints": {
            "main": "/hackrx/run",
            "docs": "/docs",
            "health": "/health"
        }
    }

@app.post("/hackrx/run", response_model=QueryResponse, dependencies=[Depends(verify_token)])
async def run_query(request: QueryRequest, background_tasks: BackgroundTasks):
    """Optimized main endpoint with smart caching, parallel processing, and connection pooling."""
    try:
        logger.info(f"🚀 Processing request with {len(request.questions)} questions")
        
        # Step 1: Smart cache check for all components
        url = str(request.documents)
        text_cached, cached_text, cached_chunks, doc_hash = await smart_cache_check(url)
        
        # Step 2: Get document text (from cache or download with connection pooling)
        if text_cached and cached_text:
            logger.info("📦 Using cached document text")
            full_text = cached_text
        else:
            logger.info("⬇️ Downloading document with optimized connection pooling")
            full_text = await process_pdf_from_url_async(url)
        
        if not full_text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the PDF")
        
        # Step 3: Get text chunks (from cache or generate)
        if cached_chunks:
            logger.info("📦 Using cached text chunks")
            chunks = cached_chunks
        else:
            logger.info("✂️ Creating text chunks")
            chunks = chunk_text(full_text)
            # Cache chunks for future use
            await cache_chunks(doc_hash, chunks)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No text chunks could be created")
        
        # Step 4: Generate embeddings and store (with smart duplicate detection)
        document_id = url.replace("://", "_").replace("/", "_")
        logger.info("🧠 Processing embeddings with parallel generation")
        await embed_and_store_async(chunks, document_id)
        
        # Step 5: Process questions with optimized concurrency
        async def process_question_optimized(question: str, index: int) -> str:
            try:
                logger.info(f"❓ Processing question {index + 1}/{len(request.questions)}")
                
                # Use optimized retrieval with relevance filtering and metadata filtering to prevent context contamination
                context = await retrieve_relevant_chunks_async(question, top_k=RETRIEVAL_TOP_K, document_id=url)
                
                if not context.strip() or "No highly relevant information" in context:
                    return "No relevant information found in the document for this question."
                
                # Generate answer with the selected model
                answer = await generate_answer_async(context, question)
                logger.info(f"✅ Completed question {index + 1}")
                return answer
                
            except Exception as e:
                logger.error(f"❌ Error processing question {index + 1}: {e}")
                return f"Error processing this question: {str(e)}"
        
        # Optimized concurrency control
        optimal_concurrency = min(MAX_CONCURRENT_REQUESTS, len(request.questions), 3)
        semaphore = asyncio.Semaphore(optimal_concurrency)
        
        async def limited_process_question(question: str, index: int) -> str:
            async with semaphore:
                return await process_question_optimized(question, index)
        
        # Process questions with controlled concurrency
        logger.info(f"🔄 Processing {len(request.questions)} questions with {optimal_concurrency} concurrent workers")
        tasks = [
            limited_process_question(question, i) 
            for i, question in enumerate(request.questions)
        ]
        
        final_answers = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions in the results
        processed_answers = []
        for i, answer in enumerate(final_answers):
            if isinstance(answer, Exception):
                logger.error(f"❌ Exception in question {i + 1}: {answer}")
                processed_answers.append(f"Error processing question {i + 1}: {str(answer)}")
            else:
                processed_answers.append(answer)
        
        logger.info(f"🎉 Successfully processed all {len(request.questions)} questions")
        return QueryResponse(answers=processed_answers)
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"💥 Unexpected error processing request: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Add cleanup for connection pooling
@app.on_event("shutdown")
async def cleanup():
    """Cleanup resources on shutdown."""
    global global_session
    if global_session and not global_session.closed:
        await global_session.close()
        logger.info("🧹 Cleaned up HTTP session")

@app.get("/health")
async def health_check():
    """Health check endpoint with performance information."""
    cache_stats = {}
    if CACHE_ENABLED:
        try:
            # Get cache directory stats
            doc_cache_files = len([f for f in os.listdir(f"{CACHE_DIR}/documents") if f.endswith('.json')])
            emb_cache_files = len([f for f in os.listdir(f"{CACHE_DIR}/embeddings") if f.endswith('.json')])
            cache_stats = {
                "cache_enabled": True,
                "cache_directory": CACHE_DIR,
                "cache_ttl_hours": CACHE_TTL_HOURS,
                "cached_documents": doc_cache_files,
                "cached_embeddings": emb_cache_files
            }
        except Exception as e:
            cache_stats = {"cache_enabled": True, "cache_error": str(e)}
    else:
        cache_stats = {"cache_enabled": False}
    
    return {
        "status": "healthy",
        "model_type": MODEL_TYPE,
        "gemini_configured": bool(GEMINI_API_KEY),
        "gemini_available": gemini_model is not None,
        "local_llm_configured": bool(LOCAL_LLM_ENDPOINT),
        "local_llm_available": local_llm_available,
        "lm_studio_endpoint": LOCAL_LLM_ENDPOINT if MODEL_TYPE == 'local' else None,
        "lm_studio_model": LOCAL_LLM_MODEL if MODEL_TYPE == 'local' else None,
        "together_configured": bool(TOGETHER_API_KEY),
        "together_model": TOGETHER_MODEL if MODEL_TYPE == 'together' else None,
        "pinecone_configured": bool(PINECONE_API_KEY),
        "embedding_model_loaded": embedding_model is not None,
        "performance_config": {
            "batch_size": BATCH_SIZE,
            "embedding_batch_size": EMBEDDING_BATCH_SIZE,
            "max_concurrent_requests": MAX_CONCURRENT_REQUESTS,
            "retrieval_top_k": RETRIEVAL_TOP_K,
            "connection_pool_size": CONNECTION_POOL_SIZE,
            "connection_pool_per_host": CONNECTION_POOL_PER_HOST,
            "parallel_workers": min(multiprocessing.cpu_count(), 4)
        },
        "cache": cache_stats
    }

if __name__ == "__main__":
    import uvicorn
    
    # Parse arguments again for the main execution
    args = parse_args()
    
    print(f"\n🚀 Starting LLM Document Processing System")
    print(f"📋 Model: {args.model.upper()}")
    
    if args.model == 'gemini':
        print(f"🤖 Using Google Gemini (gemini-2.5-flash-lite)")
        if not GEMINI_API_KEY:
            print("⚠️  Warning: GEMINI_API_KEY not found in environment")
    elif args.model == 'local':
        print(f"🏠 Using LM Studio at: {LOCAL_LLM_ENDPOINT}")
        print(f"📦 Model: {LOCAL_LLM_MODEL}")
        print(f"⏱️  Timeout: {LOCAL_LLM_TIMEOUT}s")
        print("📋 Make sure LM Studio is running with a model loaded")
    elif args.model == 'together':
        print(f"🌐 Using Together AI")
        print(f"📦 Model: {TOGETHER_MODEL}")
        print(f"⏱️  Timeout: {TOGETHER_TIMEOUT}s")
        if not TOGETHER_API_KEY:
            print("⚠️  Warning: TOGETHER_API_KEY not found in environment")
    
    print(f"🌐 Server will be available at: http://0.0.0.0:8000")
    print(f"📖 API docs at: http://0.0.0.0:8000/docs")
    print(f"❤️  Health check at: http://0.0.0.0:8000/health\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
