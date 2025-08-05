# Configuration settings for the LLM Document Processing System (Gemini)

# Gemini Settings
GEMINI_MODEL = "gemini-1.5-flash"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # SentenceTransformers model

# Text Processing Settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Retrieval Settings
TOP_K_CHUNKS = 5
MIN_RELEVANCE_SCORE = 0.7

# Pinecone Settings
VECTOR_DIMENSION = 384  # all-MiniLM-L6-v2 dimension
SIMILARITY_METRIC = "cosine"

# API Settings
REQUEST_TIMEOUT = 30
MAX_PDF_SIZE_MB = 50

# Prompt Templates
SYSTEM_PROMPT = """You are a helpful assistant for answering questions based on a provided document.
Your task is to answer the user's question accurately and concisely based ONLY on the context below.
Do not use any external knowledge. If the information is not present in the context, you must state "The provided document does not contain information on this topic."

Be specific and reference relevant details from the document when possible."""

ANSWER_PROMPT_TEMPLATE = """CONTEXT:
---
{context}
---

QUESTION: {question}

ANSWER:"""
