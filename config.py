# Configuration settings for the LLM Document Processing System (Gemini)

# Gemini Settings
GEMINI_MODEL = "gemini-2.5-flash-lite"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # SentenceTransformers model

# Text Processing Settings
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Retrieval Settings
TOP_K_CHUNKS = 20  # Maximum possible for comprehensive coverage
MIN_RELEVANCE_SCORE = 0.25  # Very inclusive for maximum information capture

# Pinecone Settings
VECTOR_DIMENSION = 384  # all-MiniLM-L6-v2 dimension
SIMILARITY_METRIC = "cosine"

# API Settings
REQUEST_TIMEOUT = 30
MAX_PDF_SIZE_MB = 50

# Enhanced Prompt Templates for Insurance/Policy Questions
SYSTEM_PROMPT = """Answer policy questions directly and briefly. When asked about specific plans, use the general policy terms if plan-specific details aren't available. Start with "Yes" or "No" for yes/no questions."""

ANSWER_PROMPT_TEMPLATE = """Context: {context}

Question: {question}

Answer briefly in 1 sentence with key details only (amounts, conditions). Don't over-explain:"""
