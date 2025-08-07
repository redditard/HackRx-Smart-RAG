# 🚀 LLM Document Processing System - Complete Feature Guide

## 🌟 System Overview

This intelligent query-retrieval system now supports **three different LLM providers**:
- **🤖 Google Gemini** - Fast cloud-based AI with excellent quality
- **🏠 Local LLM** - Complete privacy with LM Studio integration  
- **🌐 Together AI** - Scalable cloud inference with open-source models

## 🔧 Key Features

### 📄 Document Processing
- **Async PDF Processing**: Lightning-fast document ingestion with concurrent processing
- **Smart Text Chunking**: Semantic chunking using LangChain splitters
- **Document Caching**: Hash-based caching system prevents reprocessing
- **Batch Embeddings**: Processes multiple documents simultaneously

### 🔍 Vector Search & Retrieval
- **Pinecone Integration**: Professional-grade vector database
- **Batch Upserts**: Optimized bulk operations for better performance
- **SentenceTransformers**: Free, high-quality embeddings (all-MiniLM-L6-v2)
- **Async Operations**: Non-blocking vector operations

### 🤖 Multi-Model LLM Support

#### Google Gemini (gemini-1.5-flash)
```bash
python main.py --model gemini
```
- ✅ Fast responses (typically 1-3 seconds)
- ✅ High quality natural language generation
- ✅ Built-in safety filters
- ✅ Reliable cloud infrastructure

#### Local LLM (LM Studio)
```bash
python main.py --model local
```
- ✅ Complete privacy - no data leaves your machine
- ✅ No API costs after initial setup
- ✅ Customizable models and parameters
- ✅ OpenAI-compatible API

#### Together AI
```bash
python main.py --model together
```
- ✅ Access to cutting-edge open-source models
- ✅ Competitive pricing (often cheaper than OpenAI)
- ✅ Fast inference infrastructure
- ✅ Multiple model options (Llama, Mistral, etc.)

### ⚡ Performance Optimizations

#### Async Architecture
- **Non-blocking I/O**: All file operations are asynchronous
- **Concurrent Processing**: Multiple documents processed simultaneously
- **Background Tasks**: Heavy operations don't block API responses
- **Connection Pooling**: Efficient HTTP client management

#### Intelligent Caching
- **Document Cache**: Processed PDFs stored locally with hash verification
- **Embedding Cache**: Vector embeddings cached to avoid recomputation
- **TTL Management**: Configurable cache expiration (default: 24 hours)
- **Storage Optimization**: Efficient JSON-based storage

#### Batch Operations
- **Pinecone Batching**: Up to 100 vectors per batch operation
- **Concurrent Limits**: Configurable concurrent request limits
- **Memory Management**: Chunked processing for large documents

## 🎯 API Endpoints

### Core Functionality
- `POST /query` - Intelligent query with RAG
- `POST /upload` - Document upload and processing  
- `GET /health` - System status and configuration
- `GET /` - API documentation and endpoints

### Advanced Features
- **Background Processing**: Large documents processed asynchronously
- **Progress Tracking**: Real-time status updates for long operations
- **Error Handling**: Comprehensive error responses with details
- **Authentication**: Bearer token security

## 🔧 Configuration Options

### Environment Variables
```bash
# Model Selection
DEFAULT_MODEL_TYPE=gemini  # gemini, local, or together

# Google Gemini
GEMINI_API_KEY=your_key_here

# Local LLM (LM Studio)
LOCAL_LLM_ENDPOINT=http://localhost:1234/v1/chat/completions
LOCAL_LLM_MODEL=your_model_name
LOCAL_LLM_TIMEOUT=120

# Together AI
TOGETHER_API_KEY=your_key_here
TOGETHER_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo
TOGETHER_TIMEOUT=120

# Vector Database
PINECONE_API_KEY=your_key_here
PINECONE_ENVIRONMENT=us-east-1-aws
PINECONE_INDEX_NAME=hackrx-documents

# Performance Tuning
PINECONE_BATCH_SIZE=100
MAX_CONCURRENT_REQUESTS=5
CACHE_ENABLED=true
CACHE_TTL_HOURS=24
```

### Command Line Options
```bash
# Start with specific model
python main.py --model gemini
python main.py --model local  
python main.py --model together

# Using the startup script
python run_server.py --model together --host 0.0.0.0 --port 8080 --reload
```

## 📊 Performance Benchmarks

### Document Processing Speed
- **Small PDFs** (1-5 pages): ~2-3 seconds
- **Medium PDFs** (10-50 pages): ~5-10 seconds  
- **Large PDFs** (100+ pages): ~15-30 seconds

### Query Response Times
- **Gemini**: 1-3 seconds average
- **Local LLM**: 3-10 seconds (depends on hardware)
- **Together AI**: 2-5 seconds average

### Memory Usage
- **Base System**: ~200MB RAM
- **With Embeddings**: ~500MB RAM
- **Processing Large PDF**: ~1GB RAM peak

## 🛡️ Security Features

- **Bearer Token Authentication**: Secure API access
- **Input Validation**: Comprehensive request validation
- **Error Sanitization**: Safe error messages without data leaks
- **Rate Limiting**: Configurable request limits
- **Local Processing Option**: Complete data privacy with local models

## 🔍 Monitoring & Health Checks

The `/health` endpoint provides comprehensive system status:

```json
{
  "status": "healthy",
  "model_type": "together",
  "gemini_configured": true,
  "local_llm_configured": true,
  "together_configured": true,
  "embedding_model_loaded": true,
  "pinecone_configured": true,
  "performance_config": {
    "batch_size": 100,
    "max_concurrent_requests": 5
  },
  "cache": {
    "cache_enabled": true,
    "cached_documents": 15,
    "cached_embeddings": 23
  }
}
```

## 🚀 Getting Started

1. **Choose Your Model**: Decide between Gemini, Local LLM, or Together AI
2. **Set Environment Variables**: Configure your chosen provider's API keys
3. **Install Dependencies**: `pip install -r requirements.txt`
4. **Start the Server**: `python main.py --model [your_choice]`
5. **Test the API**: Visit `http://localhost:8000/docs`

## 📚 Documentation

- **[LM Studio Setup Guide](LM_STUDIO_GUIDE.md)**: Complete local LLM setup
- **[Together AI Setup Guide](TOGETHER_AI_GUIDE.md)**: Together AI configuration
- **[API Documentation](http://localhost:8000/docs)**: Interactive API docs
- **[Performance Tuning](README.md)**: Optimization best practices

## 🎉 What's New in Latest Version

- ✨ **Together AI Support**: Third LLM provider option
- ⚡ **Full Async Architecture**: All operations are now non-blocking
- 🚀 **Document Caching**: Smart caching prevents reprocessing
- 📦 **Batch Operations**: Optimized Pinecone batch upserts
- 🔧 **Enhanced Configuration**: Comprehensive environment variable support
- 📊 **Better Monitoring**: Detailed health checks and status reporting
- 🛡️ **Improved Security**: Enhanced authentication and validation

---

**Ready to process documents intelligently? Choose your LLM and start building! 🚀**
