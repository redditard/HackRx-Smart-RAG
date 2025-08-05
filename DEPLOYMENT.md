# Deployment Guide (Gemini Version)

## Quick Start

### 1. Local Development Setup

```bash
# Clone/navigate to the project
cd intelligent_retrieval_system

# Run the setup script
./setup.sh

# Or manually:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your actual API keys
```

### 2. Required API Keys

You need to obtain the following API keys:

#### Google AI (Gemini) API Key
1. Go to https://makersuite.google.com/app/apikey
2. Create a new API key
3. Add it to your `.env` file as `GEMINI_API_KEY`

#### Pinecone API Key
1. Go to https://app.pinecone.io/
2. Create a new project
3. Get your API key and environment
4. Add them to your `.env` file:
   - `PINECONE_API_KEY`
   - `PINECONE_ENVIRONMENT`

### 3. Running the Application

```bash
# Activate virtual environment
source venv/bin/activate

# Start the server
uvicorn main:app --reload

# Or for production
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 4. Testing the API

```bash
# Run tests
python test_main.py

# Or manual testing
curl -X POST "http://localhost:8000/hackrx/run" \
  -H "Authorization: Bearer 0Y988a27dc0bf2e755e893e2665069fe4d09189215f7824b023cc07db597bb" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://example.com/document.pdf",
    "questions": ["What is this about?"]
  }'
```

## Docker Deployment

### Build and Run with Docker

```bash
# Build the image
docker build -t llm-doc-processor .

# Run the container
docker run -p 8000:8000 --env-file .env llm-doc-processor

# Or use docker-compose
docker-compose up -d
```

## Production Deployment

### Environment Variables

Set these environment variables in your production environment:

```bash
GEMINI_API_KEY=your_gemini_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENVIRONMENT=your_pinecone_env
PINECONE_INDEX_NAME=your_index_name
```

### Security Considerations

1. **API Keys**: Store API keys securely using environment variables or secret management systems
2. **Authentication**: The current implementation uses a hardcoded token. In production, implement proper authentication
3. **Rate Limiting**: Consider adding rate limiting to prevent abuse
4. **HTTPS**: Always use HTTPS in production
5. **Input Validation**: The system validates inputs, but consider additional security measures

### Performance Optimization

1. **Caching**: Implement caching for frequently accessed documents
2. **Database**: For production, consider using a persistent database for metadata
3. **Load Balancing**: Use a load balancer for high availability
4. **Monitoring**: Implement proper logging and monitoring

### Scaling

1. **Horizontal Scaling**: The application is stateless and can be scaled horizontally
2. **Vector Database**: Pinecone handles scaling automatically
3. **Async Processing**: Consider implementing async processing for large documents

## Troubleshooting

### Common Issues

1. **API Key Errors**
   - Ensure API keys are correctly set in `.env`
   - Check API key validity and quota

2. **PDF Processing Errors**
   - Ensure the PDF URL is accessible
   - Check PDF format and size limits

3. **Pinecone Connection Issues**
   - Verify Pinecone environment and API key
   - Check network connectivity

4. **Memory Issues**
   - Large PDFs may consume significant memory
   - Consider implementing chunked processing

### Monitoring

The application provides these endpoints for monitoring:

- `/health` - Health check endpoint
- `/` - Basic information endpoint
- Logs are written to stdout/stderr for container environments

### Performance Metrics

Key metrics to monitor:
- Response time per request
- Token usage (OpenAI API costs)
- Vector operations (Pinecone costs)
- Memory usage
- Error rates
