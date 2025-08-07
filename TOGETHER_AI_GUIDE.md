# Together AI Setup Guide

This guide explains how to set up and use Together AI with the Intelligent Retrieval System.

## What is Together AI?

Together AI provides fast, scalable AI inference through their cloud API. They offer competitive pricing and support for many popular open-source models including Llama, Mistral, and more.

## Setup Instructions

### 1. Get Together AI API Key

1. Visit [Together AI](https://together.ai/)
2. Sign up for an account
3. Navigate to your API Keys section
4. Create a new API key
5. Copy the API key (starts with something like `together_abc123...`)

### 2. Configure Environment Variables

Add these variables to your `.env` file:

```bash
# Together AI Configuration
TOGETHER_API_KEY=your_together_api_key_here
TOGETHER_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo
TOGETHER_TIMEOUT=120
```

### 3. Available Models

Together AI supports many models. Here are some popular options:

**Llama Models:**
- `meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo` (recommended, fast)
- `meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo` (more capable, slower)
- `meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo` (supports vision)

**Mistral Models:**
- `mistralai/Mistral-7B-Instruct-v0.3`
- `mistralai/Mixtral-8x7B-Instruct-v0.1`

**Other Models:**
- `NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO`
- `teknium/OpenHermes-2.5-Mistral-7B`

You can find the complete list at [Together AI Models](https://docs.together.ai/docs/inference-models).

### 4. Run with Together AI

```bash
# Start the server with Together AI
python main.py --model together

# Or set it as default in your .env file:
DEFAULT_MODEL_TYPE=together
python main.py
```

## Usage Examples

### Basic Query
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main features of the system?",
    "context_limit": 5
  }'
```

### Upload and Query Document
```bash
# Upload a PDF
curl -X POST "http://localhost:8000/upload" \
  -F "file=@document.pdf"

# Query the uploaded document
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Summarize the key points from the document",
    "context_limit": 3
  }'
```

## Advantages of Together AI

- **Fast Inference**: Optimized for speed with their inference infrastructure
- **Cost Effective**: Competitive pricing compared to other cloud AI services
- **Multiple Models**: Access to many open-source models
- **No Local Setup**: No need to download or run models locally
- **Scalable**: Handles concurrent requests efficiently

## Troubleshooting

### API Key Issues
```bash
# Check if your API key is set correctly
python -c "import os; print('API Key found:', bool(os.getenv('TOGETHER_API_KEY')))"
```

### Model Not Found
If you get a model not found error, check the [Together AI Models page](https://docs.together.ai/docs/inference-models) for the correct model name.

### Rate Limits
Together AI has rate limits. If you hit them:
- Check your account limits in the Together AI dashboard
- Implement exponential backoff in your requests
- Consider upgrading your plan for higher limits

### Connection Issues
```bash
# Test Together AI API connectivity
curl -H "Authorization: Bearer $TOGETHER_API_KEY" \
  https://api.together.xyz/v1/models
```

## Performance Tips

1. **Choose the Right Model**: 
   - Use 8B models for faster responses
   - Use 70B models for more complex reasoning

2. **Optimize Context Length**:
   - Keep context under the model's limit
   - Use `context_limit` parameter to control retrieved chunks

3. **Monitor Usage**:
   - Check your Together AI dashboard for usage stats
   - Set up billing alerts to avoid unexpected charges

## Cost Comparison

Together AI typically offers competitive pricing:
- Input tokens: ~$0.0002 per 1K tokens
- Output tokens: ~$0.0002 per 1K tokens

Check the [Together AI Pricing](https://together.ai/pricing) page for current rates.

## Support

- Together AI Documentation: https://docs.together.ai/
- Together AI Discord: https://discord.gg/together
- System Issues: Check the health endpoint at http://localhost:8000/health
