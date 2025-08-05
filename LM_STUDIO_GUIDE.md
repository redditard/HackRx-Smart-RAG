# LM Studio Integration Guide

## Setup Instructions for LM Studio

### 1. Install and Setup LM Studio

1. Download LM Studio from [https://lmstudio.ai/](https://lmstudio.ai/)
2. Install and launch LM Studio
3. Download a model (recommended: Llama 3 8B Instruct or similar)

### 2. Start LM Studio Server

1. In LM Studio, go to the "Local Server" tab
2. Load your preferred model
3. Start the server (default port: 1234)
4. Make sure the server is running at `http://localhost:1234`

### 3. Configure Your Environment

Update your `.env` file:
```bash
# LM Studio Configuration
LOCAL_LLM_ENDPOINT=http://localhost:1234/v1/chat/completions
LOCAL_LLM_MODEL=lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF
LOCAL_LLM_TIMEOUT=120
DEFAULT_MODEL_TYPE=local
```

### 4. Run the Application

```bash
# Start with LM Studio
python main.py --model local

# Or use the startup script
python run_server.py --model local
```

## Model Selection

The system supports switching between models:

```bash
# Use Google Gemini (cloud)
python main.py --model gemini

# Use LM Studio (local)
python main.py --model local
```

## Fallback Behavior

- If you select `local` but LM Studio isn't running, the system automatically falls back to Gemini
- This ensures your API remains available even if the local model is unavailable

## Health Check

Check the status of your configuration:
```bash
curl http://localhost:8000/health
```

This will show:
- Current model type
- LM Studio availability
- Gemini configuration status
- Pinecone status

## Benefits of LM Studio Integration

✅ **Privacy**: All processing happens locally  
✅ **No API Costs**: No charges for LLM usage  
✅ **Customizable**: Use any model supported by LM Studio  
✅ **Fast**: No network latency for LLM calls  
✅ **Reliable**: Automatic fallback to Gemini if needed  

## Troubleshooting

**Connection Refused Error:**
- Make sure LM Studio is running
- Check that the server is started in LM Studio
- Verify the correct port (default: 1234)

**Model Not Found:**
- Ensure a model is loaded in LM Studio
- Check the model name in your `.env` file
- Models list available at: `http://localhost:1234/v1/models`

**Slow Responses:**
- Increase `LOCAL_LLM_TIMEOUT` in `.env`
- Use a smaller/faster model
- Check your system resources (RAM, GPU)
