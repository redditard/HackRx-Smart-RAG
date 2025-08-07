#!/usr/bin/env python3
"""
Startup script for LLM Document Processing System
Usage:
    python run_server.py --model gemini     # Use Google Gemini
    python run_server.py --model local      # Use Local LLM (e.g., LM Studio)
    python run_server.py --model together   # Use Together AI
"""

import argparse
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description='LLM Document Processing System Launcher')
    parser.add_argument('--model', choices=['gemini', 'local', 'together'], 
                       default=os.getenv('DEFAULT_MODEL_TYPE', 'gemini'),
                       help='Choose the LLM model: gemini, local, or together (default: gemini)')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to (default: 8000)')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload for development')
    
    args = parser.parse_args()
    
    # Set the model type in environment for the main module
    os.environ['DEFAULT_MODEL_TYPE'] = args.model
    
    print("=" * 60)
    print("🚀 LLM Document Processing System")
    print("=" * 60)
    print(f"📋 Model Type: {args.model.upper()}")
    
    if args.model == 'gemini':
        gemini_key = os.getenv('GEMINI_API_KEY')
        if gemini_key:
            print(f"🤖 Google Gemini: ✅ API Key configured")
        else:
            print(f"🤖 Google Gemini: ❌ GEMINI_API_KEY missing")
            print("   Please set your Gemini API key in the .env file")
    
    elif args.model == 'local':
        local_endpoint = os.getenv('LOCAL_LLM_ENDPOINT', 'http://localhost:1234/v1/chat/completions')
        local_model = os.getenv('LOCAL_LLM_MODEL', 'lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF')
        print(f"🏠 LM Studio Endpoint: {local_endpoint}")
        print(f"📦 LM Studio Model: {local_model}")
        print("   Make sure LM Studio is running with a model loaded")
    
    elif args.model == 'together':
        together_key = os.getenv('TOGETHER_API_KEY')
        together_model = os.getenv('TOGETHER_MODEL', 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo')
        if together_key:
            print(f"🌐 Together AI: ✅ API Key configured")
        else:
            print(f"🌐 Together AI: ❌ TOGETHER_API_KEY missing")
            print("   Please set your Together AI API key in the .env file")
        print(f"📦 Together AI Model: {together_model}")
    
    # Check Pinecone configuration
    pinecone_key = os.getenv('PINECONE_API_KEY')
    if pinecone_key:
        print(f"🌲 Pinecone: ✅ API Key configured")
    else:
        print(f"🌲 Pinecone: ❌ PINECONE_API_KEY missing")
        print("   Vector storage functionality will be limited")
    
    print(f"🌐 Server: http://{args.host}:{args.port}")
    print(f"📖 API Docs: http://{args.host}:{args.port}/docs")
    print(f"❤️  Health Check: http://{args.host}:{args.port}/health")
    print("=" * 60)
    
    # Import and run the server
    try:
        import uvicorn
        from main import app
        
        uvicorn.run(
            "main:app", 
            host=args.host, 
            port=args.port, 
            reload=args.reload,
            log_level="info"
        )
    except ImportError as e:
        print(f"❌ Error importing dependencies: {e}")
        print("Make sure all required packages are installed:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
