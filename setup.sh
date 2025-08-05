#!/bin/bash

# Setup script for LLM Document Processing System

echo "====================================="
echo "LLM Document Processing System Setup"
echo "====================================="
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

echo "✓ Python 3 found"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

echo "✓ Requirements installed"

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠ .env file not found. Please create one from .env.example and add your API keys."
    echo "  cp .env.example .env"
    echo "  Then edit .env with your actual API keys."
else
    echo "✓ .env file found"
fi

echo
echo "Setup complete! Next steps:"
echo "1. Make sure your .env file has valid API keys"
echo "2. Run the server: uvicorn main:app --reload"
echo "3. Access the API docs at: http://127.0.0.1:8000/docs"
echo "4. Test with: python test_api.py"
echo
echo "Don't forget to activate the virtual environment:"
echo "source venv/bin/activate"
