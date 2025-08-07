#!/usr/bin/env python3
"""
Test script to verify Together AI integration without running the full server.
This checks if the Together AI client can be imported and basic configuration works.
"""

import os
import sys
import subprocess

def check_together_client():
    """Check if together client can be imported"""
    try:
        import together
        print("✓ Together AI client library found")
        return True
    except ImportError:
        print("✗ Together AI client library not found")
        print("  Install with: pip install together==1.2.12")
        return False

def check_env_vars():
    """Check if environment variables are set"""
    together_key = os.getenv('TOGETHER_API_KEY')
    together_model = os.getenv('TOGETHER_MODEL', 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo')
    
    print(f"Together API Key: {'✓ Set' if together_key else '✗ Not set'}")
    print(f"Together Model: {together_model}")
    
    return bool(together_key)

def test_basic_functionality():
    """Test basic Together AI functionality"""
    if not check_together_client():
        return False
    
    if not check_env_vars():
        print("\n⚠️  Set TOGETHER_API_KEY in your .env file to test API calls")
        return False
    
    try:
        import together
        client = together.Together(api_key=os.getenv('TOGETHER_API_KEY'))
        
        # Test a simple completion
        response = client.chat.completions.create(
            model=os.getenv('TOGETHER_MODEL', 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo'),
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello in exactly 3 words."}
            ],
            max_tokens=10,
            temperature=0.7
        )
        
        print("✓ Together AI API test successful")
        print(f"  Response: {response.choices[0].message.content.strip()}")
        return True
        
    except Exception as e:
        print(f"✗ Together AI API test failed: {e}")
        return False

def main():
    print("Together AI Integration Test")
    print("=" * 30)
    
    # Check if we can import the main modules
    try:
        from main import IntelligentRetrievalSystem
        print("✓ Main application imports successful")
    except ImportError as e:
        print(f"✗ Main application import failed: {e}")
        return
    
    # Test Together AI specific functionality
    print("\nTesting Together AI components:")
    success = test_basic_functionality()
    
    if success:
        print("\n🎉 Together AI integration test passed!")
        print("You can now run: python main.py --model together")
    else:
        print("\n⚠️  Together AI integration test had issues")
        print("Check the error messages above and install missing dependencies")

if __name__ == "__main__":
    main()
