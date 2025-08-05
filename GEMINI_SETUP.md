# Getting Google AI (Gemini) API Key

## Step-by-Step Guide

### 1. Visit Google AI Studio
Go to [https://makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)

### 2. Sign In
- Sign in with your Google account
- If you don't have one, create a Google account first

### 3. Create API Key
- Click on "Create API Key"
- Choose your project or create a new one
- Copy the generated API key

### 4. Add to Environment
Add the API key to your `.env` file:
```
GEMINI_API_KEY=your_actual_api_key_here
```

## Important Notes

### Free Tier
- Google AI Studio provides free access to Gemini models
- Free tier includes generous quotas for development and testing
- No credit card required for getting started

### Rate Limits
- Free tier has rate limits
- For production use, consider upgrading to paid tier
- Monitor your usage in the Google AI Studio dashboard

### Supported Models
- `gemini-1.5-flash`: Fast and efficient (recommended)
- `gemini-1.5-pro`: More capable but slower
- `gemini-1.0-pro`: Previous generation

### Best Practices
1. **Keep your API key secure**
   - Never commit API keys to version control
   - Use environment variables
   - Rotate keys regularly

2. **Monitor usage**
   - Check your quota in Google AI Studio
   - Set up alerts for high usage

3. **Error handling**
   - Handle rate limit errors gracefully
   - Implement retry logic with exponential backoff

## Troubleshooting

### Common Issues
1. **"API key not valid"**
   - Check if the key is correctly copied
   - Ensure no extra spaces or characters

2. **"Quota exceeded"**
   - Check your usage in Google AI Studio
   - Wait for quota reset or upgrade plan

3. **"Model not found"**
   - Ensure you're using a supported model name
   - Check Google AI Studio for available models

### Testing Your API Key
Run this simple test to verify your API key works:

```python
import google.generativeai as genai
import os

# Configure the API key
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# Test the model
model = genai.GenerativeModel('gemini-1.5-flash')
response = model.generate_content("Hello, world!")
print(response.text)
```

If this works, your API key is correctly configured!
