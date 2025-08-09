# 🚀 Deployment Guide: Using GitHub Secrets with Your Agentic RAG App

## Overview

Your GitHub secrets (`GROQ_API_KEY`, `SERPER_API_KEY`, `GEMINI_API_KEY`) are stored securely in GitHub, but they need to be properly exposed to your deployed application. Here's how to do it for different deployment platforms.

## 🔑 How GitHub Secrets Work

**Important**: GitHub secrets are **only available during GitHub Actions workflows**, not directly in your deployed application. You need to pass them as environment variables to your deployment platform.

## Deployment Options

### 1. 🌐 Streamlit Cloud (Recommended)

**Step 1**: Deploy your app to Streamlit Cloud
- Go to [share.streamlit.io](https://share.streamlit.io)
- Connect your GitHub repository
- Deploy the app

**Step 2**: Configure secrets in Streamlit Cloud
- Go to your app's settings in Streamlit Cloud
- Navigate to the "Secrets" section
- Add your secrets in TOML format:

```toml
GROQ_API_KEY = "your-actual-groq-key-here"
SERPER_API_KEY = "your-actual-serper-key-here"
GEMINI_API_KEY = "your-actual-gemini-key-here"
```

**Step 3**: The app will automatically use `st.secrets` to access these values.

### 2. 🔄 GitHub Actions + Any Platform

If you want to use GitHub Actions to deploy to other platforms (Heroku, Railway, etc.):

**Step 1**: Create a deployment workflow (`.github/workflows/deploy.yml`)
```yaml
name: Deploy App
on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to Platform
      env:
        GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
        SERPER_API_KEY: ${{ secrets.SERPER_API_KEY }}
        GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
      run: |
        # Your deployment commands here
        # These environment variables will be available during deployment
```

### 3. 🐳 Docker Deployment

**Step 1**: Create a Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Environment variables will be passed at runtime
CMD ["streamlit", "run", "enhanced_streamlit_app.py"]
```

**Step 2**: Run with environment variables
```bash
docker run -e GROQ_API_KEY="your-key" -e SERPER_API_KEY="your-key" -e GEMINI_API_KEY="your-key" your-app
```

### 4. ☁️ Other Cloud Platforms

For platforms like Heroku, Railway, Render, etc.:

1. Deploy your code to the platform
2. In the platform's dashboard, add environment variables:
   - `GROQ_API_KEY` = your actual key
   - `SERPER_API_KEY` = your actual key  
   - `GEMINI_API_KEY` = your actual key

## 🔧 How the Code Works

The updated `get_api_keys()` function now supports multiple deployment scenarios:

```python
def get_api_keys():
    """Get API keys from Streamlit secrets or environment variables"""
    # Try Streamlit secrets first (for Streamlit Cloud)
    try:
        groq_key = st.secrets.get("GROQ_API_KEY", "")
        serper_key = st.secrets.get("SERPER_API_KEY", "")
        gemini_key = st.secrets.get("GEMINI_API_KEY", "")
        
        if groq_key and serper_key:
            return groq_key, serper_key, gemini_key
    except:
        pass
    
    # Fallback to environment variables (for other deployments)
    groq_key = os.getenv("GROQ_API_KEY", "")
    serper_key = os.getenv("SERPER_API_KEY", "")
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    
    return groq_key, serper_key, gemini_key
```

## ✅ Testing Your Deployment

1. **Local Testing**: Set environment variables locally
   ```bash
   export GROQ_API_KEY="your-key"
   export SERPER_API_KEY="your-key"
   export GEMINI_API_KEY="your-key"
   streamlit run enhanced_streamlit_app.py
   ```

2. **Deployment Testing**: Check the app's sidebar - it should show:
   - ✅ "API keys configured from environment" (success)
   - ❌ "API keys not found in environment variables" (needs configuration)

## 🔒 Security Best Practices

1. **Never commit API keys** to your repository
2. **Use different keys** for development and production
3. **Regularly rotate** your API keys
4. **Monitor usage** of your API keys
5. **Set up billing alerts** for your API providers

## 🆘 Troubleshooting

**Problem**: App shows "API keys not found"
- **Solution**: Verify keys are properly set in your deployment platform's environment variables or secrets

**Problem**: GitHub Actions can't access secrets
- **Solution**: Ensure secrets are added to your repository settings and properly referenced in workflow

**Problem**: Streamlit Cloud can't find secrets
- **Solution**: Check that secrets are added in TOML format in Streamlit Cloud's secrets section

## 📞 Need Help?

If you encounter issues:
1. Check your deployment platform's documentation for environment variables
2. Verify your GitHub secrets are properly configured
3. Test locally first with environment variables
4. Check the app's status display in the sidebar for diagnostic information
