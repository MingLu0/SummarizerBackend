# 🚀 Hugging Face Spaces Deployment Guide

This guide will help you deploy your SummarizerApp to Hugging Face Spaces for **FREE**!

## 🎯 Why Hugging Face Spaces?

- ✅ **100% Free** - No credit card required
- ✅ **16GB RAM** - Perfect for Mistral 7B model
- ✅ **Docker Support** - Easy deployment
- ✅ **Auto HTTPS** - Secure connections
- ✅ **Built for AI** - Designed for ML/AI applications
- ✅ **GitHub Integration** - Automatic deployments

## 📋 Prerequisites

1. **Hugging Face Account** - Sign up at [huggingface.co](https://huggingface.co)
2. **GitHub Repository** - Your code should be on GitHub
3. **Docker Knowledge** - Basic understanding helpful but not required

## 🛠️ Step-by-Step Deployment

### Step 1: Create a New Space

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click **"Create new Space"**
3. Fill in the details:
   - **Space name**: `summarizer-app` (or your preferred name)
   - **License**: MIT
   - **SDK**: **Docker** (important!)
   - **Hardware**: CPU (free tier)
   - **Visibility**: Public or Private

### Step 2: Configure Your Repository

You need to make these changes to your GitHub repository:

#### A. Rename Files
```bash
# Rename the Hugging Face specific files
mv Dockerfile.hf Dockerfile
mv README_HF.md README.md
```

#### B. Update Dockerfile (if needed)
The `Dockerfile.hf` is already optimized for Hugging Face Spaces, but verify it uses:
- Port `7860` (required by HF Spaces)
- `mistral:7b` model (smaller, faster)
- Proper startup script

#### C. Push Changes to GitHub
```bash
git add .
git commit -m "Add Hugging Face Spaces configuration"
git push origin main
```

### Step 3: Connect GitHub to Hugging Face

1. In your Hugging Face Space settings
2. Go to **"Repository"** tab
3. Click **"Connect to GitHub"**
4. Select your `SummerizerApp` repository
5. Choose the `main` branch

### Step 4: Configure Environment Variables

In your Hugging Face Space settings:

1. Go to **"Settings"** tab
2. Scroll to **"Environment Variables"**
3. Add these variables:

```
OLLAMA_MODEL=mistral:7b
OLLAMA_HOST=http://localhost:11434
OLLAMA_TIMEOUT=30
SERVER_HOST=0.0.0.0
SERVER_PORT=7860
LOG_LEVEL=INFO
MAX_TEXT_LENGTH=32000
MAX_TOKENS_DEFAULT=256
```

### Step 5: Deploy

1. Go to the **"Deploy"** tab in your Space
2. Click **"Deploy"**
3. Wait for the build to complete (5-10 minutes)

**What happens during deployment:**
- Docker image builds
- Ollama installs
- Mistral 7B model downloads (~4GB)
- FastAPI app starts
- Health checks run

## 🔍 Verification

### Check Your Deployment

1. **Visit your Space URL**: `https://your-username-summarizer-app.hf.space`
2. **Test Health Endpoint**: `https://your-username-summarizer-app.hf.space/health`
3. **View API Docs**: `https://your-username-summarizer-app.hf.space/docs`

### Test the API

```bash
# Test summarization
curl -X POST "https://your-username-summarizer-app.hf.space/api/v1/summarize" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is a long article about artificial intelligence and machine learning. It discusses various topics including natural language processing, computer vision, and deep learning techniques. The article covers the history of AI, current applications, and future prospects.",
    "max_tokens": 100
  }'
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Build Fails
- **Check Dockerfile**: Ensure it's named `Dockerfile` (not `Dockerfile.hf`)
- **Check README**: Ensure it has the proper frontmatter
- **Check logs**: View build logs in Hugging Face interface

#### 2. Model Not Loading
- **Wait longer**: Model download takes 5-10 minutes on first run
- **Check logs**: Look for Ollama-related errors
- **Verify model name**: Ensure `mistral:7b` is correct

#### 3. Out of Memory
- **Use smaller model**: Switch to `mistral:7b` (already configured)
- **Check hardware**: Ensure you're using CPU tier, not GPU

#### 4. Port Issues
- **Verify port**: Must use port `7860` for Hugging Face Spaces
- **Check SERVER_PORT**: Environment variable should be `7860`

### Debugging Commands

If you need to debug locally with HF configuration:

```bash
# Test with HF settings
cp env.hf .env
docker build -f Dockerfile.hf -t summarizer-hf .
docker run -p 7860:7860 summarizer-hf
```

## 📊 Performance Expectations

### Startup Time
- **First deployment**: 8-12 minutes (includes model download)
- **Subsequent deployments**: 3-5 minutes
- **Cold start**: 30-60 seconds

### Runtime Performance
- **Memory usage**: ~7-8GB RAM
- **Response time**: 2-5 seconds per request
- **Concurrent requests**: 1-2 (CPU limitation)

### Limitations
- **No GPU**: CPU-only inference
- **Shared resources**: May be slower during peak usage
- **Sleep mode**: Space may sleep after 48 hours of inactivity

## 🔧 Customization Options

### Use Different Model
Edit environment variables:
```
OLLAMA_MODEL=llama3.1:7b  # Smaller than 8b
OLLAMA_MODEL=mistral:7b   # Default, fastest
```

### Enable Security Features
```
API_KEY_ENABLED=true
API_KEY=your-secret-key
RATE_LIMIT_ENABLED=true
```

### Custom Domain
1. Go to Space settings
2. Add custom domain in "Settings" tab
3. Configure DNS as instructed

## 📈 Monitoring

### View Logs
1. Go to your Space
2. Click **"Logs"** tab
3. Monitor startup and runtime logs

### Health Monitoring
- **Health endpoint**: `/health`
- **Metrics**: Built-in Hugging Face monitoring
- **Uptime**: Check Space status page

## 🎉 Success!

Once deployed, your SummarizerApp will be available at:
`https://your-username-summarizer-app.hf.space`

### What You Get
- ✅ **Free hosting** forever
- ✅ **HTTPS endpoint** for your API
- ✅ **16GB RAM** for AI models
- ✅ **Automatic deployments** from GitHub
- ✅ **Built-in monitoring** and logs

### Next Steps
1. **Share your API** with others
2. **Integrate with apps** using the REST API
3. **Monitor usage** and performance
4. **Upgrade to GPU** if needed (paid tier)

---

**Congratulations! Your text summarization service is now live on Hugging Face Spaces! 🚀**
