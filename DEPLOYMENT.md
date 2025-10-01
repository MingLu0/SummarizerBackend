# Cloud Deployment Guide

This guide covers multiple options for deploying your text summarizer backend to the cloud.

## 🚀 **Option 1: Railway (Recommended - Easiest)**

Railway is perfect for this project because it supports Docker Compose and persistent volumes.

### Steps:

1. **Create Railway Account**
   ```bash
   # Install Railway CLI
   npm install -g @railway/cli
   
   # Login
   railway login
   ```

2. **Deploy from GitHub**
   - Go to [railway.app](https://railway.app)
   - Connect your GitHub repository
   - Select your `SummerizerApp` repository
   - Railway will automatically detect `docker-compose.yml`

3. **Set Environment Variables**
   In Railway dashboard, add these environment variables:
   ```
   OLLAMA_MODEL=llama3.1:8b
   OLLAMA_HOST=http://ollama:11434
   OLLAMA_TIMEOUT=30
   SERVER_HOST=0.0.0.0
   SERVER_PORT=8000
   LOG_LEVEL=INFO
   ```

4. **Deploy**
   ```bash
   # Or deploy via CLI
   railway up
   ```

### Railway Advantages:
- ✅ Supports Docker Compose
- ✅ Persistent volumes for Ollama models
- ✅ Automatic HTTPS
- ✅ Easy environment variable management
- ✅ Built-in monitoring

---

## ☁️ **Option 2: Google Cloud Run**

### Steps:

1. **Build and Push to Google Container Registry**
   ```bash
   # Set up gcloud CLI
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   
   # Build and push
   docker build -t gcr.io/YOUR_PROJECT_ID/summarizer-backend .
   docker push gcr.io/YOUR_PROJECT_ID/summarizer-backend
   ```

2. **Deploy with Cloud Run**
   ```bash
   gcloud run deploy summarizer-backend \
     --image gcr.io/YOUR_PROJECT_ID/summarizer-backend \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --memory 4Gi \
     --cpu 2 \
     --timeout 300 \
     --set-env-vars OLLAMA_MODEL=llama3.1:8b,SERVER_HOST=0.0.0.0,SERVER_PORT=8000
   ```

### Cloud Run Advantages:
- ✅ Serverless scaling
- ✅ Pay per request
- ✅ Global CDN
- ✅ Integrated with Google Cloud

---

## 🐳 **Option 3: AWS ECS with Fargate**

### Steps:

1. **Create ECR Repository**
   ```bash
   aws ecr create-repository --repository-name summarizer-backend
   ```

2. **Build and Push**
   ```bash
   # Get login token
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com
   
   # Build and push
   docker build -t summarizer-backend .
   docker tag summarizer-backend:latest YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/summarizer-backend:latest
   docker push YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/summarizer-backend:latest
   ```

3. **Create ECS Task Definition**
   ```json
   {
     "family": "summarizer-backend",
     "networkMode": "awsvpc",
     "requiresCompatibilities": ["FARGATE"],
     "cpu": "2048",
     "memory": "4096",
     "executionRoleArn": "arn:aws:iam::YOUR_ACCOUNT:role/ecsTaskExecutionRole",
     "containerDefinitions": [
       {
         "name": "summarizer-backend",
         "image": "YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/summarizer-backend:latest",
         "portMappings": [
           {
             "containerPort": 8000,
             "protocol": "tcp"
           }
         ],
         "environment": [
           {
             "name": "OLLAMA_MODEL",
             "value": "llama3.1:8b"
           },
           {
             "name": "SERVER_HOST",
             "value": "0.0.0.0"
           },
           {
             "name": "SERVER_PORT",
             "value": "8000"
           }
         ],
         "logConfiguration": {
           "logDriver": "awslogs",
           "options": {
             "awslogs-group": "/ecs/summarizer-backend",
             "awslogs-region": "us-east-1",
             "awslogs-stream-prefix": "ecs"
           }
         }
       }
     ]
   }
   ```

---

## 🌊 **Option 4: DigitalOcean App Platform**

### Steps:

1. **Create App Spec**
   ```yaml
   # .do/app.yaml
   name: summarizer-backend
   services:
   - name: api
     source_dir: /
     github:
       repo: MingLu0/SummarizerBackend
       branch: main
     run_command: uvicorn app.main:app --host 0.0.0.0 --port 8080
     environment_slug: python
     instance_count: 1
     instance_size_slug: basic-xxl
     http_port: 8080
     envs:
     - key: OLLAMA_MODEL
       value: llama3.1:8b
     - key: SERVER_HOST
       value: 0.0.0.0
     - key: SERVER_PORT
       value: 8080
   ```

2. **Deploy**
   ```bash
   doctl apps create --spec .do/app.yaml
   ```

---

## 🔧 **Option 5: Render (Simple)**

### Steps:

1. **Connect GitHub Repository**
   - Go to [render.com](https://render.com)
   - Connect your GitHub account
   - Select your repository

2. **Create Web Service**
   - Choose "Web Service"
   - Select your repository
   - Use these settings:
     ```
     Build Command: docker-compose build
     Start Command: docker-compose up
     Environment: Docker
     ```

3. **Set Environment Variables**
   ```
   OLLAMA_MODEL=llama3.1:8b
   OLLAMA_HOST=http://ollama:11434
   SERVER_HOST=0.0.0.0
   SERVER_PORT=8000
   ```

---

## ⚠️ **Important Considerations**

### **Model Download in Cloud**
Your Ollama models need to be downloaded after deployment. Add this to your deployment:

```bash
# Add to docker-compose.yml or startup script
ollama pull llama3.1:8b
```

### **Memory Requirements**
- **llama3.1:8b** needs ~8GB RAM
- **llama3.1:7b** needs ~7GB RAM
- **mistral:7b** needs ~7GB RAM

### **Cost Optimization**
- Use smaller models for production: `mistral:7b` or `llama3.1:7b`
- Consider using spot instances for development
- Monitor usage and scale accordingly

### **Security**
- Enable API key authentication for production
- Use HTTPS (most platforms provide this automatically)
- Set up rate limiting
- Monitor logs for abuse

---

## 🎯 **Recommended Deployment Flow**

1. **Start with Railway** (easiest setup)
2. **Test with a smaller model** (mistral:7b)
3. **Monitor performance and costs**
4. **Scale up model size if needed**
5. **Add security features**

### **Quick Railway Deploy:**
```bash
# 1. Install Railway CLI
npm install -g @railway/cli

# 2. Login and deploy
railway login
railway init
railway up
```

Your backend will be live at `https://your-app.railway.app`! 🚀
