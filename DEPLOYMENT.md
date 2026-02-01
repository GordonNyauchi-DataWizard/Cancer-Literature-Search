# Deployment Guide

This guide covers deploying the Cancer Literature Search system to various platforms.

## Table of Contents

1. [Local Deployment](#local-deployment)
2. [Streamlit Cloud](#streamlit-cloud)
3. [Hugging Face Spaces](#hugging-face-spaces)
4. [Docker Deployment](#docker-deployment)
5. [AWS Deployment](#aws-deployment)
6. [Troubleshooting](#troubleshooting)

---

## Local Deployment

### Prerequisites
- Python 3.8+
- 16GB RAM recommended
- 10GB free disk space

### Steps

```bash
# 1. Clone repository
git clone https://github.com/yourusername/cancer-literature-search.git
cd cancer-literature-search

# 2. Run setup script
chmod +x setup.sh
./setup.sh

# 3. Add your PDFs to papers/ directory

# 4. Build index
python cli.py --rebuild

# 5. Run application
# CLI:
python cli.py

# Web interface:
streamlit run app.py
```

---

## Streamlit Cloud

Deploy your app for free on Streamlit Cloud with public URL.

### Steps

1. **Prepare Repository**
   ```bash
   # Ensure these files are in your repo:
   - app.py
   - semantic_search.py
   - requirements.txt
   - README.md
   ```

2. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/yourusername/cancer-literature-search.git
   git push -u origin main
   ```

3. **Deploy on Streamlit Cloud**
   
   a. Go to [share.streamlit.io](https://share.streamlit.io)
   
   b. Click "New app"
   
   c. Connect your GitHub repository
   
   d. Configure:
      - **Repository**: yourusername/cancer-literature-search
      - **Branch**: main
      - **Main file**: app.py
   
   e. Add secrets in "Advanced settings":
      ```toml
      ANTHROPIC_API_KEY = "your-api-key-here"
      ```
   
   f. Click "Deploy"

4. **Note on PDFs**
   
   Since Git doesn't handle large files well:
   
   **Option A**: Use Git LFS
   ```bash
   git lfs install
   git lfs track "papers/*.pdf"
   git add .gitattributes
   git add papers/
   git commit -m "Add papers with LFS"
   git push
   ```
   
   **Option B**: Upload to cloud storage
   ```python
   # Modify app.py to download from S3/GCS
   import boto3
   
   def download_papers():
       s3 = boto3.client('s3')
       s3.download_file('my-bucket', 'papers.zip', 'papers.zip')
       # Extract...
   ```

### Streamlit Cloud Limits
- Free tier: 1GB RAM
- App sleeps after inactivity
- Public by default

---

## Hugging Face Spaces

Deploy on Hugging Face for free hosting with GPU support.

### Steps

1. **Create Space**
   
   a. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
   
   b. Click "Create new Space"
   
   c. Configure:
      - **Space name**: cancer-literature-search
      - **License**: MIT
      - **SDK**: Streamlit
      - **Hardware**: CPU Basic (free) or GPU (upgrade)

2. **Prepare Files**
   
   Create `README.md` in Space format:
   ```markdown
   ---
   title: Cancer Literature Search
   emoji: 🔬
   colorFrom: blue
   colorTo: purple
   sdk: streamlit
   sdk_version: 1.31.0
   app_file: app.py
   pinned: false
   ---
   
   # Cancer Literature Search
   
   [Your description here]
   ```

3. **Upload Files**
   ```bash
   # Clone your space
   git clone https://huggingface.co/spaces/yourusername/cancer-literature-search
   cd cancer-literature-search
   
   # Copy files
   cp /path/to/app.py .
   cp /path/to/semantic_search.py .
   cp /path/to/requirements.txt .
   
   # Commit and push
   git add .
   git commit -m "Initial deployment"
   git push
   ```

4. **Add Secrets**
   
   In Space settings → Repository secrets:
   ```
   ANTHROPIC_API_KEY = your-key-here
   ```

5. **Handle Large Files**
   
   Hugging Face has built-in LFS support:
   ```bash
   git lfs install
   git lfs track "papers/*.pdf"
   git add .gitattributes papers/
   git commit -m "Add papers"
   git push
   ```

### Hugging Face Spaces Advantages
- Free GPU access (limited hours)
- Better for large datasets
- Integrated with Hugging Face ecosystem

---

## Docker Deployment

Containerize your application for consistent deployment anywhere.

### Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY semantic_search.py .
COPY app.py .
COPY cli.py .

# Create directories
RUN mkdir -p papers index

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./papers:/app/papers
      - ./index:/app/index
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    restart: unless-stopped
```

### Build and Run

```bash
# Build image
docker build -t cancer-search .

# Run container
docker run -p 8501:8501 \
  -v $(pwd)/papers:/app/papers \
  -v $(pwd)/index:/app/index \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  cancer-search

# Or use docker-compose
docker-compose up -d
```

### Push to Docker Hub

```bash
# Tag image
docker tag cancer-search yourusername/cancer-search:latest

# Push
docker push yourusername/cancer-search:latest

# Others can pull and run
docker pull yourusername/cancer-search:latest
docker run -p 8501:8501 yourusername/cancer-search:latest
```

---

## AWS Deployment

Deploy on AWS for production use with scalability.

### Option 1: AWS ECS (Elastic Container Service)

1. **Build and Push Docker Image to ECR**
   ```bash
   # Authenticate
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin your-account-id.dkr.ecr.us-east-1.amazonaws.com
   
   # Create repository
   aws ecr create-repository --repository-name cancer-search
   
   # Tag and push
   docker tag cancer-search:latest your-account-id.dkr.ecr.us-east-1.amazonaws.com/cancer-search:latest
   docker push your-account-id.dkr.ecr.us-east-1.amazonaws.com/cancer-search:latest
   ```

2. **Create ECS Task Definition**
   ```json
   {
     "family": "cancer-search",
     "networkMode": "awsvpc",
     "requiresCompatibilities": ["FARGATE"],
     "cpu": "1024",
     "memory": "2048",
     "containerDefinitions": [
       {
         "name": "cancer-search",
         "image": "your-account-id.dkr.ecr.us-east-1.amazonaws.com/cancer-search:latest",
         "portMappings": [
           {
             "containerPort": 8501,
             "protocol": "tcp"
           }
         ],
         "environment": [
           {
             "name": "ANTHROPIC_API_KEY",
             "value": "your-key-here"
           }
         ]
       }
     ]
   }
   ```

3. **Create ECS Service**
   ```bash
   aws ecs create-service \
     --cluster your-cluster \
     --service-name cancer-search \
     --task-definition cancer-search \
     --desired-count 1 \
     --launch-type FARGATE \
     --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}"
   ```

### Option 2: AWS EC2

1. **Launch EC2 Instance**
   - AMI: Amazon Linux 2
   - Instance type: t3.large (8GB RAM)
   - Storage: 50GB

2. **SSH and Setup**
   ```bash
   ssh -i your-key.pem ec2-user@your-instance-ip
   
   # Install Docker
   sudo yum update -y
   sudo amazon-linux-extras install docker
   sudo service docker start
   sudo usermod -a -G docker ec2-user
   
   # Pull and run
   docker pull yourusername/cancer-search:latest
   docker run -d -p 80:8501 \
     -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
     yourusername/cancer-search:latest
   ```

3. **Configure Security Group**
   - Allow inbound TCP on port 80 from 0.0.0.0/0

### Option 3: AWS Lambda + API Gateway

For serverless deployment (CLI only, not Streamlit):

```python
# lambda_function.py
import json
from semantic_search import CancerSearchApp

app = CancerSearchApp()
# Load index from S3
app.build_or_load_index()

def lambda_handler(event, context):
    query = event.get('query', '')
    results = app.search(query)
    
    return {
        'statusCode': 200,
        'body': json.dumps(results)
    }
```

---

## Environment Variables

Set these for production:

```bash
# Required
export ANTHROPIC_API_KEY="your-key-here"

# Optional
export PDF_DIR="/path/to/papers"
export INDEX_DIR="/path/to/index"
export MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
export TOP_K="10"
export LOG_LEVEL="INFO"
```

---

## Monitoring and Logging

### Streamlit Cloud
- Built-in logs in dashboard
- View real-time metrics

### Docker
```bash
# View logs
docker logs container-id

# Follow logs
docker logs -f container-id

# Container stats
docker stats container-id
```

### AWS CloudWatch
```python
# Add CloudWatch logging
import watchtower
import logging

logger = logging.getLogger(__name__)
logger.addHandler(watchtower.CloudWatchLogHandler())
```

---

## Scaling Considerations

### Horizontal Scaling
- Use load balancer (ALB on AWS, nginx)
- Deploy multiple container instances
- Share index via network file system (EFS on AWS)

### Vertical Scaling
- Increase RAM for larger datasets
- Use GPU instances for faster embedding

### Caching
```python
# Add Redis for caching
import redis

cache = redis.Redis(host='localhost', port=6379)

def cached_search(query):
    cached = cache.get(query)
    if cached:
        return json.loads(cached)
    
    result = app.search(query)
    cache.setex(query, 3600, json.dumps(result))
    return result
```

---

## Security Best Practices

1. **Never commit secrets**
   ```bash
   # Use environment variables
   export ANTHROPIC_API_KEY="..."
   
   # Or secrets manager
   aws secretsmanager create-secret --name anthropic-key --secret-string "..."
   ```

2. **Use HTTPS**
   - Streamlit Cloud: Automatic
   - AWS: Use ACM certificate + ALB
   - Self-hosted: Use nginx reverse proxy with Let's Encrypt

3. **Restrict access**
   ```python
   # Add authentication to Streamlit
   import streamlit_authenticator as stauth
   
   authenticator = stauth.Authenticate(...)
   name, authentication_status, username = authenticator.login()
   
   if authentication_status:
       # Show app
   ```

4. **Rate limiting**
   ```python
   from functools import wraps
   import time
   
   def rate_limit(max_calls, period):
       def decorator(func):
           calls = []
           @wraps(func)
           def wrapper(*args, **kwargs):
               now = time.time()
               calls[:] = [c for c in calls if c > now - period]
               if len(calls) >= max_calls:
                   raise Exception("Rate limit exceeded")
               calls.append(now)
               return func(*args, **kwargs)
           return wrapper
       return decorator
   ```

---

## Troubleshooting

### Issue: Out of memory
**Solution**: 
- Increase container/instance memory
- Reduce `BATCH_SIZE`
- Use smaller embedding model

### Issue: Slow performance
**Solution**:
- Enable GPU
- Implement caching
- Use FAISS for large datasets

### Issue: API rate limits
**Solution**:
- Implement exponential backoff
- Cache LLM responses
- Use rate limiting

---

## Cost Estimation

### Streamlit Cloud
- Free tier: $0/month (limited resources)

### Hugging Face Spaces
- CPU Basic: Free
- GPU: ~$0.50/hour (limited free hours)

### AWS (Monthly)
- EC2 t3.large: ~$60
- ECS Fargate (1 vCPU, 2GB): ~$30
- S3 storage (100GB): ~$2.30
- Data transfer: ~$9/100GB

### Anthropic API (per million tokens)
- Haiku: $0.25 in / $1.25 out
- Sonnet: $3 in / $15 out
- Opus: $15 in / $75 out

---

For more deployment options or custom requirements, refer to the platform documentation or open an issue on GitHub.
