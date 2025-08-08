FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    build-essential \
    python3-dev \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Install core dependencies first
RUN pip install fastapi==0.104.1
RUN pip install uvicorn[standard]==0.24.0
RUN pip install pydantic==2.5.0
RUN pip install aiohttp==3.9.1
RUN pip install PyPDF2==3.0.1
RUN pip install python-multipart==0.0.6
RUN pip install requests==2.31.0
RUN pip install python-dotenv==1.0.0

# Install numpy first (many packages depend on it)
RUN pip install numpy==1.24.3

# Install PyTorch CPU version
RUN pip install torch==2.1.1 --index-url https://download.pytorch.org/whl/cpu

# Install ML libraries
RUN pip install transformers==4.35.2
RUN pip install tokenizers==0.14.1
RUN pip install huggingface-hub==0.20.3
RUN pip install sentence-transformers==2.2.2

# Install remaining dependencies
RUN pip install pinecone-client==3.0.0
RUN pip install google-generativeai==0.3.2
RUN pip install scikit-learn==1.3.2
RUN pip install scipy==1.11.4
RUN pip install Pillow==10.1.0
RUN pip install typing-extensions==4.8.0
RUN pip install safetensors==0.4.1

# Copy application code
COPY main.py .

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

EXPOSE 8080

# Start the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
