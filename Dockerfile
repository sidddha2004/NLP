FROM python:3.9-slim

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade pip
RUN pip install --upgrade pip

# Install dependencies in order of importance
RUN pip install fastapi==0.104.1 uvicorn==0.24.0
RUN pip install pydantic==2.5.0 aiohttp==3.9.1
RUN pip install PyPDF2==3.0.1 python-multipart==0.0.6 requests==2.31.0
RUN pip install numpy==1.24.3

# Install PyTorch CPU-only (much smaller)
RUN pip install torch==2.1.1 --index-url https://download.pytorch.org/whl/cpu

# Install ML libraries
RUN pip install sentence-transformers==2.2.2 transformers==4.35.2

# Install cloud services
RUN pip install pinecone-client==3.0.0 google-generativeai==0.3.2

# Clean up temporary files
RUN find /usr/local -name "*.pyc" -delete && \
    find /usr/local -name "__pycache__" -delete

# Remove build dependencies to save space
RUN apt-get purge -y gcc g++ && apt-get autoremove -y

# Copy your application
COPY main.py .

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=2 \
    CMD curl -f http://localhost:8080/health || exit 1

# Start application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
