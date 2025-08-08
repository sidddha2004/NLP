FROM python:3.9-alpine

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install build dependencies
RUN apk add --no-cache \
    gcc \
    g++ \
    musl-dev \
    linux-headers \
    libffi-dev \
    curl \
    && pip install --upgrade pip

# Install core dependencies
RUN pip install fastapi==0.104.1 uvicorn==0.24.0 pydantic==2.5.0
RUN pip install aiohttp==3.9.1 PyPDF2==3.0.1 requests==2.31.0 python-multipart==0.0.6

# Install minimal ML stack (CPU only)
RUN pip install numpy==1.24.3
RUN pip install torch==2.1.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install sentence-transformers==2.2.2

# Install cloud services  
RUN pip install pinecone-client==3.0.0 google-generativeai==0.3.2

# Clean up build dependencies to reduce size
RUN apk del gcc g++ musl-dev linux-headers

# Copy application
COPY main.py .

# Create user
RUN addgroup -g 1000 appuser && adduser -D -u 1000 -G appuser appuser
USER appuser

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
