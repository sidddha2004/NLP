# Use Python slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Environment configs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies (for torch, nltk, sentence-transformers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    git \
    curl \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download nltk punkt tokenizer (used for sentence chunking)
RUN python -m nltk.downloader punkt

# Copy application code
COPY main.py .
COPY policy.pdf .

# Check if policy file is copied
RUN test -f policy.pdf && echo "✓ policy.pdf copied successfully" || echo "⚠ policy.pdf not found"

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Healthcheck (Railway)
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Expose port
EXPOSE ${PORT:-8000}

# Start the app
CMD ["python", "-u", "main.py"]
