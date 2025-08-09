FROM python:3.11-slim-bullseye

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1

# Install only what's needed for build
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Make volume for Hugging Face models
VOLUME ["/app/models"]

# Preload model into mounted volume
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2', cache_folder='/app/models')"

# Remove build deps to keep size down
RUN apt-get purge -y gcc && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY main.py .

# Create non-root user
RUN adduser --disabled-password --gecos '' --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')" || exit 1

CMD ["python", "main.py"]
