# Railway-Optimized Dockerfile for Phase 2
# Specifically optimized for Railway's deployment environment

FROM python:3.11-slim as builder

# Set environment variables for build
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt /tmp/
RUN pip install --user --no-warn-script-location -r /tmp/requirements.txt

# Production stage
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/home/appuser/.local/bin:$PATH"

# Create app user
RUN useradd --create-home --shell /bin/bash appuser

# Copy installed packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Set working directory
WORKDIR /home/appuser/app

# Copy application files
COPY --chown=appuser:appuser main.py ./

# Switch to app user
USER appuser

# Expose port (Railway will set PORT env var)
EXPOSE $PORT

# Health check optimized for Railway
HEALTHCHECK --interval=60s --timeout=10s --start-period=10s --retries=2 \
    CMD python -c "import requests; requests.get(f'http://localhost:{__import__(\"os\").environ.get(\"PORT\", 8000)}/health', timeout=5)" || exit 1

# Use Railway's PORT environment variable
CMD python -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
