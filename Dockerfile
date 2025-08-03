# Use Python slim image to reduce base size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Copy policy document if it exists (make it optional)
COPY policy.pdf* ./
RUN if [ -f policy.pdf ]; then \
        echo "✓ policy.pdf copied successfully"; \
        ls -la policy.pdf; \
    else \
        echo "⚠ policy.pdf not found - service will start without pre-loaded document"; \
    fi

# Create a non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port (Railway will set the PORT env var)
EXPOSE ${PORT:-8000}

# Health check with longer timeout for Railway
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=5 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Command to run the application with uvicorn directly for better Railway compatibility
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --timeout-keep-alive 65"]
