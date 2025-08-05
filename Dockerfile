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
# Copy policy document and verify it exists
COPY policy.pdf .
RUN test -f policy.pdf && echo "✓ policy.pdf copied successfully" || echo "⚠ policy.pdf not found"
# Create a non-root user for security (optional but recommended)
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser
# Health check for Railway
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1
# Expose port (Railway will set the PORT env var)
EXPOSE ${PORT:-8000}
# Command to run the application
CMD ["python", "-u", "main.py"]
