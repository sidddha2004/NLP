# Use the smallest Python base image
FROM python:3.11-alpine as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies (minimal)
RUN apk add --no-cache \
    build-base \
    libffi-dev \
    openssl-dev

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage - ultra minimal
FROM python:3.11-alpine

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/root/.local/bin:$PATH"

# Install only essential runtime dependencies
RUN apk add --no-cache libffi openssl

# Copy only installed packages
COPY --from=builder /root/.local /root/.local

# Create app directory
WORKDIR /app

# Copy only essential files
COPY main.py .
COPY policy.pdf .

# Expose port
EXPOSE 8000

# Use exec form and single worker for memory efficiency
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--access-log", "--no-use-colors"]
