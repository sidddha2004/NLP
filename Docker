# Dockerfile

FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV POETRY_VIRTUALENVS_CREATE=false

# Set working directory
WORKDIR /app

# Install system dependencies (for pdf and docx processing)
RUN apt-get update && apt-get install -y \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose the port (default 8000)
EXPOSE 8000

# Run the FastAPI app with uvicorn
CMD ["uvicorn", "your_script_name:app", "--host", "0.0.0.0", "--port", "8000"]
