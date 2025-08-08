FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc g++ curl && rm -rf /var/lib/apt/lists/*

# Copy requirements and install in one step to avoid conflicts
COPY requirements-simple.txt requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Create simple startup script
RUN echo '#!/bin/bash\necho "Starting application..."\npython main.py' > start.sh && chmod +x start.sh

EXPOSE 8080

CMD ["./start.sh"]
