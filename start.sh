#!/bin/bash
# Startup script for Railway deployment

# Get the PORT environment variable, default to 8000 if not set
export APP_PORT=${PORT:-8000}

echo "Starting HackRX Q&A System on port $APP_PORT"

# Start the application using Python
exec python main.py
