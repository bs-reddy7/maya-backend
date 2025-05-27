# Use Python 3.10 slim image for better compatibility
FROM python:3.10-slim

# Install system dependencies for audio processing and ML libraries
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    build-essential \
    libffi-dev \
    libssl-dev \
    libasound2-dev \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Set environment variables for better Python behavior
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Copy requirements first for better Docker caching
COPY requirements.txt .

# Install Python dependencies with proper timeout for heavy packages
RUN pip install --upgrade pip setuptools wheel && \
    pip install --timeout=600 --retries=3 -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p temp_uploads logs

# Expose port for Fly.io
EXPOSE 8080

# Health check for full API with longer startup time
HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8080/ || exit 1

# Run the Maya application with proper error handling
CMD ["python", "main.py"]