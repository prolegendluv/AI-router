# Multi-stage Docker build enforcing linux/amd64 architecture
FROM --platform=linux/amd64 python:3.11-slim as builder

WORKDIR /app

# Prevent Python from writing pyc files to disk & buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install required system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Final stage for fast container startup (< 60 seconds requirement)
FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV LOCAL_ENGINE_URL="http://127.0.0.1:11434"

# Copy installed python dependencies from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source code
COPY . .

# Ensure /input and /output directories can be mounted cleanly
RUN mkdir -p /input /output

# Container entrypoint must read /input/tasks.json and write /output/results.json
ENTRYPOINT ["python", "main.py"]
