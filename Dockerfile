# Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt --disable-pip-version-check

# Create data directory
RUN mkdir -p /app/data

# Copy application code
COPY src/ ./src/

EXPOSE 8000

CMD ["uvicorn", "src.bot:app", "--host", "0.0.0.0", "--port", "8000"]