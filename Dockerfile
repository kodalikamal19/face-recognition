# Multi-Face Recognition Application Container
FROM python:3.11-slim

# Environment settings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# Install build tools if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for optimal Docker layer caching
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy required application assets
COPY src/ ./src/
COPY frontend/ ./frontend/
COPY embeddings/ ./embeddings/
COPY test/ ./test/
COPY app.py .

EXPOSE 8000

# Run FastAPI app with dynamic port binding for Render / Railway / Heroku
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
