FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SPACY_MODEL=en_core_web_sm

WORKDIR /app

# System deps (optional: for compiling wheels or additional loaders)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download ${SPACY_MODEL}

# Copy the project
COPY . .

# Default command runs the RAG chain; override as needed
ENTRYPOINT ["python"]
CMD ["rag_chain.py", "What does this repo do?"]

