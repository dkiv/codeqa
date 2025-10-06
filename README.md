# CodeQA – RAG on Your Codebase

This repo sets up a simple Retrieval-Augmented Generation (RAG) pipeline using [LangChain](https://www.langchain.com/).  
It allows you to ingest your codebase and then ask natural language questions about it.

---

## Features
- **Ingest your codebase**: Parse source files into embeddings and store them in a local [Chroma](https://www.trychroma.com/) vector store.
- **Ask questions**: Use `rag_chain.py` to query the vector store and get context-aware answers about your code.
- **Token + cost estimation**: Optional `cost_estimator.py` helps estimate how many tokens each query uses and approximate OpenAI API cost.
- **Extensible config**: Settings (like embedding model, persist directory, etc.) can be managed in `config.yaml`.
- **PDF/DOCX support**: Ingest PDFs and Word docs via `PyPDFLoader` and `Docx2txtLoader`.
- **Request-time pseudonymization**: Scrub PII (PERSON/ORG/EMAIL/PHONE/etc.) from the retrieved context and question before sending to the LLM, then decode the final answer locally.

---

## File Overview
- `ingest.py` → Reads files in your repo, splits them into chunks, embeds them, and saves them in Chroma.
- `rag_chain.py` → Loads the Chroma index and runs a retrieval + LLM chain to answer your questions.
- `cost_estimator.py` → Utility to estimate token usage and dollar cost of a prompt (if using OpenAI).
- `app.py` → Placeholder for a future API or UI layer (currently just a stub).
- `config.yaml` → Holds configuration for embedding model, vector DB path, and other options.
- `.env.example` → Example environment variables (e.g., `OPENAI_API_KEY`). Copy to `.env` and fill in.

---

## Setup
```bash
# Install dependencies
pip install -r requirements.txt

# (Optional) create .env from template
cp .env.example .env
 
# (Optional, for NER) install a spaCy model
# Only needed if you set privacy.enable_ner: true in config.yaml
python -m spacy download en_core_web_sm
```

Docker (alternative):
```bash
docker build -t codeqa .
```

---

## Usage
0. **Place your codebase in the `data/repo` folder. You can either:**
   - Copy files directly into `data/repo`, or
   - Clone a repository inside `data/repo` (e.g. `git clone <repo_url> data/repo`).

1. **Ingest your codebase**
   ```bash
   python ingest.py
   ```
   Docker:
   ```bash
   # Option A: mount only data (use --env-file for API keys)
   docker run --rm -it \
     -v "$PWD/data:/app/data" \
     codeqa ingest.py

   # Option B (dev): mount entire repo so .env/config are visible in the container
   docker run --rm -it \
     -v "$PWD:/app" \
     codeqa ingest.py

   # To force a clean rebuild of the index on each run (prevents duplicates):
   # Either set reset_index: true in config.yaml (already set), or pass RESET_INDEX=1
   docker run --rm -it \
     -v "$PWD:/app" \
     -e RESET_INDEX=1 \
     codeqa ingest.py
   ```

2. **Ask a question**
   ```bash
   python rag_chain.py "Where is GeneticAlgorithmRunner used?"
   ```
   Docker:
   ```bash
   # If your .env holds OPENAI_API_KEY and you want to use it without flags,
   # mount the repo so python-dotenv loads it from /app/.env
   docker run --rm -it \
     -v "$PWD:/app" \
     codeqa rag_chain.py "Where is GeneticAlgorithmRunner used?"

   # Or, keep the image code and just pass env vars via --env-file (no -e needed)
   docker run --rm -it \
     -v "$PWD/data:/app/data" \
     --env-file .env \
     codeqa rag_chain.py "Where is GeneticAlgorithmRunner used?"
   ```

### Privacy: Request-time Pseudonymization
- Enabled by default via `privacy.request_time: true` in `config.yaml`.
- Replaces entities (PERSON/ORG/GPE, EMAIL, PHONE, SSN, CREDIT_CARD, DATE, POSTAL_CODE, ADDRESS) with placeholders like `Person_1` before calling the LLM. The final answer is decoded back locally.
- Regex toggle: `privacy.enable_regex: true|false` controls structured PII scrubbing (EMAIL/PHONE/SSN/CREDIT_CARD/DATE/POSTAL_CODE/ADDRESS). Disable if you want to see raw values or to isolate regex effects during debugging.
- Optional NER: set `privacy.enable_ner: true` if you have spaCy and a model (e.g., `en_core_web_sm`) installed. If you are not using Docker, install the model with:
  ```bash
  python -m spacy download en_core_web_sm
  ```
  You can override the model via `SPACY_MODEL` env var. The Docker image installs `${SPACY_MODEL}` during build (default `en_core_web_sm`).
- When `privacy.enable_ner: true` and spaCy or the configured model is missing, the app raises an error with install instructions.
- Note: This does not modify embeddings or the stored index. For ingest-time pseudonymization (stronger privacy at the vector layer), add a separate flow and a stable mapping.

3. **Estimate cost**
   ```bash
   python cost_estimator.py "Your prompt here"
   ```

---

## Notes
- By default, embeddings are stored locally in `data/index`.
- HuggingFace embeddings are used (runs locally, no API calls).
- If you enable OpenAI models, make sure to add your `OPENAI_API_KEY` to `.env`.

### Redundant context troubleshooting
- If you see repeated or near-identical chunks in results:
  - Ensure you reset the index before ingesting (set `reset_index: true` in `config.yaml` or run with `RESET_INDEX=1`).
  - Try MMR retrieval (default in `config.yaml`: `retrieval.search_type: mmr`) to diversify results.
  - Optionally reduce `chunk.overlap` in `config.yaml` to minimize overlapping content.

---

## Docker

Build the image (includes spaCy + `en_core_web_sm`):
```bash
docker build -t codeqa .
```

Run ingestion (persist index to host):
```bash
# Minimal: mount only data
docker run --rm -it \
  -v "$PWD/data:/app/data" \
  codeqa ingest.py

# Dev-friendly: mount entire repo so .env/config are used automatically
docker run --rm -it \
  -v "$PWD:/app" \
  codeqa ingest.py
```

Ask a question:
```bash
# Using host .env automatically (mount repo)
docker run --rm -it \
  -v "$PWD:/app" \
  codeqa rag_chain.py "Where is GeneticAlgorithmRunner used?"

# Or, pass envs via file without mounting .env
docker run --rm -it \
  -v "$PWD/data:/app/data" \
  --env-file .env \
  codeqa rag_chain.py "Where is GeneticAlgorithmRunner used?"
```

Environment knobs:
- `SPACY_MODEL` (default `en_core_web_sm`) controls which spaCy model to load for NER.
- `CONFIG` can point to an alternate `config.yaml` if needed.

---

## Roadmap
- Replace `app.py` with a lightweight FastAPI server.
- Add Docker support.
- Expand config.yaml to cover LLM settings, retriever options, and more.
