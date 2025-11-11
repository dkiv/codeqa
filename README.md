# CodeQA – RAG on Your Codebase

This repo provides a simple Retrieval‑Augmented Generation (RAG) pipeline using LangChain. It ingests your code/docs into a local Chroma vector store and lets you ask natural‑language questions about them.

---

## Features
- Ingest code/docs to local Chroma (no external services required for embeddings)
- Ask questions via a compact RAG chain with inline file:line citations
- Optional token/cost estimation for OpenAI calls
- PDF/DOCX loaders available (if installed)
- Request‑time pseudonymization (regex + optional spaCy NER)

---

## File Overview
- `ingest.py` — Discover files, split, embed, and persist to Chroma
- `rag_chain.py` — Load the index and run retrieval + LLM
- `cost_estimator.py` — Estimate tokens and approximate cost
- `config.yaml` — Central configuration (paths, chunking, retrieval, models, privacy)
- `.env.example` — Example environment variables

---

## Local (No Docker)

Prerequisites
- Python 3.10+ (tested on 3.11)

Setup
```bash
# 1) Create and activate a virtualenv in .venv/
python3 -m venv .venv
source .venv/bin/activate   # Windows PowerShell: .\.venv\Scripts\Activate.ps1

# 2) Install dependencies
pip install -r requirements.txt

# 3) (Optional) Environment file
cp .env.example .env

# 4) (Optional, only if privacy.enable_ner: true)
python -m spacy download en_core_web_sm
```

Prepare data
- Place or clone the code/docs you want to search into `data/repo`.

Ingest
```bash
# Build or rebuild the vector index
python ingest.py

# To force a clean rebuild (prevents duplicates)
RESET_INDEX=1 python ingest.py
```

Ask a question
```bash
python rag_chain.py "Where is GeneticAlgorithmRunner used?"
```

Enable OpenAI calls (optional)
- By default, `rag_chain.py` prints a copy‑paste prompt and does not call an API.
- To call OpenAI:
  - Put your key in `.env` as `OPENAI_API_KEY=...`
  - Set `app.use_openai_api: true` in `config.yaml` or export `APP_USE_OPENAI_API=true`

Cost estimation (optional)
```bash
python cost_estimator.py "Your prompt here"
```

Notes
- `.venv/` is ignored by ingestion and Docker; it is safe to keep in the repo root.
- Embeddings persist to `data/index` by default.

---

## Docker

Build
```bash
docker build -t codeqa .
```

Ingest
```bash
# Option A: mount only data (simple, persists index)
docker run --rm -it \
  -v "$PWD/data:/app/data" \
  codeqa ingest.py

# Option B: dev‑friendly (mount entire repo so .env/config are visible)
docker run --rm -it \
  -v "$PWD:/app" \
  codeqa ingest.py

# Clean rebuild
docker run --rm -it \
  -v "$PWD:/app" \
  -e RESET_INDEX=1 \
  codeqa ingest.py
```

Ask a question
```bash
# Using host .env automatically (mount repo)
docker run --rm -it \
  -v "$PWD:/app" \
  codeqa rag_chain.py "Where is GeneticAlgorithmRunner used?"

# Or pass envs without mounting .env
docker run --rm -it \
  -v "$PWD/data:/app/data" \
  --env-file .env \
  codeqa rag_chain.py "Where is GeneticAlgorithmRunner used?"
```

Environment knobs
- `SPACY_MODEL` (default `en_core_web_sm`) controls which spaCy model to install in the image.
- `CONFIG` can point to an alternate `config.yaml` if needed.

---

## Configuration & Privacy
- Edit `config.yaml` to adjust `repo_path`, `index_path`, chunking, retrieval (`k`, `search_type: mmr`), and models.
- Privacy (request‑time pseudonymization):
  - `privacy.enable_regex: true|false` for EMAIL/PHONE/SSN/CREDIT_CARD/DATE/POSTAL_CODE/ADDRESS
  - `privacy.enable_ner: true|false` to include PERSON/ORG/GPE (requires spaCy + model)

Troubleshooting redundant context
- Reset the index before re‑ingesting (`reset_index: true` in `config.yaml` or `RESET_INDEX=1`)
- Try MMR retrieval (`retrieval.search_type: mmr`) to diversify results
- Reduce `chunk.overlap` to minimize overlapping content

---

## Roadmap
- Replace `app.py` with a lightweight FastAPI server
- Expand config to cover more retriever/LLM options
