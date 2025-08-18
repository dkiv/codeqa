# CodeQA – RAG on Your Codebase

This repo sets up a simple Retrieval-Augmented Generation (RAG) pipeline using [LangChain](https://www.langchain.com/).  
It allows you to ingest your codebase and then ask natural language questions about it.

---

## Features
- **Ingest your codebase**: Parse source files into embeddings and store them in a local [Chroma](https://www.trychroma.com/) vector store.
- **Ask questions**: Use `rag_chain.py` to query the vector store and get context-aware answers about your code.
- **Token + cost estimation**: Optional `cost_estimator.py` helps estimate how many tokens each query uses and approximate OpenAI API cost.
- **Extensible config**: Settings (like embedding model, persist directory, etc.) can be managed in `config.yaml`.

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

2. **Ask a question**
   ```bash
   python rag_chain.py "Where is GeneticAlgorithmRunner used?"
   ```

3. **Estimate cost**
   ```bash
   python cost_estimator.py "Your prompt here"
   ```

---

## Notes
- By default, embeddings are stored locally in `./chroma_index`.
- HuggingFace embeddings are used (runs locally, no API calls).
- If you enable OpenAI models, make sure to add your `OPENAI_API_KEY` to `.env`.

---

## Roadmap
- Replace `app.py` with a lightweight FastAPI server.
- Add Docker support.
- Expand config.yaml to cover LLM settings, retriever options, and more.