# Codebase Q&A Assistant

A LangChain + RAG project that lets you ask natural-language questions about a codebase.

## How it works
1. Ingests a repository (code + docs) into a vector database.
2. Splits and embeds code/documents into searchable chunks.
3. Retrieves relevant context for a query and feeds it into an LLM.
4. Returns an answer with citations back to source files/lines.

## Usage
```bash
# Ingest repo into vector DB
python ingest.py

# Ask a question
python app.py "Where is update_particle_state used?"