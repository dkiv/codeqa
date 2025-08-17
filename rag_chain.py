"""
rag_chain.py

Defines the Retrieval-Augmented Generation (RAG) pipeline:
- Retrieves relevant code/doc chunks from the vector DB (and optionally BM25).
- Builds a prompt with retrieved context and user query.
- Calls the LLM to generate an answer with citations.
"""