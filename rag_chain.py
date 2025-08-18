"""
rag_chain.py

Minimal LangChain RAG chain for querying the Chroma index built by ingest.py.

- Loads the persisted Chroma index from config/index_path
- Uses the same HuggingFaceEmbeddings model as ingest for query embeddings
- Retrieves top-k chunks and builds a prompt with inline file:line citations
- Calls an LLM and returns the answer text

Requirements:
    pip install langchain langchain-community langchain-text-splitters chromadb sentence-transformers
    # And whichever chat model backend you plan to use (e.g., langchain-openai)
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from cost_estimator import count_tokens, estimate_cost_usd, estimate_and_format

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from langchain_chroma import Chroma  # type: ignore
# from langchain_openai import ChatOpenAI  # type: ignore
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger("codeqa.rag")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

DEFAULT_CONFIG: Dict[str, Any] = {
    "index_path": "data/index",
    "model": {
        "embedding": "BAAI/bge-small-en",
        "chat": "gpt-4o-mini",
        "temperature": 0.0,
    },
    "retrieval": {
        "k": 8,
        "max_context_chunks": 6,
        "max_chunk_chars": 1200,
    },
}

SYSTEM_PROMPT = (
    "You are a helpful codebase assistant. Answer the user's question using ONLY the provided context. "
    "If the context is insufficient, say so and suggest where to look in the code. "
    "Cite your sources inline as [file.py:start-end]. Keep answers concise and actionable."
)
PROMPT_TMPL = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer:"),
    ]
)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load YAML config if available; otherwise return defaults."""
    if not config_path:
        here = Path(__file__).resolve().parent
        for c in (here / "config.yaml", here.parent / "config.yaml"):
            if c.exists():
                config_path = str(c)
                break
    if config_path and Path(config_path).exists() and yaml is not None:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        merged = {**DEFAULT_CONFIG, **cfg}
    else:
        merged = DEFAULT_CONFIG.copy()
    return merged


def _build_retriever(index_path: str, embedding_model: str, k: int):
    """Open persisted Chroma and return a retriever. Uses matching embedding model for queries."""
    emb = HuggingFaceEmbeddings(model_name=embedding_model)
    vs = Chroma(persist_directory=index_path, embedding_function=emb, collection_name="codeqa")
    return vs.as_retriever(search_kwargs={"k": k})


def _format_context(docs: List[Any], max_chunk_chars: int = 1200) -> str:
    parts: List[str] = []
    for d in docs:
        src = d.metadata.get("source", "?")
        loc = d.metadata.get("loc", "?")
        txt = (d.page_content or "").strip()
        if len(txt) > max_chunk_chars:
            txt = txt[: max_chunk_chars - 3] + "..."
        parts.append(f"[{src}:{loc}]\n{txt}")
    return "\n\n".join(parts)


def _build_llm(model_name: str, temperature: float = 0.0):
    """Create the chat model. Defaults to OpenAI via langchain-openai.

    Set OPENAI_API_KEY in your environment. Replace with another provider if needed.
    """
    return ChatOpenAI(model=model_name, temperature=temperature)


def ask(question: str, config_path: Optional[str] = None) -> str:
    """Answer a question using the local Chroma index.

    Example:
        from rag_chain import ask
        print(ask("Where is update_particle_state used?"))
    """
    cfg = load_config(config_path)
    index_path: str = cfg["index_path"]
    embedding_model: str = cfg["model"]["embedding"]
    chat_model: str = cfg["model"].get("chat", "gpt-4o-mini")
    temperature: float = float(cfg["model"].get("temperature", 0.0))

    k: int = int(cfg["retrieval"].get("k", 8))
    max_context_chunks: int = int(cfg["retrieval"].get("max_context_chunks", 6))
    max_chunk_chars: int = int(cfg["retrieval"].get("max_chunk_chars", 1200))

    retriever = _build_retriever(index_path, embedding_model, k)
    docs = retriever.invoke(question)[:max_context_chunks]
    if not docs:
        return "I couldn't find relevant context in the index. Try re-ingesting or broadening your query."

    context = _format_context(docs, max_chunk_chars=max_chunk_chars)
    # Debug: print the constructed prompt for copy-paste into ChatGPT web
    full_prompt = f"{SYSTEM_PROMPT}\n\nQuestion: {question}\n\nContext:\n{context}\n\nAnswer:"
    print("\n--- Copy this into ChatGPT ---\n")
    print(full_prompt)

    # Rough, pre-call estimate (replace model/rates in cost_estimator.py to be accurate)
    model_name = cfg["model"].get("chat", "gpt-4o-mini")
    max_output_tokens = 512  # adjust to your cap
    print(estimate_and_format(full_prompt, model_name, max_output_tokens))

    # msg = PROMPT_TMPL.invoke({"question": question, "context": context})
    # llm = _build_llm(chat_model, temperature=temperature)
    # res = llm.invoke(msg.to_messages())
    # return here since we're in print-and-copy debug mode
    return ""


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]).strip() or "What does this repo do?"
    print(ask(q))