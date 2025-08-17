"""
ingest.py

Handles repository ingestion:
- Loads source files (e.g. .py, .md) from the target repo.
- Splits code/documents into chunks with metadata (file, line numbers).
- Embeds chunks and persists them to a vector database (e.g. Chroma).
"""
"""
ingest.py

Minimal, runnable stub for repository ingestion.
- Loads source files from a target repo (TODO)
- Splits code/documents into chunks (TODO)
- Embeds and persists to a vector DB (TODO)

This version is intentionally dependency-light so it can run without
LangChain installed. Fill in the TODOs as you wire up the real pipeline.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # Optional, we gracefully fall back to defaults


# ----------------------------
# Logging
# ----------------------------
logger = logging.getLogger("codeqa.ingest")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# ----------------------------
# Defaults (overridable via config.yaml)
# ----------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    "repo_path": "data/repo",
    "index_path": "data/index",
    "file_globs": ["**/*.py", "**/*.md"],
    "exclude_globs": [
        "**/.git/**",
        "**/__pycache__/**",
        "**/node_modules/**",
        "**/dist/**",
        "**/build/**",
        "**/.venv/**",
    ],
    "chunk": {"size": 800, "overlap": 120, "language": "python"},
    "model": {"embedding": "BAAI/bge-small-en"},
}


# ----------------------------
# Stub data structures (so the file runs without LangChain)
# ----------------------------
class Doc:
    """Lightweight document placeholder.

    Replace with `langchain.schema.Document` or your preferred type later.
    """

    def __init__(self, page_content: str, metadata: Optional[Dict[str, Any]] = None):
        self.page_content = page_content
        self.metadata = metadata or {}


# ----------------------------
# Stub functions (fill these in later)
# ----------------------------

def load_repo_docs(repo_path: str, file_globs: List[str], exclude_globs: List[str]) -> List[Doc]:
    """Discover and load files from `repo_path` matching `file_globs`.

    TODO:
      - Use a proper loader (e.g., LangChain DirectoryLoader) that returns Documents
      - Read file contents and attach metadata (path, start/end lines)
      - Respect `exclude_globs`
    """
    logger.info("[INGEST] Scanning repo at %s", repo_path)
    # Minimal no-op implementation (returns empty list so script runs)
    # Replace with real loading logic.
    return []


def split_docs(docs: List[Doc], chunk_size: int, chunk_overlap: int, language: str = "python") -> List[Doc]:
    """Split documents into chunks.

    TODO:
      - Use a code-aware splitter (e.g., LangChain RecursiveCharacterTextSplitter.from_language)
      - Add line-number metadata to each chunk for later citations
    """
    logger.info(
        "[INGEST] Splitting %d docs into chunks (size=%d, overlap=%d, lang=%s)",
        len(docs), chunk_size, chunk_overlap, language,
    )
    # Minimal no-op implementation: return input unchanged
    return docs


def embed_and_persist(chunks: List[Doc], index_path: str, embedding_model: str) -> None:
    """Embed chunks and persist to a vector database.

    TODO:
      - Initialize embeddings (e.g., sentence-transformers BAAI/bge-small-en)
      - Create/persist a vector store (e.g., Chroma with `persist_directory=index_path`)
      - Upsert chunk embeddings + metadata
    """
    # Ensure index directory exists even in stub mode so users see side effects
    Path(index_path).mkdir(parents=True, exist_ok=True)
    logger.info(
        "[INGEST] (stub) Would embed %d chunks with '%s' and persist to '%s'",
        len(chunks), embedding_model, index_path,
    )


# ----------------------------
# Orchestration
# ----------------------------

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load YAML config if available; otherwise return DEFAULT_CONFIG."""
    if not config_path:
        # Look for config.yaml alongside this file or project root
        here = Path(__file__).resolve().parent
        candidates = [here / "config.yaml", here.parent / "config.yaml"]
        for cand in candidates:
            if cand.exists():
                config_path = str(cand)
                break

    if config_path and Path(config_path).exists() and yaml is not None:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        # Shallow merge on top of defaults
        merged = {**DEFAULT_CONFIG, **cfg}
    else:
        merged = DEFAULT_CONFIG.copy()

    # Normalize some fields to expected types
    merged.setdefault("chunk", {}).setdefault("size", DEFAULT_CONFIG["chunk"]["size"])
    merged.setdefault("chunk", {}).setdefault("overlap", DEFAULT_CONFIG["chunk"]["overlap"])
    merged.setdefault("chunk", {}).setdefault("language", DEFAULT_CONFIG["chunk"]["language"])
    return merged


def build_index(config_path: Optional[str] = None) -> None:
    """End-to-end ingestion: load → split → embed → persist.

    This is a stubbed pipeline that logs each step so you can run it right away.
    Replace internals with real loaders/splitters/vector stores incrementally.
    """
    cfg = load_config(config_path)
    repo_path: str = cfg["repo_path"]
    index_path: str = cfg["index_path"]
    file_globs: List[str] = cfg.get("file_globs", [])
    exclude_globs: List[str] = cfg.get("exclude_globs", [])
    chunk_size: int = cfg["chunk"]["size"]
    chunk_overlap: int = cfg["chunk"]["overlap"]
    language: str = cfg["chunk"].get("language", "python")
    embedding_model: str = cfg["model"]["embedding"]

    logger.info("[INGEST] Starting build_index")
    logger.info("[INGEST] repo_path=%s | index_path=%s", repo_path, index_path)

    docs = load_repo_docs(repo_path, file_globs, exclude_globs)
    chunks = split_docs(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap, language=language)
    embed_and_persist(chunks, index_path=index_path, embedding_model=embedding_model)

    logger.info("[INGEST] Done. (This was a stub run—replace TODOs to enable real indexing.)")


if __name__ == "__main__":
    # Allow optional CONFIG env var override: CONFIG=./config.yaml python ingest.py
    config_env = os.environ.get("CONFIG")
    build_index(config_env)