"""
ingest.py

LangChain-first repository ingestion for the Codebase Q&A RAG app.

- Loads .py/.md files from a target repo using LangChain loaders
- Splits into code-aware chunks with line-number metadata
- Embeds with HuggingFaceEmbeddings and persists to Chroma

Requirements:
    pip install langchain langchain-community langchain-text-splitters sentence-transformers chromadb
"""
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

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
# Lightweight Doc wrapper
# ----------------------------
class Doc:
    def __init__(self, page_content: str, metadata: Optional[Dict[str, Any]] = None):
        self.page_content = page_content
        self.metadata = metadata or {}

# ----------------------------
# Ingestion steps (LangChain-only)
# ----------------------------

def load_repo_docs(repo_path: str, file_globs: List[str], exclude_globs: List[str]) -> List[Doc]:
    """Discover and load files from `repo_path` using LangChain's TextLoader."""
    import fnmatch
    from langchain_community.document_loaders import TextLoader  # type: ignore

    root = Path(repo_path).resolve()
    if not root.exists():
        raise FileNotFoundError(f"[INGEST] repo_path does not exist: {root}")

    # Gather candidate files
    candidates: List[Path] = []
    for g in file_globs:
        candidates.extend(root.glob(g))
    candidates = [p for p in set(candidates) if p.is_file()]

    # Filter out excluded paths
    filtered: List[Path] = []
    for p in candidates:
        rel = p.relative_to(root).as_posix()
        if any(fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(p.as_posix(), pat) for pat in exclude_globs):
            continue
        filtered.append(p)

    logger.info("[INGEST] Found %d files after filtering", len(filtered))

    docs: List[Doc] = []
    for path in filtered:
        loader = TextLoader(path.as_posix(), autodetect_encoding=True, encoding=None)
        for d in loader.load():
            meta = dict(d.metadata or {})
            meta["source"] = path.as_posix()
            # rough whole-file span; chunking will assign precise ranges
            try:
                with open(path, "r", encoding=meta.get("encoding", "utf-8"), errors="ignore") as fh:
                    line_count = sum(1 for _ in fh)
                meta["loc"] = f"1-{line_count}"
            except Exception:
                pass
            docs.append(Doc(page_content=d.page_content, metadata=meta))

    logger.info("[INGEST] Loaded %d documents", len(docs))
    return docs


def split_docs(docs: List[Doc], chunk_size: int, chunk_overlap: int, language: str = "python") -> List[Doc]:
    """Split documents into code-aware chunks and compute accurate line ranges."""
    from langchain_text_splitters import Language, RecursiveCharacterTextSplitter  # type: ignore
    from langchain.schema import Document as LCDocument  # type: ignore

    logger.info(
        "[INGEST] Splitting %d docs into chunks (size=%d, overlap=%d, lang=%s)",
        len(docs), chunk_size, chunk_overlap, language,
    )
    if not docs:
        return []

    # group by file path
    by_source: Dict[str, List[Doc]] = {}
    for d in docs:
        src = d.metadata.get("source", "<unknown>")
        by_source.setdefault(src, []).append(d)

    # pick language enum
    try:
        lang_enum = getattr(Language, language.upper())
    except Exception:
        lang_enum = Language.PYTHON

    splitter = RecursiveCharacterTextSplitter.from_language(
        language=lang_enum,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    out: List[Doc] = []
    for src, doc_list in by_source.items():
        original_text = "\n\n".join(d.page_content for d in doc_list)
        lc_docs = [LCDocument(page_content=original_text, metadata={"source": src})]
        lc_chunks = splitter.split_documents(lc_docs)

        # map chunks back to line ranges using a moving cursor
        cursor = 0
        for ch in lc_chunks:
            content = ch.page_content
            idx = original_text.find(content, cursor)
            if idx == -1:
                idx = original_text.find(content)
            if idx == -1:
                out.append(Doc(page_content=content, metadata={"source": src}))
                continue
            start_char = idx
            end_char = idx + len(content)
            start_line = original_text.count("\n", 0, start_char) + 1
            end_line = original_text.count("\n", 0, end_char) + 1
            out.append(Doc(page_content=content, metadata={"source": src, "loc": f"{start_line}-{end_line}"}))
            cursor = end_char

    logger.info("[INGEST] Produced %d chunks", len(out))
    return out


def embed_and_persist(chunks: List[Doc], index_path: str, embedding_model: str) -> None:
    """Embed chunks with HuggingFaceEmbeddings and persist to Chroma via LangChain."""
    from langchain_community.vectorstores import Chroma  # type: ignore
    from langchain.embeddings import HuggingFaceEmbeddings  # type: ignore

    Path(index_path).mkdir(parents=True, exist_ok=True)
    texts = [d.page_content for d in chunks]
    metadatas = [d.metadata for d in chunks]

    logger.info("[INGEST] Using LangChain Chroma with HuggingFaceEmbeddings '%s'", embedding_model)
    emb = HuggingFaceEmbeddings(model_name=embedding_model)

    # from_texts persists when persist_directory is provided
    _ = Chroma.from_texts(
        texts=texts,
        embedding=emb,
        metadatas=metadatas,
        persist_directory=index_path,
        collection_name="codeqa",
    )
    logger.info("[INGEST] Persisted %d chunks to Chroma at %s", len(texts), index_path)

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
        merged = {**DEFAULT_CONFIG, **cfg}
    else:
        merged = DEFAULT_CONFIG.copy()

    merged.setdefault("chunk", {}).setdefault("size", DEFAULT_CONFIG["chunk"]["size"])
    merged.setdefault("chunk", {}).setdefault("overlap", DEFAULT_CONFIG["chunk"]["overlap"])
    merged.setdefault("chunk", {}).setdefault("language", DEFAULT_CONFIG["chunk"]["language"])
    return merged


def build_index(config_path: Optional[str] = None) -> None:
    """End-to-end ingestion: load → split → embed → persist."""
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

    logger.info("[INGEST] Done.")


if __name__ == "__main__":
    # Allow optional CONFIG env var override: CONFIG=./config.yaml python ingest.py
    config_env = os.environ.get("CONFIG")
    build_index(config_env)