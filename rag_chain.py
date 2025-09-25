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
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml  # type: ignore
from dotenv import find_dotenv, load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma  # type: ignore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI  # type: ignore

from cost_estimator import count_tokens, estimate_and_format, estimate_cost_usd
from privacy import build_request_pseudonymizer, PrivacyConfig  # type: ignore

load_dotenv(find_dotenv())

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
        "search_type": "similarity",
        "mmr_fetch_k": 40,
        "mmr_lambda": 0.5,
    },
    "app": {
        "use_openai_api": False,
    },
    "privacy": {
        "request_time": True,
        "enable_ner": False,
        "entity_types": [
            "PERSON",
            "ORG",
            "GPE",
            "EMAIL",
            "PHONE",
            "SSN",
            "CREDIT_CARD",
            "DATE",
            "ADDRESS",
        ],
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
    """
    Precedence:
      1) DEFAULT_CONFIG
      2) config.yaml (deep-merged)
      3) environment variables (incl. .env via python-dotenv), which win last
    """
    cfg: Dict[str, Any] = DEFAULT_CONFIG.copy()

    # Discover config.yaml if not provided
    if not config_path:
        here = Path(__file__).resolve().parent
        for c in (here / "config.yaml", here.parent / "config.yaml"):
            if c.exists():
                config_path = str(c)
                break

    # Load + deep-merge YAML over defaults
    if config_path and Path(config_path).exists() and yaml is not None:
        with open(config_path, "r") as f:
            file_cfg = yaml.safe_load(f) or {}

        def deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
            for k, v in src.items():
                if isinstance(v, dict) and isinstance(dst.get(k), dict):
                    deep_merge(dst[k], v)  # type: ignore[index]
                else:
                    dst[k] = v
            return dst

        cfg = deep_merge(cfg, file_cfg)

    # ---- Env overrides (after load_dotenv(find_dotenv()) at top of file) ----
    getenv = os.getenv

    # Core paths/models
    cfg["index_path"] = getenv("INDEX_PATH", cfg["index_path"])
    cfg["model"]["embedding"] = getenv("EMBEDDING_MODEL", cfg["model"]["embedding"])
    cfg["model"]["chat"] = getenv("CHAT_MODEL", cfg["model"]["chat"])
    cfg["model"]["temperature"] = float(
        getenv("CHAT_TEMPERATURE", cfg["model"]["temperature"])
    )

    # Retrieval knobs (optional envs)
    cfg.setdefault("retrieval", {})
    cfg["retrieval"]["k"] = int(getenv("RETRIEVAL_K", cfg["retrieval"].get("k", 8)))
    cfg["retrieval"]["max_context_chunks"] = int(
        getenv("MAX_CONTEXT_CHUNKS", cfg["retrieval"].get("max_context_chunks", 6))
    )
    cfg["retrieval"]["max_chunk_chars"] = int(
        getenv("MAX_CHUNK_CHARS", cfg["retrieval"].get("max_chunk_chars", 1200))
    )

    # App flags
    cfg.setdefault("app", {})
    use_api_env = getenv(
        "APP_USE_OPENAI_API", str(cfg["app"].get("use_openai_api", False))
    )
    cfg["app"]["use_openai_api"] = str(use_api_env).lower() in {"1", "true", "yes"}

    return cfg


def _build_retriever(
    index_path: str,
    embedding_model: str,
    k: int,
    search_type: str = "similarity",
    mmr_fetch_k: Optional[int] = None,
    mmr_lambda: float = 0.5,
):
    """Open persisted Chroma and return a retriever. Supports MMR to reduce redundancy."""
    emb = HuggingFaceEmbeddings(model_name=embedding_model)
    vs = Chroma(
        persist_directory=index_path, embedding_function=emb, collection_name="codeqa"
    )
    if search_type == "mmr":
        kwargs = {"k": k}
        if mmr_fetch_k is None:
            mmr_fetch_k = max(10, k * 4)
        kwargs.update({"fetch_k": mmr_fetch_k, "lambda_multiplier": mmr_lambda})
        return vs.as_retriever(search_type="mmr", search_kwargs=kwargs)
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
    if ChatOpenAI is None:
        raise RuntimeError(
            "langchain-openai is not installed. Run `pip install langchain-openai` or set app.use_openai_api=false in config.yaml."
        )
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
    search_type: str = str(cfg["retrieval"].get("search_type", "similarity"))
    mmr_fetch_k: int = int(cfg["retrieval"].get("mmr_fetch_k", max(10, k * 4)))
    mmr_lambda: float = float(cfg["retrieval"].get("mmr_lambda", 0.5))

    use_openai_api: bool = bool(cfg.get("app", {}).get("use_openai_api", False))
    privacy_cfg_raw = cfg.get("privacy", {}) or {}
    privacy_request_time: bool = bool(privacy_cfg_raw.get("request_time", True))
    privacy_enable_ner: bool = bool(privacy_cfg_raw.get("enable_ner", False))
    privacy_entity_types = set(privacy_cfg_raw.get("entity_types", []))

    retriever = _build_retriever(
        index_path,
        embedding_model,
        k,
        search_type=search_type,
        mmr_fetch_k=mmr_fetch_k,
        mmr_lambda=mmr_lambda,
    )
    docs = retriever.invoke(question)[:max_context_chunks]
    if not docs:
        return "I couldn't find relevant context in the index. Try re-ingesting or broadening your query."

    # Build context, optionally scrubbing PII at request time
    if privacy_request_time:
        # Build mapping from retrieved texts + question
        base_texts = [getattr(d, "page_content", "") or "" for d in docs]
        p = build_request_pseudonymizer(
            texts=base_texts + [question],
            enable_ner=privacy_enable_ner,
            entity_types=privacy_entity_types,
        )
        # Scrub question and chunk texts before prompt
        question_scrubbed = p.apply(question)
        parts: List[str] = []
        for d in docs:
            src = d.metadata.get("source", "?")
            loc = d.metadata.get("loc", "?")
            txt = p.apply((d.page_content or "").strip())
            if len(txt) > max_chunk_chars:
                txt = txt[: max_chunk_chars - 3] + "..."
            parts.append(f"[{src}:{loc}]\n{txt}")
        context = "\n\n".join(parts)
    else:
        p = None  # type: ignore
        question_scrubbed = question
        context = _format_context(docs, max_chunk_chars=max_chunk_chars)
    # Debug: print the constructed prompt for copy-paste into ChatGPT web
    full_prompt = (
        f"{SYSTEM_PROMPT}\n\nQuestion: {question_scrubbed}\n\nContext:\n{context}\n\nAnswer:"
    )
    print("\n--- Copy this into ChatGPT ---\n")
    print(full_prompt)

    # Rough, pre-call estimate (replace model/rates in cost_estimator.py to be accurate)
    model_name = cfg["model"].get("chat", "gpt-4o-mini")
    max_output_tokens = 512  # adjust to your cap
    print(estimate_and_format(full_prompt, model_name, max_output_tokens))

    if not use_openai_api:
        # print-and-copy debug mode (no API calls)
        return ""

    # Build and call the LLM
    msg = PROMPT_TMPL.invoke({"question": question_scrubbed, "context": context})
    llm = _build_llm(chat_model, temperature=temperature)
    res = llm.invoke(msg.to_messages())

    # Optional: refine with actual output tokens
    try:
        output_tokens = count_tokens(getattr(res, "content", str(res)), model_name)
        final_cost = estimate_cost_usd(
            count_tokens(full_prompt, model_name), output_tokens, model_name
        )
        print(
            f"[TokenEstimate] actual_output_tokens={output_tokens} final_cost_usd~{final_cost:.6f}"
        )
    except Exception:
        pass

    answer = getattr(res, "content", str(res))
    # Decode the answer back to original values if pseudonymized
    if privacy_request_time and p is not None:
        try:
            answer = p.decode(answer)
        except Exception:
            pass
    return answer


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]).strip() or "What does this repo do?"
    print(ask(q))
