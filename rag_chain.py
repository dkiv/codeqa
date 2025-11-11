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
try:
    from langchain_core.prompts import ChatPromptTemplate  # langchain>=0.1
except Exception:  # pragma: no cover
    from langchain.prompts import ChatPromptTemplate  # fallback for older versions
from langchain_chroma import Chroma  # type: ignore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI  # type: ignore

from cost_estimator import count_tokens, estimate_and_format, estimate_cost_usd
from privacy import build_request_pseudonymizer  # type: ignore

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
        "enable_regex": True,
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
            "POSTAL_CODE",
            "ADDRESS",
        ],
    },
}

SYSTEM_PROMPT = """
You are a helpful corpus assistant.

Answer the user's question using only the supplied Context snippets. Do not use outside knowledge.

If the Context is insufficient or conflicting, reply 'Insufficient context' and suggest specific places to look in this corpus (file paths, document titles, pages/sections) and refined queries or ingestion steps.

Cite sources inline as [source:locator] using the bracketed labels shown in Context (e.g., [src/service.py:88-121], [contracts/msa.pdf:p12]). If no locator is shown, cite as [source].

Keep answers concise and actionable.
"""
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


def _extract_settings(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key settings from the merged config.

    Returns a flat dict so the call site stays concise.
    """
    k = int(cfg["retrieval"].get("k", 8))
    mmr_fetch_k_default = max(10, k * 4)
    return {
        "index_path": str(cfg["index_path"]),
        "embedding_model": str(cfg["model"]["embedding"]),
        "chat_model": str(cfg["model"].get("chat", "gpt-4o-mini")),
        "temperature": float(cfg["model"].get("temperature", 0.0)),
        "k": k,
        "max_context_chunks": int(cfg["retrieval"].get("max_context_chunks", 6)),
        "max_chunk_chars": int(cfg["retrieval"].get("max_chunk_chars", 1200)),
        "search_type": str(cfg["retrieval"].get("search_type", "similarity")),
        "mmr_fetch_k": int(cfg["retrieval"].get("mmr_fetch_k", mmr_fetch_k_default)),
        "mmr_lambda": float(cfg["retrieval"].get("mmr_lambda", 0.5)),
        "use_openai_api": bool(cfg.get("app", {}).get("use_openai_api", False)),
        "privacy_request_time": bool(cfg.get("privacy", {}).get("request_time", True)),
        "privacy_enable_regex": bool(cfg.get("privacy", {}).get("enable_regex", True)),
        "privacy_enable_ner": bool(cfg.get("privacy", {}).get("enable_ner", False)),
        "privacy_entity_types": set(cfg.get("privacy", {}).get("entity_types", [])),
    }


def _retrieve_docs(retriever: Any, question: str, max_context_chunks: int) -> List[Any]:
    docs = retriever.invoke(question)
    return docs[:max_context_chunks]


def _build_context_and_question(
    docs: List[Any],
    question: str,
    *,
    privacy_request_time: bool,
    privacy_enable_regex: bool,
    privacy_enable_ner: bool,
    privacy_entity_types: Any,
    max_chunk_chars: int,
):
    """Return (context, question_scrubbed, pseudonymizer_or_None)."""
    if privacy_request_time:
        base_texts = [getattr(d, "page_content", "") or "" for d in docs]
        p = build_request_pseudonymizer(
            texts=base_texts + [question],
            enable_regex=privacy_enable_regex,
            enable_ner=privacy_enable_ner,
            entity_types=privacy_entity_types,
        )
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
        return context, question_scrubbed, p
    else:
        context = _format_context(docs, max_chunk_chars=max_chunk_chars)
        return context, question, None


def _make_full_prompt(question_scrubbed: str, context: str) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\nQuestion: {question_scrubbed}\n\nContext:\n{context}\n\nAnswer:"
    )


def _print_pre_call_estimate(full_prompt: str, model_name: str, max_output_tokens: int = 512) -> None:
    try:
        print(estimate_and_format(full_prompt, model_name, max_output_tokens))
    except Exception:
        pass


def _invoke_llm(question_scrubbed: str, context: str, model_name: str, temperature: float):
    msg = PROMPT_TMPL.invoke({"question": question_scrubbed, "context": context})
    llm = _build_llm(model_name, temperature=temperature)
    return llm.invoke(msg.to_messages())


def _print_post_call_cost(full_prompt: str, output_text: str, model_name: str) -> None:
    try:
        prompt_tokens = count_tokens(full_prompt, model_name)
        output_tokens = count_tokens(output_text, model_name)
        final_cost = estimate_cost_usd(prompt_tokens, output_tokens, model_name)
        print(
            f"[TokenEstimate] actual_output_tokens={output_tokens} final_cost_usd~{final_cost:.6f}"
        )
    except Exception:
        pass


def _maybe_decode(answer: str, pseudonymizer: Optional[Any]):
    if pseudonymizer is None:
        return answer
    try:
        return pseudonymizer.decode(answer)
    except Exception:
        return answer


def ask(question: str, config_path: Optional[str] = None) -> str:
    """Answer a question using the local Chroma index.

    Example:
        from rag_chain import ask
        print(ask("Where is update_particle_state used?"))
    """
    cfg = load_config(config_path)
    s = _extract_settings(cfg)

    retriever = _build_retriever(
        s["index_path"],
        s["embedding_model"],
        s["k"],
        search_type=s["search_type"],
        mmr_fetch_k=s["mmr_fetch_k"],
        mmr_lambda=s["mmr_lambda"],
    )
    docs = _retrieve_docs(retriever, question, s["max_context_chunks"])
    if not docs:
        return "I couldn't find relevant context in the index. Try re-ingesting or broadening your query."

    context, question_scrubbed, p = _build_context_and_question(
        docs,
        question,
        privacy_request_time=s["privacy_request_time"],
        privacy_enable_regex=s["privacy_enable_regex"],
        privacy_enable_ner=s["privacy_enable_ner"],
        privacy_entity_types=s["privacy_entity_types"],
        max_chunk_chars=s["max_chunk_chars"],
    )
    # Debug: print the constructed prompt for copy-paste into ChatGPT web
    full_prompt = _make_full_prompt(question_scrubbed, context)
    print("\n--- Copy this into ChatGPT ---\n")
    print(full_prompt)

    # Rough, pre-call estimate (replace model/rates in cost_estimator.py to be accurate)
    model_name = s["chat_model"]
    max_output_tokens = 512  # adjust to your cap
    _print_pre_call_estimate(full_prompt, model_name, max_output_tokens)

    if not s["use_openai_api"]:
        # print-and-copy debug mode (no API calls)
        return ""

    # Build and call the LLM
    res = _invoke_llm(
        question_scrubbed, context, s["chat_model"], temperature=s["temperature"]
    )

    # Optional: refine with actual output tokens
    _print_post_call_cost(full_prompt, getattr(res, "content", str(res)), model_name)

    answer = getattr(res, "content", str(res))
    # Decode the answer back to original values if pseudonymized
    answer = _maybe_decode(answer, p)
    return answer


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]).strip() or "What does this repo do?"
    print(ask(q))
