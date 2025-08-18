# cost_estimator.py
from __future__ import annotations

from typing import Dict

try:
    import tiktoken  # type: ignore
except Exception as e:  # pragma: no cover
    tiktoken = None
    _IMPORT_ERR = e
else:
    _IMPORT_ERR = None


# Map models to encodings (fallback to encoding_for_model if not listed)
_MODEL_TO_ENCODING: Dict[str, str] = {
    # "gpt-4o": "o200k_base",
    # "gpt-4o-mini": "o200k_base",
    # "gpt-3.5-turbo": "cl100k_base",
}

# USD per 1K tokens (fill with your actual rates for accurate estimates)
PRICING_PER_1K: Dict[str, Dict[str, float]] = {
    # Example placeholders; replace with real numbers for your model:
    # "gpt-4o-mini": {"input": 0.005, "output": 0.015},
    # "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
}


def _get_encoder(model: str):
    if tiktoken is None:
        raise RuntimeError(
            "tiktoken not installed. Please `pip install tiktoken`."
        ) from _IMPORT_ERR
    enc_name = _MODEL_TO_ENCODING.get(model)
    if enc_name:
        return tiktoken.get_encoding(enc_name)
    # Best-effort: ask tiktoken; fallback to cl100k_base
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model: str) -> int:
    """Return exact token count for `text` under `model`'s encoding."""
    enc = _get_encoder(model)
    return len(enc.encode(text))


def estimate_cost_usd(input_tokens: int, output_tokens: int, model: str) -> float:
    """Estimate cost using PRICING_PER_1K[model] with separate input/output rates."""
    rates = PRICING_PER_1K.get(model)
    if not rates:
        # Unknown model pricing: return 0.0 so caller can decide how to handle
        return 0.0
    return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1000.0


def estimate_and_format(prompt_text: str, model: str, max_output_tokens: int) -> str:
    """
    Convenience helper: count prompt tokens and print a one-line estimate.
    Use before the LLM call; pass actual output tokens later for a final line.
    """
    inp = count_tokens(prompt_text, model)
    est = estimate_cost_usd(inp, max_output_tokens, model)
    return (
        f"[TokenEstimate] model={model} "
        f"input_tokens={inp} "
        f"max_output_tokens={max_output_tokens} "
        f"est_cost_usd~{est:.6f}"
    )