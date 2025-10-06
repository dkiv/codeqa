"""
privacy.py

Request-time pseudonymization utilities for scrubbing sensitive entities from
retrieved context and questions before calling the LLM. Provides reversible
replacement (pseudonymization) so the final answer can be decoded.

Design goals:
- Zero hard dependency on external NLP packages; optional spaCy NER if available.
- Regex coverage for structured PII (emails, phones, SSN, credit cards, dates, addresses).
- Deterministic, consistent placeholders per request (e.g., Person_1, Org_2).
- Fast and safe: single-pass span replacement (no index drift), skip overlaps.

Note: This is request-scoped pseudonymization. Embeddings and the vector store
still contain the original content. For ingest-time pseudonymization, a separate
flow and stable corpus-wide mapping would be needed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple


# ----------------------------
# Optional NER (spaCy)
# ----------------------------
try:  # pragma: no cover - optional dependency
    import spacy  # type: ignore
except Exception:  # pragma: no cover
    spacy = None  # type: ignore
_NLP = None  # lazily loaded on first use via SPACY_MODEL env or default


# ----------------------------
# Regex patterns for structured PII
# ----------------------------
_EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
_PHONE_RE = re.compile(
    r"\b(?:\+?1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)?\d{3}[\s.-]?\d{4}\b"
)
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_DATE_NUMERIC_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
_DATE_TEXTUAL_RE = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|"
    r"Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{2,4}\b",
    re.IGNORECASE,
)
_CREDIT_CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
_POSTAL_CA_RE = re.compile(
    r"\b[ABCEGHJ-NPRSTVXY]\d[ABCEGHJ-NPRSTV-Z][ -]?\d[ABCEGHJ-NPRSTV-Z]\d\b",
    re.IGNORECASE,
)
_ZIP_US_RE = re.compile(r"\b\d{5}(?:-\d{4})?\b")
_ADDRESS_RE = re.compile(
    r"\b\d{1,6}\s+(?:[A-Za-z0-9\.\-]+\s+){1,6}"
    r"(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Lane|Ln|Drive|Dr|Court|Ct|Way|Pl|Place)"
    r"(?:\s+(?:N|S|E|W|North|South|East|West))?"
    r"(?:[\s,]+(?:Suite|Ste|Unit|Apt|Apartment|#)\s*[A-Za-z0-9\-]+)?\b",
    re.IGNORECASE,
)
_URL_RE = re.compile(r"\bhttps?://[^\s>\]]+\b", re.IGNORECASE)


def _luhn_ok(num: str) -> bool:
    s = [c for c in num if c.isdigit()]
    if not (13 <= len(s) <= 19):
        return False
    digits = list(map(int, s))
    checksum = 0
    parity = (len(digits) - 2) % 2
    for i, d in enumerate(digits):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


STRUCTURED_PATTERNS: Dict[str, re.Pattern] = {
    "EMAIL": _EMAIL_RE,
    "PHONE": _PHONE_RE,
    "SSN": _SSN_RE,
    "DATE": _DATE_NUMERIC_RE,
    "DATE_TEXT": _DATE_TEXTUAL_RE,  # treated as DATE type
    "CREDIT_CARD": _CREDIT_CARD_RE,
    "POSTAL_CODE_CA": _POSTAL_CA_RE,
    "POSTAL_CODE_US": _ZIP_US_RE,
    "ADDRESS": _ADDRESS_RE,
    "URL": _URL_RE,  # not strictly PII, but may be sensitive
}


DEFAULT_ENTITY_TYPES: Set[str] = {
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
}


@dataclass
class PrivacyConfig:
    request_time: bool = True
    enable_regex: bool = True
    enable_ner: bool = False
    entity_types: Set[str] = field(default_factory=lambda: set(DEFAULT_ENTITY_TYPES))


class RequestPseudonymizer:
    """Per-request pseudonymizer that creates reversible replacements for detected entities."""

    def __init__(self, config: Optional[PrivacyConfig] = None):
        self.config = config or PrivacyConfig()
        self.value_to_token: Dict[str, str] = {}
        self.token_to_value: Dict[str, str] = {}
        self._counters: Dict[str, int] = {}

    # ---------- public API ----------
    def build_mapping(self, texts: Sequence[str]) -> None:
        """Scan texts, detect entities, and build a stable mapping for this request."""
        if not self.config.request_time:
            return
        seen: List[Tuple[str, str]] = []  # (value, label)
        for t in texts:
            if self.config.enable_regex:
                for val, label in self._detect_structured(t):
                    # Normalize label variants
                    if label == "DATE_TEXT":
                        norm_label = "DATE"
                    elif label.startswith("POSTAL_CODE"):
                        norm_label = "POSTAL_CODE"
                    else:
                        norm_label = label
                    if norm_label not in self.config.entity_types:
                        continue
                    seen.append((val, norm_label))
            if self.config.enable_ner:
                for val, label in self._detect_ner(t):
                    if label in self.config.entity_types:
                        seen.append((val, label))

        for val, label in seen:
            if val not in self.value_to_token:
                token = self._next_token(label)
                self.value_to_token[val] = token
                self.token_to_value[token] = val

    def apply(self, text: str) -> str:
        """Replace detected values with tokens. Requires build_mapping() first."""
        if not self.config.request_time or not self.value_to_token:
            return text

        # Build spans for all matches using current mapping; then rewrite once.
        matches: List[Tuple[int, int, str]] = []  # (start, end, replacement)
        lowered_keys = {k.lower(): k for k in self.value_to_token.keys()}

        # Use regex search per key for precise positions; prefer longest first to avoid partial overlaps.
        sorted_vals = sorted(self.value_to_token.keys(), key=len, reverse=True)
        for val in sorted_vals:
            escaped = re.escape(val)
            for m in re.finditer(escaped, text):
                matches.append((m.start(), m.end(), self.value_to_token[val]))

        # Deduplicate and skip overlaps
        matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))
        out: List[str] = []
        idx = 0
        last_end = 0
        for start, end, repl in matches:
            if start < last_end:  # overlap; skip
                continue
            if last_end < start:
                out.append(text[last_end:start])
            out.append(repl)
            last_end = end
        out.append(text[last_end:])
        return "".join(out)

    def decode(self, text: str) -> str:
        if not self.config.request_time or not self.token_to_value:
            return text
        # Tokens are alnum+underscore; do a quick pass replacing tokens back
        # Build a single pattern for all tokens (longest first)
        tokens = sorted(self.token_to_value.keys(), key=len, reverse=True)
        if not tokens:
            return text
        pattern = re.compile("|".join(re.escape(t) for t in tokens))

        def _sub(m: re.Match) -> str:
            tok = m.group(0)
            return self.token_to_value.get(tok, tok)

        return pattern.sub(_sub, text)

    # ---------- detection helpers ----------
    def _detect_structured(self, text: str) -> List[Tuple[str, str]]:
        items: List[Tuple[str, str]] = []
        for label, pat in STRUCTURED_PATTERNS.items():
            for m in pat.finditer(text):
                s = m.group(0)
                if label == "CREDIT_CARD":
                    # Reduce false positives using Luhn
                    if not _luhn_ok(s):
                        continue
                items.append((s, label))
        return items

    def _detect_ner(self, text: str) -> List[Tuple[str, str]]:
        global _NLP
        if _NLP is None and spacy is not None:
            import os
            model_name = os.getenv("SPACY_MODEL", "en_core_web_sm")
            try:
                _NLP = spacy.load(model_name)
            except Exception:
                return []
        if _NLP is None:
            return []
        doc = _NLP(text)
        out: List[Tuple[str, str]] = []
        for ent in doc.ents:
            if ent.label_ in {"PERSON", "ORG", "GPE"}:
                out.append((ent.text, ent.label_))
        return out

    def _next_token(self, label: str) -> str:
        c = self._counters.get(label, 0) + 1
        self._counters[label] = c
        # Normalize label to title-case segment for readability
        base = label.title().replace("_", "")
        return f"{base}_{c}"


def build_request_pseudonymizer(
    texts: Sequence[str],
    enable_regex: bool = True,
    enable_ner: bool = False,
    entity_types: Optional[Set[str]] = None,
) -> RequestPseudonymizer:
    cfg = PrivacyConfig(
        request_time=True,
        enable_regex=enable_regex,
        enable_ner=enable_ner,
        entity_types=set(entity_types or DEFAULT_ENTITY_TYPES),
    )
    if cfg.enable_ner:
        # Enforce spaCy availability when NER is requested
        if spacy is None:
            raise RuntimeError(
                "privacy.enable_ner=true but spaCy is not installed. Install it (pip install spacy) and a model (e.g., en_core_web_sm)."
            )
        import os
        model_name = os.getenv("SPACY_MODEL", "en_core_web_sm")
        global _NLP
        if _NLP is None:
            try:
                _NLP = spacy.load(model_name)
            except Exception as e:
                raise RuntimeError(
                    f"privacy.enable_ner=true but spaCy model '{model_name}' is not available. Install it via: python -m spacy download {model_name}"
                ) from e
    p = RequestPseudonymizer(cfg)
    p.build_mapping(texts)
    return p
