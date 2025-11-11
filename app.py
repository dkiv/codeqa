"""
app.py

Lightweight Flask UI for the Codebase Q&A assistant.

Features:
- Simple form to submit a question
- Displays either the model answer (if app.use_openai_api=true) or a copy-paste prompt
- Shows token estimate lines printed by the chain

Run:
    python app.py

Then open http://localhost:5000
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from typing import Dict, Optional

from flask import Flask, render_template, request

# We use the existing chain as-is and capture its printed prompt/estimates.
import rag_chain


def _run_question(question: str) -> Dict[str, Optional[str]]:
    """Call rag_chain.ask while capturing printed output for UI display.

    Returns a dict with keys: question, answer, prompt_text, token_lines (str).
    """
    buf = io.StringIO()
    with redirect_stdout(buf):
        answer = rag_chain.ask(question)
    captured = buf.getvalue() or ""

    # Parse captured output to extract the prompt and token lines
    # rag_chain prints a banner line followed by the full prompt, then [TokenEstimate] lines.
    marker = "--- Copy this into ChatGPT ---"
    prompt_text: Optional[str] = None
    token_lines: str = ""

    if marker in captured:
        tail = captured.split(marker, 1)[1].strip()
        # Split off token estimate lines that start with [TokenEstimate]
        lines = tail.splitlines()
        prompt_lines = []
        tokens = []
        for ln in lines:
            if ln.startswith("[TokenEstimate]"):
                tokens.append(ln)
            else:
                prompt_lines.append(ln)
        # Trim trailing blank lines from prompt
        while prompt_lines and not prompt_lines[-1].strip():
            prompt_lines.pop()
        prompt_text = "\n".join(prompt_lines).strip() or None
        token_lines = "\n".join(tokens)

    return {
        "question": question,
        "answer": (answer or None),
        "prompt_text": prompt_text,
        "token_lines": token_lines or None,
    }


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index():  # type: ignore[override]
        return render_template("index.html", result=None)

    @app.post("/ask")
    def ask_route():  # type: ignore[override]
        q = (request.form.get("question") or "").strip()
        if not q:
            return render_template(
                "index.html",
                error="Please enter a question.",
                result=None,
            )
        result = _run_question(q)
        return render_template("index.html", result=result, error=None)

    @app.get("/health")
    def health():  # type: ignore[override]
        return {"status": "ok"}

    return app


if __name__ == "__main__":
    app = create_app()
    # Debug mode off by default; set FLASK_DEBUG=1 to enable hot reload in development
    app.run(host="127.0.0.1", port=5000, debug=False)
