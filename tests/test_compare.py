"""
tests/test_compare.py
---------------------
Pytest suite for the non-LLM components of src/compare.py.
LLM calls are mocked — no API credits consumed.

Run all tests:
  pytest tests/
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ── Load compare module ────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
COMPARE_PY = ROOT / "src" / "compare.py"

spec = importlib.util.spec_from_file_location("compare", COMPARE_PY)
mod  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


# ── Tests ──────────────────────────────────────────────────────────────────────
def test_rag_retrieval_returns_chunks():
    """retrieve_kb_context() returns 3 formatted chunks from real ChromaDB."""
    ctx = mod.retrieve_kb_context(
        "Python backend API machine learning开发工程师", top_k=3
    )
    assert ctx != ""
    assert ctx.count("[Chunk") == 3
    assert "[Chunk 1" in ctx
    assert "[Chunk 3" in ctx
    assert "source:" in ctx


def test_rag_graceful_no_db():
    """retrieve_kb_context() returns '' without crashing when DB is missing."""
    original = mod.CHROMA_DIR
    mod.CHROMA_DIR = Path("/nonexistent/path/chroma_db")
    try:
        ctx = mod.retrieve_kb_context("some query")
    finally:
        mod.CHROMA_DIR = original
    assert ctx == ""


def test_build_user_message_with_context():
    """build_user_message injects JD, CV, and retrieved context sections."""
    msg = mod.build_user_message(
        jd_text="We need a Python engineer.",
        cv_text="I am a Python developer.",
        kb_context="[Chunk 1 — source: resume.pdf]\nBuilt FastAPI service.",
    )
    assert "## Job Description" in msg
    assert "## Candidate Resume" in msg
    assert "Retrieved Context from Knowledge Base" in msg
    assert "[Chunk 1" in msg
    assert "I am a Python developer." in msg
    assert "We need a Python engineer." in msg


def test_build_user_message_no_context():
    """When kb_context is empty, CV text is used directly — no RAG header."""
    msg = mod.build_user_message(
        jd_text="Senior Go engineer needed.",
        cv_text="I write Go services.",
        kb_context="",
    )
    assert "Retrieved Context" not in msg
    assert "I write Go services." in msg
    assert "Senior Go engineer needed." in msg


def test_parse_json_clean():
    """parse_json_response parses clean JSON correctly."""
    payload = {"overall_match_score": 85, "hiring_recommendation": "YES"}
    assert mod.parse_json_response(json.dumps(payload)) == payload


def test_parse_json_with_fences():
    """parse_json_response strips ```json ... ``` markdown fences."""
    payload = {"overall_match_score": 72}
    raw = f"```json\n{json.dumps(payload)}\n```"
    assert mod.parse_json_response(raw) == payload


def test_llm_call_uses_system_and_user_roles():
    """call_llm sends exactly a system + user message; returns model content."""
    fake_response = MagicMock()
    fake_response.choices[0].message.content = '{"overall_match_score": 90}'

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = fake_response

    result = mod.call_llm(mock_client, "gpt-4o", "test user message")

    call_kwargs = mock_client.chat.completions.create.call_args
    messages = call_kwargs.kwargs.get("messages", [])
    roles = [m["role"] for m in messages]

    assert result == '{"overall_match_score": 90}'
    assert "system" in roles
    assert "user" in roles
