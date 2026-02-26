"""
test_compare.py
---------------
Unit tests for the non-LLM components of src/compare.py.
Tests RAG retrieval, prompt construction, and JSON parsing
without making any real LLM API calls.

Run:
  source .venv/bin/activate
  python src/test_compare.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# ── Load compare module ────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
COMPARE_PY  = ROOT / "src" / "compare.py"

spec = importlib.util.spec_from_file_location("compare", COMPARE_PY)
mod  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


# ── Helpers ───────────────────────────────────────────────────────────────────
def log(label: str, ok: bool, detail: str = "") -> bool:
    print(f"{'✅ PASS' if ok else '❌ FAIL'}  {label}")
    if not ok and detail:
        for line in detail.splitlines():
            print("    " + line)
    return ok


# ── Tests ─────────────────────────────────────────────────────────────────────
def test_rag_retrieval() -> bool:
    """retrieve_kb_context() returns 3 formatted chunks from real ChromaDB."""
    ctx = mod.retrieve_kb_context("Python backend API machine learning开发工程师", top_k=3)
    ok  = (
        ctx != ""
        and ctx.count("[Chunk") == 3
        and "[Chunk 1" in ctx
        and "[Chunk 3" in ctx
        and "source:" in ctx
    )
    return log(
        "RAG retrieval → 3 chunks returned with [Chunk N] and source tags",
        ok,
        f"First 300 chars:\n{ctx[:300]}" if not ok else "",
    )


def test_rag_graceful_no_db() -> bool:
    """retrieve_kb_context() returns '' and does not crash when DB is missing."""
    original_dir = mod.CHROMA_DIR
    mod.CHROMA_DIR = Path("/nonexistent/path/chroma_db")
    try:
        ctx = mod.retrieve_kb_context("some query")
        ok  = ctx == ""
    finally:
        mod.CHROMA_DIR = original_dir
    return log("Missing ChromaDB → returns empty string, no crash", ok)


def test_build_user_message_with_context() -> bool:
    """build_user_message injects both CV and retrieved context correctly."""
    msg = mod.build_user_message(
        jd_text="We need a Python engineer.",
        cv_text="I am a Python developer.",
        kb_context="[Chunk 1 — source: resume.pdf]\nBuilt FastAPI service.",
    )
    ok = (
        "## Job Description" in msg
        and "## Candidate Resume" in msg
        and "Retrieved Context from Knowledge Base" in msg
        and "[Chunk 1" in msg
        and "I am a Python developer." in msg
        and "We need a Python engineer." in msg
    )
    return log("build_user_message with context → all 3 sections present", ok, msg if not ok else "")


def test_build_user_message_no_context() -> bool:
    """When kb_context is empty, cv_text is used directly (no RAG header)."""
    msg = mod.build_user_message(
        jd_text="Senior Go engineer needed.",
        cv_text="I write Go services.",
        kb_context="",
    )
    ok = (
        "Retrieved Context" not in msg
        and "I write Go services." in msg
        and "Senior Go engineer needed." in msg
    )
    return log("build_user_message without context → no 'Retrieved Context' header", ok)


def test_parse_json_clean() -> bool:
    """parse_json_response handles clean JSON."""
    payload = {"overall_match_score": 85, "hiring_recommendation": "YES"}
    result  = mod.parse_json_response(json.dumps(payload))
    ok      = result == payload
    return log("parse_json_response on clean JSON → dict matches", ok, str(result) if not ok else "")


def test_parse_json_with_fences() -> bool:
    """parse_json_response strips ```json ... ``` markdown fences."""
    payload = {"overall_match_score": 72}
    raw     = f"```json\n{json.dumps(payload)}\n```"
    result  = mod.parse_json_response(raw)
    ok      = result == payload
    return log("parse_json_response strips ```json fences → dict matches", ok)


def test_llm_call_mock() -> bool:
    """
    call_llm passes system prompt + user message to the OpenAI client.
    Mocked so no real API call is made.
    """
    fake_response = MagicMock()
    fake_response.choices[0].message.content = '{"overall_match_score": 90}'

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = fake_response

    result = mod.call_llm(mock_client, "gpt-4o", "test user message")

    call_kwargs = mock_client.chat.completions.create.call_args
    messages    = call_kwargs.kwargs.get("messages", call_kwargs.args[0] if call_kwargs.args else [])
    roles       = [m["role"] for m in messages]

    ok = (
        result == '{"overall_match_score": 90}'
        and "system" in roles
        and "user"   in roles
    )
    return log("call_llm (mocked) → sends system+user messages, returns content", ok)


# ── Runner ────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("  compare.py — Agentic RAG Component Tests")
    print("=" * 60)

    tests = [
        test_rag_retrieval,
        test_rag_graceful_no_db,
        test_build_user_message_with_context,
        test_build_user_message_no_context,
        test_parse_json_clean,
        test_parse_json_with_fences,
        test_llm_call_mock,
    ]

    results = []
    for i, fn in enumerate(tests, 1):
        print(f"\n[{i}/{len(tests)}] {fn.__doc__.strip().splitlines()[0]} …")
        try:
            results.append(fn())
        except Exception as exc:
            print(f"❌ FAIL  Exception: {exc}")
            results.append(False)

    passed = sum(results)
    total  = len(results)
    print("\n" + "=" * 60)
    print(f"  Result: {passed}/{total} tests passed")
    print("=" * 60)
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
