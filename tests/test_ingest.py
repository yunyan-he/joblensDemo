"""
tests/test_ingest.py
--------------------
Pytest suite for the incremental indexing CLI (src/ingest.py).

Each test that writes to ChromaDB invokes ingest.py as a subprocess with
JOBLENS_KB_DIR / JOBLENS_CHROMA_DIR env vars pointing at an isolated sandbox.
This guarantees that Chroma's SQLite connection is never shared between tests
(which would cause SQLITE_READONLY_DBMOVED errors in a single process).

Pure-function tests (hash, registry helpers) are called directly.

Run:
  pytest tests/
  pytest tests/test_ingest.py -v
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT      = Path(__file__).resolve().parent.parent
INGEST_PY = ROOT / "src" / "ingest.py"
PYTHON    = sys.executable

SANDBOX_KB     = ROOT / "data" / "_test_kb"
SANDBOX_CHROMA = ROOT / "data" / "_test_chroma"


# ── Helpers ────────────────────────────────────────────────────────────────────
def run_ingest(*extra_args: str) -> subprocess.CompletedProcess:
    """Run ingest.py as a subprocess with sandbox env vars."""
    env = {**os.environ,
           "JOBLENS_KB_DIR":     str(SANDBOX_KB),
           "JOBLENS_CHROMA_DIR": str(SANDBOX_CHROMA)}
    return subprocess.run(
        [PYTHON, str(INGEST_PY), *extra_args],
        capture_output=True, text=True, env=env,
    )


def read_registry() -> dict[str, str]:
    reg_file = SANDBOX_CHROMA / ".file_hashes.json"
    return json.loads(reg_file.read_text(encoding="utf-8")) if reg_file.exists() else {}


# ── Direct import for pure-function tests ──────────────────────────────────────
def _load_ingest():
    spec = importlib.util.spec_from_file_location("ingest_pure", INGEST_PY)
    m    = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

_mod = _load_ingest()


# ── Fixtures ───────────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def sandbox():
    """Wipe and recreate sandbox dirs before every test; clean up after."""
    shutil.rmtree(SANDBOX_KB,     ignore_errors=True)
    shutil.rmtree(SANDBOX_CHROMA, ignore_errors=True)
    SANDBOX_KB.mkdir(parents=True)
    (SANDBOX_KB / "subdir").mkdir()
    yield
    shutil.rmtree(SANDBOX_KB,     ignore_errors=True)
    shutil.rmtree(SANDBOX_CHROMA, ignore_errors=True)


# ── Unit tests (no subprocess, no Chroma) ─────────────────────────────────────
def test_compute_md5_same_content_same_hash():
    """Same content → same MD5; different content → different MD5."""
    f1 = SANDBOX_KB / "f1.txt"
    f2 = SANDBOX_KB / "f2.txt"
    f1.write_text("hello", encoding="utf-8")
    f2.write_text("world", encoding="utf-8")
    assert _mod.compute_md5(f1) == _mod.compute_md5(f1)
    assert _mod.compute_md5(f1) != _mod.compute_md5(f2)


def test_hash_registry_round_trip():
    """save_hash_registry → load_hash_registry → identical dict."""
    SANDBOX_CHROMA.mkdir(parents=True, exist_ok=True)

    original = _mod.HASH_FILE
    _mod.HASH_FILE = SANDBOX_CHROMA / ".file_hashes.json"
    try:
        payload = {"a.txt": "abc123", "sub/b.pdf": "def456"}
        _mod.save_hash_registry(payload)
        assert _mod.load_hash_registry() == payload
    finally:
        _mod.HASH_FILE = original


# ── Integration tests (subprocess → fresh Chroma per test) ────────────────────
def test_new_files_detected():
    """2 new files → registry has 2 entries with correct relative-path keys."""
    (SANDBOX_KB / "doc_a.txt").write_text("Python FastAPI ML.\n", encoding="utf-8")
    (SANDBOX_KB / "subdir" / "doc_b.txt").write_text("Docker K8s CI/CD.\n", encoding="utf-8")

    result = run_ingest()
    reg = read_registry()

    assert "doc_a.txt" in reg, f"stdout:\n{result.stdout}"
    assert str(Path("subdir") / "doc_b.txt") in reg
    assert len(reg) == 2


def test_unchanged_skipped():
    """Second run with no file changes → registry unchanged & '▸ Unchanged' > 0."""
    (SANDBOX_KB / "doc_a.txt").write_text("Some content.\n", encoding="utf-8")
    run_ingest()                   # first: index
    reg_before = read_registry()

    result = run_ingest()          # second: nothing changed
    reg_after  = read_registry()

    assert reg_before == reg_after
    assert "Unchanged" in result.stdout


def test_modified_file_rehashed():
    """Overwrite a file between two runs → its hash changes in registry."""
    file_a = SANDBOX_KB / "doc_a.txt"
    file_a.write_text("Original.\n", encoding="utf-8")
    run_ingest()
    hash_before = read_registry().get("doc_a.txt")
    assert hash_before is not None

    file_a.write_text("Updated: Rust, WASM.\n", encoding="utf-8")
    run_ingest()

    assert read_registry().get("doc_a.txt") != hash_before


def test_deleted_file_removed_from_registry():
    """Delete doc_a between runs → removed from registry; doc_b remains."""
    (SANDBOX_KB / "doc_a.txt").write_text("File A.\n", encoding="utf-8")
    (SANDBOX_KB / "subdir" / "doc_b.txt").write_text("File B.\n", encoding="utf-8")
    run_ingest()

    (SANDBOX_KB / "doc_a.txt").unlink()
    run_ingest()
    reg = read_registry()

    assert "doc_a.txt" not in reg
    assert str(Path("subdir") / "doc_b.txt") in reg
    assert len(reg) == 1


def test_query_mode_returns_result():
    """--query mode against the sandbox store returns at least 1 result."""
    (SANDBOX_KB / "subdir" / "doc_b.txt").write_text(
        "Docker container pipeline Kubernetes deployment CI/CD.\n"
        "容器化部署流水线。\n",
        encoding="utf-8",
    )
    run_ingest()   # build the index first

    result = run_ingest("--query", "container pipeline", "--top-k", "1")
    assert "[1]" in result.stdout, f"No result block found.\nstdout:\n{result.stdout}"
