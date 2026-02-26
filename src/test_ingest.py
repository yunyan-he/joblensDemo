"""
test_ingest.py
--------------
Isolated smoke-test suite for the incremental indexing logic in src/ingest.py.

Uses dedicated sandbox directories (data/test_kb/ and data/test_chroma/) that
are created fresh before each test and wiped after. This means:

  - Tests never touch the real knowledge_base or chroma_db.
  - No state leak between runs.
  - All assertions use absolute values (no delta hacks).
  - Results are fully deterministic regardless of KB state.

Run:
  source .venv/bin/activate
  python src/test_ingest.py
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
INGEST_PY = ROOT / "src" / "ingest.py"

# Isolated sandbox: completely separate from real data/
SANDBOX_KB     = ROOT / "data" / "_test_kb"
SANDBOX_CHROMA = ROOT / "data" / "_test_chroma"

TEST_FILE_A = SANDBOX_KB / "doc_a.txt"
TEST_FILE_B = SANDBOX_KB / "subdir" / "doc_b.txt"


# ─── Load ingest module with patched constants ────────────────────────────────
def load_ingest_with_sandbox():
    """
    Import src/ingest.py and override its KB_DIR / CHROMA_DIR / HASH_FILE
    to point at our sandbox. This is the clean, engineering-grade approach:
    we test the real code, just pointed at a safe isolated directory.
    """
    spec = importlib.util.spec_from_file_location("ingest", INGEST_PY)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Patch module-level path constants
    mod.KB_DIR     = SANDBOX_KB
    mod.CHROMA_DIR = SANDBOX_CHROMA
    mod.HASH_FILE  = SANDBOX_CHROMA / ".file_hashes.json"
    return mod


# ─── Helpers ─────────────────────────────────────────────────────────────────
def reset_sandbox():
    """Wipe and recreate both sandbox directories from scratch."""
    shutil.rmtree(SANDBOX_KB,     ignore_errors=True)
    shutil.rmtree(SANDBOX_CHROMA, ignore_errors=True)
    SANDBOX_KB.mkdir(parents=True)
    (SANDBOX_KB / "subdir").mkdir()


def read_registry(mod) -> dict[str, str]:
    if mod.HASH_FILE.exists():
        return json.loads(mod.HASH_FILE.read_text(encoding="utf-8"))
    return {}


def log(label: str, ok: bool, detail: str = "") -> bool:
    print(f"{'✅ PASS' if ok else '❌ FAIL'}  {label}")
    if not ok and detail:
        for line in detail.splitlines():
            print("    " + line)
    return ok


# ─── Individual tests ─────────────────────────────────────────────────────────
def test_baseline_empty(mod) -> bool:
    """Fresh sandbox: nothing to index → registry should stay empty."""
    reset_sandbox()
    # KB has no supported files yet → ensure_knowledge_base should sys.exit.
    # We catch SystemExit gracefully.
    try:
        mod.incremental_update(mod.KB_DIR)
    except SystemExit:
        pass  # expected: KB is empty, script exits with message
    reg = read_registry(mod)
    ok  = len(reg) == 0
    return log("Empty KB → no registry created", ok)


def test_new_files(mod) -> bool:
    """Add 2 files → both appear as New in Chroma, both hashes written to registry."""
    reset_sandbox()
    TEST_FILE_A.write_text("Python FastAPI machine learning.\nPython机器学习接口。\n", encoding="utf-8")
    TEST_FILE_B.write_text("Docker Kubernetes CI/CD pipelines.\n容器化与持续集成。\n", encoding="utf-8")

    mod.incremental_update(mod.KB_DIR)
    reg = read_registry(mod)

    ok = (
        len(reg) == 2
        and "doc_a.txt" in reg
        and str(Path("subdir") / "doc_b.txt") in reg
    )
    return log("2 new files → registry has 2 entries with correct keys", ok,
               f"Registry keys: {list(reg.keys())}")


def test_unchanged(mod) -> bool:
    """
    Run again without touching any file → nothing re-embedded.
    Verified by checking the registry hashes haven't changed.
    """
    reg_before = read_registry(mod)
    mod.incremental_update(mod.KB_DIR)
    reg_after  = read_registry(mod)

    ok = reg_before == reg_after
    return log("Second run, no file changes → registry identical (nothing re-embedded)", ok,
               f"Before: {reg_before}\nAfter:  {reg_after}")


def test_modified_file(mod) -> bool:
    """Overwrite doc_a.txt → its hash must change in the registry."""
    hash_before = read_registry(mod).get("doc_a.txt")
    time.sleep(0.05)
    TEST_FILE_A.write_text("UPDATED: Rust, WebAssembly, edge computing.\n更新：边缘计算与WebAssembly。\n", encoding="utf-8")

    mod.incremental_update(mod.KB_DIR)
    hash_after = read_registry(mod).get("doc_a.txt")

    ok = hash_before != hash_after and hash_after is not None
    return log("Modified doc_a.txt → registry hash updated", ok,
               f"Before: {hash_before}\nAfter:  {hash_after}")


def test_deleted_file(mod) -> bool:
    """Delete doc_a.txt → registry should only contain doc_b.txt."""
    TEST_FILE_A.unlink()
    mod.incremental_update(mod.KB_DIR)
    reg = read_registry(mod)

    ok = (
        "doc_a.txt" not in reg
        and str(Path("subdir") / "doc_b.txt") in reg
        and len(reg) == 1
    )
    return log("Deleted doc_a.txt → removed from registry, doc_b still present", ok,
               f"Registry: {reg}")


def test_query(mod) -> bool:
    """Query the sandbox Chroma store and verify we get a result back."""
    embeddings = mod.get_embeddings()
    store      = mod.open_or_create_store(embeddings, mod.CHROMA_DIR)
    results    = store.similarity_search("container pipeline Docker", k=1)

    ok = len(results) > 0 and "doc_b.txt" in results[0].metadata.get("source", "")
    return log("--query on sandbox store → returns doc_b.txt chunk", ok,
               f"Got: {[r.metadata.get('source') for r in results]}")


# ─── Runner ───────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("  ingest.py — Isolated Incremental Indexing Tests")
    print("=" * 60)

    mod = load_ingest_with_sandbox()

    tests = [
        test_baseline_empty,
        test_new_files,
        test_unchanged,
        test_modified_file,
        test_deleted_file,
        test_query,
    ]

    results = []
    for i, test_fn in enumerate(tests, 1):
        print(f"\n[{i}/{len(tests)}] {test_fn.__doc__.strip().splitlines()[0]} …")
        try:
            results.append(test_fn(mod))
        except Exception as exc:
            print(f"❌ FAIL  Exception: {exc}")
            results.append(False)

    # Clean up sandbox after all tests
    shutil.rmtree(SANDBOX_KB,     ignore_errors=True)
    shutil.rmtree(SANDBOX_CHROMA, ignore_errors=True)
    print("\n  (Sandbox directories cleaned up.)")

    passed = sum(results)
    total  = len(results)
    print("\n" + "=" * 60)
    print(f"  Result: {passed}/{total} tests passed")
    print("=" * 60)
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
