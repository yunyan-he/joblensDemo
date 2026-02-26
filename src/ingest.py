"""
ingest.py
---------
RAG ingestion pipeline for the joblensDemo project.

Modes
-----
  Update (default)  ← smart incremental, only processes changed files
    python src/ingest.py
    Computes MD5 for every file in knowledge_base and compares against a stored
    hash registry. Only new/modified files are re-embedded; deleted files are
    removed from Chroma. Unchanged files are skipped entirely.

  Query only  ← instant, no embedding at all
    python src/ingest.py --query "你想搜索的内容"
    Loads the existing chroma_db and runs similarity search.
    Use --top-k N  to return more than 1 result (default 1).

──────────────────────────────────────────────────────────────────────────────
PREREQUISITES — install inside the project's virtual environment:

  source .venv/bin/activate

  pip install langchain langchain-community langchain-huggingface \
              pypdf chromadb sentence-transformers docx2txt

──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── 1. Project root & Paths ──────────────────────────────────────────────────
# Anchoring to the project root (parent of src/) makes paths work correctly
# regardless of which directory the script is launched from.
ROOT       = Path(__file__).resolve().parent.parent
KB_DIR     = ROOT / "data" / "knowledge_base"   # input  — put your docs here
CHROMA_DIR = ROOT / "data" / "chroma_db"        # output — Chroma persists here

# Hash registry: tracks MD5 of each indexed file to enable incremental updates.
# Stored inside chroma_db so it stays with the vector data and is gitignored.
HASH_FILE  = CHROMA_DIR / ".file_hashes.json"

# ── 2. Chunking parameters ───────────────────────────────────────────────────
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 100

# ── 3. Embedding model ───────────────────────────────────────────────────────
# moka-ai/m3e-base is a bilingual (Chinese/English) sentence-transformer model.
# It is downloaded automatically from HuggingFace on first run (~400 MB).
EMBED_MODEL = "moka-ai/m3e-base"

# ── 4. Cross-lingual test query ──────────────────────────────────────────────
TEST_QUERY = "What are the core technical skills? 主要的技术栈是什么？"


# Supported extensions and their LangChain loader class names
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


# ─────────────────────────────────────────────────────────────────────────────
def ensure_knowledge_base(path: Path) -> None:
    """Create the knowledge_base folder if it does not exist yet."""
    if not path.exists():
        path.mkdir(parents=True)
        print(
            f"\n[INFO] The folder '{path}' did not exist — it has been created.\n"
            "       Please copy your documents (.pdf, .docx, .txt, .md) there,\n"
            "       then re-run this script.\n"
        )
        sys.exit(0)

    # rglob("*") walks the full directory tree including nested subfolders
    supported_files = [
        f for f in path.rglob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not supported_files:
        print(
            f"\n[WARNING] '{path}' contains no supported documents\n"
            "          (.pdf, .docx, .txt, .md) — checked recursively.\n"
            "          Please add your documents and re-run this script.\n"
        )
        sys.exit(0)

    print(f"[INFO] Found {len(supported_files)} supported file(s) in '{path}' (recursive):")
    for f in supported_files:
        print(f"       • {f.relative_to(path)}")


# ─────────────────────────────────────────────────────────────────────────────
def get_loader(file_path: Path):
    """Return the appropriate LangChain loader for a given file extension."""
    from langchain_community.document_loaders import (
        PyPDFLoader,
        Docx2txtLoader,
        TextLoader,
    )

    ext = file_path.suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(str(file_path))
    elif ext == ".docx":
        return Docx2txtLoader(str(file_path))
    elif ext in (".txt", ".md"):
        # TextLoader handles plain text and Markdown files.
        # encoding="utf-8" is important for Chinese characters.
        return TextLoader(str(file_path), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")


def load_and_split_file(file_path: Path, kb_dir: Path) -> list:
    """Load a single file and split it into chunks. Returns chunk list."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    loader = get_loader(file_path)
    raw_docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
    )
    return splitter.split_documents(raw_docs)


# ─────────────────────────────────────────────────────────────────────────────
# Hash registry helpers
# ─────────────────────────────────────────────────────────────────────────────
def compute_md5(file_path: Path) -> str:
    """Return the MD5 hex-digest of a file's binary content."""
    import hashlib
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def load_hash_registry() -> dict[str, str]:
    """Load the persisted {relative_path: md5} registry, or empty dict."""
    import json
    if HASH_FILE.exists():
        return json.loads(HASH_FILE.read_text(encoding="utf-8"))
    return {}


def save_hash_registry(registry: dict[str, str]) -> None:
    """Persist the updated registry to disk."""
    import json
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    HASH_FILE.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Vector store helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_embeddings():
    """Load the embedding model (cached by HuggingFace after first download)."""
    from langchain_huggingface import HuggingFaceEmbeddings
    print(f"[INFO] Loading embedding model '{EMBED_MODEL}' …")
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def open_or_create_store(embeddings, persist_dir: Path):
    """Open an existing Chroma store or create a new empty one."""
    from langchain_community.vectorstores import Chroma
    return Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
    )


def delete_file_from_store(store, abs_source: str) -> int:
    """
    Remove all chunks whose 'source' metadata equals abs_source.
    Returns the number of chunks deleted.
    """
    coll = store._collection
    # Fetch IDs of all chunks belonging to this source file
    existing = coll.get(where={"source": abs_source}, include=[])
    ids = existing.get("ids", [])
    if ids:
        coll.delete(ids=ids)
    return len(ids)


# ─────────────────────────────────────────────────────────────────────────────
def run_retrieval_test(vector_store, query: str) -> None:
    """Run a quick similarity search to verify that retrieval works."""
    print(f"\n[STEP 5] Running test retrieval …")
    print(f"         Query: \"{query}\"")

    results = vector_store.similarity_search(query, k=1)

    if not results:
        print("         [WARNING] No results returned — the index may be empty.")
        return

    top = results[0]
    source   = top.metadata.get("source", "unknown")
    page     = top.metadata.get("page",   "?")
    snippet  = top.page_content[:400].replace("\n", " ")

    print("\n" + "─" * 70)
    print(f"  Top match  →  {Path(source).name}  (page {page})")
    print("─" * 70)
    print(f"  {snippet} …")
    print("─" * 70)
    print("\n✔ Cross-lingual retrieval is working correctly.\n")


# ─────────────────────────────────────────────────────────────────────────────
def load_existing_store(embed_model: str, persist_dir: Path):
    """
    Load the already-built Chroma vector store from disk.
    No embedding computation — loads in a couple of seconds.
    """
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma

    if not persist_dir.exists():
        print(
            f"\n[ERROR] Vector store not found at '{persist_dir}'.\n"
            "        Run without --query first to build it:\n"
            "        python src/ingest.py\n"
        )
        sys.exit(1)

    print(f"[INFO] Loading existing vector store from '{persist_dir}' …")
    embeddings = HuggingFaceEmbeddings(
        model_name=embed_model,
        encode_kwargs={"normalize_embeddings": True},
    )
    store = Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
    )
    print("       ✔ Vector store ready.")
    return store


# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="joblensDemo — RAG ingestion & retrieval tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  Rebuild vector store (needed after adding new docs):\n"
            "    python src/ingest.py\n\n"
            "  Query existing store (instant, no re-embedding):\n"
            "    python src/ingest.py --query \"Python 相关技能\"\n"
            "    python src/ingest.py --query \"machine learning\" --top-k 3\n"
        ),
    )
    p.add_argument(
        "--query", "-q",
        metavar="TEXT",
        default=None,
        help="Run retrieval only against the existing DB (no rebuild).",
    )
    p.add_argument(
        "--top-k", "-k",
        type=int,
        default=1,
        metavar="N",
        help="Number of results to return in query mode (default: 1).",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    print("=" * 70)
    print("  joblensDemo — Knowledge Base Tool")
    print("  Embedding model : moka-ai/m3e-base  (bilingual Chinese/English)")
    print("=" * 70)

    # ── QUERY-ONLY MODE ───────────────────────────────────────────────
    if args.query is not None:
        store = load_existing_store(EMBED_MODEL, CHROMA_DIR)
        # Override k in run_retrieval_test by passing it inline
        query = args.query or TEST_QUERY
        print(f"\n[QUERY] \"{query}\"  (top {args.top_k} result(s))")
        results = store.similarity_search(query, k=args.top_k)
        if not results:
            print("[WARNING] No results returned — the index may be empty.")
            return
        for i, doc in enumerate(results, 1):
            source  = doc.metadata.get("source", "unknown")
            page    = doc.metadata.get("page", "?")
            snippet = doc.page_content[:400].replace("\n", " ")
            print("\n" + "─" * 70)
            print(f"  [{i}] {Path(source).name}  (page {page})")
            print("─" * 70)
            print(f"  {snippet} …")
        print("─" * 70 + "\n")
        return

def incremental_update(kb_dir: Path) -> None:
    """
    Incremental indexing using MD5 hashes.

    1. Scan all files in kb_dir, compute MD5 for each.
    2. Load hash registry from last run.
    3. Diff:
       - Deleted  : in registry, not on disk        → remove from Chroma
       - Modified : on disk, hash changed            → remove old + re-embed
       - New      : on disk, not in registry         → embed and add
       - Unchanged: hash identical                   → skip ✔ (fast!)
    4. Persist updated registry.
    """
    ensure_knowledge_base(kb_dir)

    # ─ Scan current files ────────────────────────────────────────────────────────
    current_files: dict[str, str] = {}   # rel_path_str → md5
    for fp in sorted(kb_dir.rglob("*")):
        if fp.is_file() and fp.suffix.lower() in SUPPORTED_EXTENSIONS:
            rel = str(fp.relative_to(kb_dir))
            current_files[rel] = compute_md5(fp)

    registry = load_hash_registry()      # rel_path_str → md5 (from last run)

    # ─ Classify ───────────────────────────────────────────────────────────────────
    to_delete  = [r for r in registry if r not in current_files]          # gone
    to_add     = [r for r in current_files if r not in registry]          # new
    to_update  = [
        r for r in current_files
        if r in registry and current_files[r] != registry[r]             # changed
    ]
    unchanged  = len(current_files) - len(to_add) - len(to_update)

    print(f"\n  ► Unchanged : {unchanged:3d}  (skipped)")
    print(f"  ► New       : {len(to_add):3d}")
    print(f"  ► Modified  : {len(to_update):3d}")
    print(f"  ► Deleted   : {len(to_delete):3d}")

    needs_work = to_delete or to_add or to_update
    if not needs_work:
        print("\n✔ Knowledge base is up-to-date. Nothing to do.\n")
        return

    # ─ Load model & store once (only when changes exist) ─────────────────────
    embeddings = get_embeddings()
    store      = open_or_create_store(embeddings, CHROMA_DIR)

    # ─ Process deletions (deleted + old version of modified) ────────────────
    for rel in to_delete + to_update:
        abs_src = str(kb_dir / rel)
        n = delete_file_from_store(store, abs_src)
        tag = "🗑️  Deleted" if rel in to_delete else "🔄 Modified"
        print(f"  {tag}  {rel}  (−{n} chunks)")
        if rel in to_delete:
            registry.pop(rel, None)

    # ─ Process additions (new + updated files) ────────────────────────────
    for rel in to_add + to_update:
        fp = kb_dir / rel
        try:
            chunks = load_and_split_file(fp, kb_dir)
            store.add_documents(chunks)
            registry[rel] = current_files[rel]       # update hash
            tag = "✨ New     " if rel in to_add else "✔ Updated "
            print(f"  {tag}  {rel}  (+{len(chunks)} chunks)")
        except Exception as exc:
            print(f"  ✘ FAILED   {rel}  — {exc}")

    save_hash_registry(registry)
    print(f"\n✔ Vector store updated. Registry saved to '{HASH_FILE.name}'.\n")


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    print("=" * 70)
    print("  joblensDemo — Knowledge Base Tool")
    print("  Embedding model : moka-ai/m3e-base  (bilingual Chinese/English)")
    print("=" * 70)

    # ── QUERY-ONLY MODE ───────────────────────────────────────────────
    if args.query is not None:
        store = load_existing_store(EMBED_MODEL, CHROMA_DIR)
        query = args.query or TEST_QUERY
        print(f"\n[QUERY] \"{query}\"  (top {args.top_k} result(s))")
        results = store.similarity_search(query, k=args.top_k)
        if not results:
            print("[WARNING] No results returned — the index may be empty.")
            return
        for i, doc in enumerate(results, 1):
            source  = doc.metadata.get("source", "unknown")
            page    = doc.metadata.get("page", "?")
            snippet = doc.page_content[:400].replace("\n", " ")
            print("\n" + "─" * 70)
            print(f"  [{i}] {Path(source).name}  (page {page})")
            print("─" * 70)
            print(f"  {snippet} …")
        print("─" * 70 + "\n")
        return

    # ── INCREMENTAL UPDATE MODE (default) ────────────────────────────────
    incremental_update(KB_DIR)
    run_retrieval_test(load_existing_store(EMBED_MODEL, CHROMA_DIR), TEST_QUERY)


if __name__ == "__main__":
    main()
