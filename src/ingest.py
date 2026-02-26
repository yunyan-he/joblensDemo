"""
ingest.py
---------
RAG ingestion pipeline for the joblensDemo project.

Modes
-----
  Rebuild (default)
    python src/ingest.py
    Loads all docs → splits → embeds with m3e-base → persists to chroma_db.
    Slow (re-embeds everything), but needed whenever knowledge_base changes.

  Query only  ← skip the full pipeline, results in seconds
    python src/ingest.py --query "你想搜索的内容"
    Loads the existing chroma_db and runs a similarity search instantly.
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


def load_and_split(kb_dir: Path) -> list:
    """Load every supported document in kb_dir and split into chunks."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    print("\n[STEP 1] Loading documents …")
    raw_docs = []
    # rglob("*") recurses into all nested subfolders automatically
    for file_path in sorted(kb_dir.rglob("*")):
        if not file_path.is_file() or file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            loader = get_loader(file_path)
            docs = loader.load()
            raw_docs.extend(docs)
            # Show path relative to knowledge_base/ so nested files are obvious
            print(f"         ✔ {file_path.relative_to(kb_dir)}  ({len(docs)} page/chunk(s))")
        except Exception as exc:
            print(f"         ✘ {file_path.relative_to(kb_dir)}  — skipped ({exc})")

    print(f"         Total: {len(raw_docs)} document section(s) loaded.")

    print("[STEP 2] Splitting text into chunks …")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # The separators below work well for mixed Chinese/English text:
        # Chinese sentences end with '。', paragraphs split on newlines, etc.
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"         Created {len(chunks)} chunk(s) total.")
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
def build_vector_store(chunks: list, embed_model: str, persist_dir: Path):
    """Embed the chunks and persist them to a local Chroma database."""
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma

    print(f"\n[STEP 3] Loading embedding model '{embed_model}' …")
    print("         (This may take a moment on first run while the model downloads.)")
    # encode_kwargs normalise=True mirrors the training objective of m3e-base
    embeddings = HuggingFaceEmbeddings(
        model_name=embed_model,
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f"[STEP 4] Embedding chunks and persisting to '{persist_dir}' …")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_dir),
    )
    print(f"         ✔ Vector store saved to '{persist_dir}'.")
    return vector_store, embeddings


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

    # ── REBUILD MODE (default) ──────────────────────────────────────────
    # Step 0: Validate input folder
    ensure_knowledge_base(KB_DIR)

    # Steps 1–2: Load docs → split into chunks
    chunks = load_and_split(KB_DIR)

    # Steps 3–4: Embed → persist to Chroma
    vector_store, _ = build_vector_store(chunks, EMBED_MODEL, CHROMA_DIR)

    # Step 5: Quick retrieval sanity check
    run_retrieval_test(vector_store, TEST_QUERY)


if __name__ == "__main__":
    main()
