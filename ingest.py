"""
ingest.py
---------
RAG ingestion pipeline for the joblensDemo project.

Loads all supported documents from ./knowledge_base/ (.pdf, .docx, .txt, .md),
splits the text, embeds using the bilingual (Chinese + English) model
moka-ai/m3e-base, and persists the resulting vectors to ./chroma_db/.

A quick retrieval test is run at the end to verify cross-lingual search works.

──────────────────────────────────────────────────────────────────────────────
PREREQUISITES — install inside the project's virtual environment:

  source .venv/bin/activate          # activate the existing .venv

  pip install langchain langchain-community langchain-huggingface \
              pypdf chromadb sentence-transformers docx2txt

──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── 1. Paths ─────────────────────────────────────────────────────────────────
KB_DIR     = Path("./knowledge_base")   # input  — put your PDFs here
CHROMA_DIR = Path("./chroma_db")        # output — Chroma will persist here

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
def main() -> None:
    print("=" * 70)
    print("  joblensDemo — PDF Ingestion Pipeline")
    print("  Embedding model : moka-ai/m3e-base  (bilingual Chinese/English)")
    print("=" * 70)

    # Step 0: Validate input folder
    ensure_knowledge_base(KB_DIR)

    # Steps 1–2: Load PDFs → split into chunks
    chunks = load_and_split(KB_DIR)

    # Steps 3–4: Embed → persist to Chroma
    vector_store, _ = build_vector_store(chunks, EMBED_MODEL, CHROMA_DIR)

    # Step 5: Quick retrieval sanity check
    run_retrieval_test(vector_store, TEST_QUERY)


if __name__ == "__main__":
    main()
