"""
compare.py  —  joblensDemo Phase 1 · Forward RAG · V1.0
─────────────────────────────────────────────────────────────────────────────
Agentic RAG pipeline:

  1. Load JD + CV from disk.
  2. Query local ChromaDB (moka-ai/m3e-base) with the JD → top-3 relevant
     chunks from the candidate's personal knowledge base.
  3. Build an augmented user prompt:
       [JD]  +  [CV]  +  [Retrieved Context from knowledge base]
  4. Call the configured LLM (OpenAI / OpenRouter / any compatible endpoint).
  5. Parse and pretty-print the strict JSON analysis.

Configuration (via .env):
  LLM_API_KEY    — provider API key  (required)
  LLM_BASE_URL   — provider base URL (optional, defaults to OpenAI)
  LLM_MODEL_NAME — model identifier  (optional, defaults to gpt-4o)

Usage:
  python src/compare.py
  python src/compare.py --jd data/inputs/jd.txt --cv data/inputs/cv.txt
  python src/compare.py --top-k 5   # retrieve more KB chunks
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

# ─────────────────────────────────────────────────────────────────────────────
# 1. Paths
# ─────────────────────────────────────────────────────────────────────────────
_ROOT      = Path(__file__).resolve().parent.parent
CHROMA_DIR = _ROOT / "data" / "chroma_db"
EMBED_MODEL = "moka-ai/m3e-base"

# Default I/O paths
DEFAULT_JD_PATH  = str(_ROOT / "data" / "inputs" / "jd.txt")
DEFAULT_CV_PATH  = str(_ROOT / "data" / "inputs" / "cv.txt")
DEFAULT_OUT_PATH = str(_ROOT / "data" / "outputs" / "analysis_result.json")
DEFAULT_MODEL    = "gpt-4o"
DEFAULT_TOP_K    = 3          # number of KB chunks to retrieve

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# 2. System Prompt  (strict JSON contract, language-agnostic)
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are an elite, highly analytical Tech Recruiter and Technical Architect.
Perform a brutally objective matching analysis between the given Job Description
(JD) and the Candidate's profile (CV + retrieved knowledge-base context).

# Language Rule (ABSOLUTE — no exceptions):
ALL string values in the JSON output MUST be written in English, regardless of
the language of the input JD or CV.

# Evaluation Philosophy:
1. Dynamic Persona Extraction: Deduce the JD's core priority, then deduce the
   candidate's actual technical identity from the CV AND the retrieved context.
2. Transferable Skills Mapping: Do NOT rely on exact keyword matches. Recognise
   systems-thinking, API integration patterns, and engineering mindset as
   transferable across domains.
3. Deep Context Use: The [Retrieved Context] section contains the candidate's
   most relevant project details fetched from their knowledge base. Use this
   evidence to strengthen or nuance your assessment.
4. Actionable Honesty: Identify gaps realistically, but always provide a
   tactical mitigation strategy the candidate can execute before the interview.

# Output — valid JSON only, no markdown fences, no text outside the object:
{
  "jd_core_focus": "1-sentence summary of what this job cares about most.",
  "candidate_current_persona": "1-sentence summary of the candidate's technical identity derived from CV + retrieved context.",
  "overall_match_score": <integer 0-100>,
  "hiring_recommendation": "<STRONG_YES | YES | MAYBE | NO>",
  "transferable_strengths": [
    {
      "skill_needed_in_jd": "Skill or requirement from the JD",
      "evidence_from_cv": "Specific evidence. If from the CV, quote the relevant line. If from a [Retrieved Context] chunk, you MUST cite the source file in brackets, e.g. '[from: course descriptions.docx] Built a compiler using ...'",
      "why_it_translates": "Why this experience satisfies the requirement"
    }
  ],
  "critical_gaps": [
    {
      "missing_skill": "What is required but absent",
      "severity": "<High | Medium | Low>",
      "crash_course_suggestion": "Specific action the candidate can take in 3 days to credibly address this gap"
    }
  ],
  "keyword_match": {
    "matched": ["keywords present in both JD and candidate profile"],
    "missing": ["important JD keywords absent from candidate profile — high ATS risk"]
  },
  "customized_pitch": "2-sentence elevator pitch the candidate should use for THIS role."
}
"""


# ─────────────────────────────────────────────────────────────────────────────
# 3. Configuration
# ─────────────────────────────────────────────────────────────────────────────
def load_config() -> dict[str, str | None]:
    """Load provider settings from .env. Exits if LLM_API_KEY is missing."""
    load_dotenv(_ROOT / ".env")

    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        console.print("[bold red]Error:[/] LLM_API_KEY is not set. Add it to your .env file.")
        sys.exit(1)

    return {
        "api_key":  api_key,
        "base_url": os.getenv("LLM_BASE_URL"),          # None → OpenAI default
        "model":    os.getenv("LLM_MODEL_NAME", DEFAULT_MODEL),
    }


def build_client(config: dict) -> OpenAI:
    return OpenAI(api_key=config["api_key"], base_url=config["base_url"])  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# 4. File reader
# ─────────────────────────────────────────────────────────────────────────────
def read_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        console.print(f"[bold red]Error:[/] File not found → [yellow]{path}[/]")
        sys.exit(1)
    return p.read_text(encoding="utf-8").strip()


# ─────────────────────────────────────────────────────────────────────────────
# 5. RAG retrieval
# ─────────────────────────────────────────────────────────────────────────────
def retrieve_kb_context(jd_text: str, top_k: int = DEFAULT_TOP_K) -> str:
    """
    Query the local ChromaDB with the first 500 characters of the JD.
    Returns a formatted string of the top-k retrieved chunks, or an empty
    string with a warning if the DB is not found.
    """
    if not CHROMA_DIR.exists():
        console.print(
            "[bold yellow]Warning:[/] ChromaDB not found at "
            f"[yellow]{CHROMA_DIR}[/]. "
            "Run [bold]python src/ingest.py[/] first to build the knowledge base.\n"
            "         Proceeding without RAG context (CV only)."
        )
        return ""

    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma

    console.print(f"[dim]Loading KB embeddings ({EMBED_MODEL}) …[/]")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    store = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=embeddings)

    # Use first 500 chars of JD as the retrieval query (dense, topic-rich)
    query = jd_text[:500]
    results = store.similarity_search(query, k=top_k)

    if not results:
        console.print("[bold yellow]Warning:[/] KB query returned no results.")
        return ""

    chunks = []
    for i, doc in enumerate(results, 1):
        source  = Path(doc.metadata.get("source", "unknown")).name
        snippet = doc.page_content.strip()
        chunks.append(f"[Chunk {i} — source: {source}]\n{snippet}")

    console.print(f"[dim]Retrieved {len(chunks)} chunk(s) from knowledge base.[/]")
    return "\n\n".join(chunks)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Prompt builder
# ─────────────────────────────────────────────────────────────────────────────
def build_user_message(jd_text: str, cv_text: str, kb_context: str) -> str:
    """
    Assemble the user-turn message that the LLM will analyse.
    The candidate section merges the raw CV with retrieved KB context.
    """
    if kb_context:
        candidate_section = (
            "Here is the candidate's core resume AND the most highly relevant "
            "historical project details retrieved from their knowledge base:\n\n"
            f"{cv_text}\n\n"
            "[Retrieved Context from Knowledge Base]:\n"
            f"{kb_context}"
        )
    else:
        candidate_section = cv_text

    return (
        f"## Job Description\n\n{jd_text}\n\n"
        f"## Candidate Resume / Profile\n\n{candidate_section}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 7. LLM call
# ─────────────────────────────────────────────────────────────────────────────
def call_llm(client: OpenAI, model: str, user_message: str) -> str:
    """
    Send the augmented prompt to the LLM and return the raw text response.
    Avoids response_format=json_object for compatibility with non-OpenAI providers.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


# ─────────────────────────────────────────────────────────────────────────────
# 8. JSON parser
# ─────────────────────────────────────────────────────────────────────────────
def parse_json_response(raw: str) -> dict:
    """Strip optional markdown fences and parse JSON. Exits on failure."""
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        console.print(f"[bold red]JSON parse error:[/] {exc}")
        console.print(Panel(raw, title="Raw LLM Response", border_style="red"))
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Output helpers
# ─────────────────────────────────────────────────────────────────────────────
def display_result(data: dict) -> None:
    formatted = json.dumps(data, indent=2, ensure_ascii=False)
    syntax = Syntax(formatted, "json", theme="monokai", line_numbers=True)
    console.print(Panel(syntax, title="[bold green]Analysis Result[/]", border_style="green"))


def save_result(data: dict, path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"\n[bold cyan]✔ Result saved →[/] [yellow]{out.resolve()}[/]\n")


# ─────────────────────────────────────────────────────────────────────────────
# 10. CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="joblensDemo — Agentic RAG · JD ↔ Candidate analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--jd",    default=DEFAULT_JD_PATH,  help="Path to Job Description file")
    p.add_argument("--cv",    default=DEFAULT_CV_PATH,   help="Path to CV / Resume file")
    p.add_argument("--out",   default=DEFAULT_OUT_PATH,  help="Path for JSON output file")
    p.add_argument(
        "--top-k", "-k", type=int, default=DEFAULT_TOP_K, metavar="N",
        help="Number of KB chunks to retrieve from ChromaDB",
    )
    p.add_argument(
        "--no-rag", action="store_true",
        help="Skip KB retrieval and use CV text only (faster, no ChromaDB needed)",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 11. Entrypoint
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    console.rule("[bold blue]joblensDemo · Forward RAG · V1.0[/]")

    # ── Config & client ────────────────────────────────────────────────────
    config = load_config()
    console.print(f"[dim]Provider:[/] {config['base_url'] or 'api.openai.com (default)'}")
    console.print(f"[dim]Model:   [/] {config['model']}\n")
    client = build_client(config)

    # ── Step 1: Read inputs ────────────────────────────────────────────────
    console.print(f"Reading JD  → [yellow]{args.jd}[/]")
    jd_text = read_file(args.jd)

    console.print(f"Reading CV  → [yellow]{args.cv}[/]\n")
    cv_text = read_file(args.cv)

    # ── Step 2: RAG retrieval ──────────────────────────────────────────────
    if args.no_rag:
        kb_context = ""
        console.print("[dim]--no-rag flag set. Skipping KB retrieval.[/]\n")
    else:
        console.print("[bold]Querying knowledge base …[/]")
        kb_context = retrieve_kb_context(jd_text, top_k=args.top_k)

    # ── Step 3: Build augmented prompt ─────────────────────────────────────
    user_message = build_user_message(jd_text, cv_text, kb_context)

    # ── Step 4: LLM call ───────────────────────────────────────────────────
    console.print("\n[bold]Calling LLM …[/]")
    with console.status("[bold green]Waiting for response …"):
        raw_response = call_llm(client, config["model"], user_message)

    # ── Step 5: Parse, display, save ───────────────────────────────────────
    result = parse_json_response(raw_response)
    display_result(result)
    save_result(result, args.out)


if __name__ == "__main__":
    main()
