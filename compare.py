"""
compare.py
----------
API-agnostic JD ↔ CV analysis CLI.

Configuration is driven entirely by environment variables (via .env):
  LLM_API_KEY    — your provider API key
  LLM_BASE_URL   — provider base URL  (omit to use OpenAI's default)
  LLM_MODEL_NAME — model identifier   (omit to fall back to gpt-4o)

Usage:
  python compare.py [--jd path/to/jd.txt] [--cv path/to/cv.txt] [--out path/to/output.json]
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

# ---------------------------------------------------------------------------
# 1. System Prompt
#    Fill in your structured JSON prompt here.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT: str = """
You are an elite, highly analytical Tech Recruiter and Technical Architect. Perform a brutally objective matching analysis between the given Job Description (JD) and Candidate Resume (CV).

# Language Rule (ABSOLUTE — no exceptions):
ALL string values in the JSON output MUST be written in English, regardless of the language of the input JD or CV. The JD may be in German, the CV may contain Chinese or German terms — you must still output every JSON field value in English. Do not mix languages within a single string value.

# Evaluation Philosophy:
1. Dynamic Persona Extraction: Deduce the JD's core priority, then deduce the candidate's actual technical identity from the CV.
2. Transferable Skills Mapping: Do NOT rely on exact keyword matches. Recognize systems-thinking, API integration patterns, and engineering mindset as transferable across domains.
3. Actionable Honesty: Identify gaps realistically, but always provide a tactical mitigation strategy the candidate can execute before the interview.

# Output — valid JSON only, no markdown fences, no text outside the object:
{
  "jd_core_focus": "1-sentence summary of what this job cares about most.",
  "candidate_current_persona": "1-sentence summary of the candidate's current technical identity.",
  "overall_match_score": <integer 0-100>,
  "hiring_recommendation": "<STRONG_YES | YES | MAYBE | NO>",
  "transferable_strengths": [
    {
      "skill_needed_in_jd": "Skill or requirement from the JD",
      "evidence_from_cv": "Specific CV item that maps to it",
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
    "matched": ["keywords present in both JD and CV"],
    "missing": ["important JD keywords absent from CV — high ATS risk"]
  },
  "customized_pitch": "2-sentence elevator pitch the candidate should use for THIS role, bridging past experience to the JD's goals."
}
"""


# ---------------------------------------------------------------------------
# 2. Defaults
# ---------------------------------------------------------------------------
DEFAULT_JD_PATH: str = "jd.txt"
DEFAULT_CV_PATH: str = "cv.txt"
DEFAULT_OUT_PATH: str = "analysis_result.json"
DEFAULT_MODEL: str = "gpt-4o"

console = Console()


# ---------------------------------------------------------------------------
# 3. Configuration loader
# ---------------------------------------------------------------------------
def load_config() -> dict[str, str | None]:
    """
    Load provider settings from the environment (after reading .env).
    Returns a dict with keys: api_key, base_url, model.
    """
    load_dotenv()

    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        console.print("[bold red]Error:[/] LLM_API_KEY is not set. Add it to your .env file.")
        sys.exit(1)

    base_url = os.getenv("LLM_BASE_URL")          # None → OpenAI default
    model = os.getenv("LLM_MODEL_NAME", DEFAULT_MODEL)

    return {"api_key": api_key, "base_url": base_url, "model": model}


# ---------------------------------------------------------------------------
# 4. OpenAI client factory
# ---------------------------------------------------------------------------
def build_client(config: dict[str, str | None]) -> OpenAI:
    """
    Instantiate the OpenAI-compatible client dynamically.
    Passing base_url=None keeps the official OpenAI endpoint.
    """
    return OpenAI(
        api_key=config["api_key"],
        base_url=config["base_url"],  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# 5. File reader
# ---------------------------------------------------------------------------
def read_file(path: str) -> str:
    """Read a text file and return its contents, with helpful error messaging."""
    file = Path(path)
    if not file.exists():
        console.print(f"[bold red]Error:[/] File not found → [yellow]{path}[/]")
        sys.exit(1)
    return file.read_text(encoding="utf-8").strip()


# ---------------------------------------------------------------------------
# 6. LLM call
# ---------------------------------------------------------------------------
def call_llm(client: OpenAI, model: str, jd_text: str, cv_text: str) -> str:
    """
    Send the JD + CV to the LLM and return the raw text response.

    We deliberately avoid response_format={"type": "json_object"} because
    some alternative providers (DeepSeek, OpenRouter, local models) do not
    support that parameter. Instead the SYSTEM_PROMPT must instruct the model
    to return pure JSON, and we strip any stray markdown fences below.
    """
    user_message = (
        f"## Job Description\n\n{jd_text}\n\n"
        f"## Candidate Resume / CV\n\n{cv_text}"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# 7. Safe JSON parser (strips markdown fences if present)
# ---------------------------------------------------------------------------
def parse_json_response(raw: str) -> dict:
    """
    Robustly parse a JSON payload that may be wrapped in markdown code fences.
    Raises SystemExit on failure so the caller doesn't need to handle it.
    """
    # Strip optional ```json … ``` or ``` … ``` wrappers
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        console.print(f"[bold red]JSON parse error:[/] {exc}")
        console.print(Panel(raw, title="Raw LLM Response", border_style="red"))
        sys.exit(1)


# ---------------------------------------------------------------------------
# 8. Output helpers
# ---------------------------------------------------------------------------
def display_result(data: dict) -> None:
    """Pretty-print the parsed JSON to the terminal using rich."""
    formatted = json.dumps(data, indent=2, ensure_ascii=False)
    syntax = Syntax(formatted, "json", theme="monokai", line_numbers=True)
    console.print(Panel(syntax, title="[bold green]Analysis Result[/]", border_style="green"))


def save_result(data: dict, path: str) -> None:
    """Persist the result to disk as a formatted JSON file."""
    out = Path(path)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"\n[bold cyan]✔ Result saved →[/] [yellow]{out.resolve()}[/]\n")


# ---------------------------------------------------------------------------
# 9. Argument parser
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a Job Description with a Resume using an LLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--jd", default=DEFAULT_JD_PATH, help="Path to the Job Description file")
    parser.add_argument("--cv", default=DEFAULT_CV_PATH, help="Path to the CV / Resume file")
    parser.add_argument("--out", default=DEFAULT_OUT_PATH, help="Path for the JSON output file")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 10. Entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    console.rule("[bold blue]JD ↔ CV Analyser[/]")

    # --- Config & client ---
    config = load_config()
    provider = config["base_url"] or "api.openai.com (default)"
    console.print(f"[dim]Provider:[/] {provider}")
    console.print(f"[dim]Model:   [/] {config['model']}\n")

    client = build_client(config)

    # --- Read inputs ---
    console.print(f"Reading JD from  [yellow]{args.jd}[/]")
    jd_text = read_file(args.jd)

    console.print(f"Reading CV from  [yellow]{args.cv}[/]\n")
    cv_text = read_file(args.cv)

    # --- Call LLM ---
    console.print("[bold]Calling LLM…[/]")
    with console.status("[bold green]Waiting for response…"):
        raw_response = call_llm(client, config["model"], jd_text, cv_text)

    # --- Parse ---
    result = parse_json_response(raw_response)

    # --- Display & save ---
    display_result(result)
    save_result(result, args.out)


if __name__ == "__main__":
    main()
