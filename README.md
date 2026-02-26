# joblensDemo 🎯

An agentic **JD ↔ personal knowledge base** matching engine. Phase 1 of a two-phase automated job application pipeline.

## Architecture

```
joblensDemo/
├── src/
│   ├── ingest.py        # Phase 1 — Build local vector knowledge base
│   ├── compare.py       # Phase 1 — Deep JD ↔ CV matching via LLM
│   └── phase2/          # Phase 2 — Reverse RAG (planned)
├── data/                # 🔒 Private — gitignored
│   ├── knowledge_base/  # Your PDFs, DOCX, TXT, MD files
│   ├── chroma_db/       # Auto-generated vector store
│   ├── inputs/          # jd.txt, cv.txt
│   └── outputs/         # analysis_result.json
├── .env                 # Your API key (gitignored)
├── .env.example         # Template
└── requirements.txt
```

## Setup

```bash
# 1. Clone & create virtualenv
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure credentials
cp .env.example .env
# → Edit .env and fill in LLM_API_KEY, LLM_BASE_URL, LLM_MODEL_NAME
```

## Usage

### Phase 1-A: Build the knowledge base
Drop your documents (`.pdf`, `.docx`, `.txt`, `.md`) into `data/knowledge_base/` — nested folders are supported.

```bash
python src/ingest.py
```

Vectors are persisted to `data/chroma_db/` using `moka-ai/m3e-base` (bilingual Chinese/English, 100% local & free).

### Phase 1-B: Match a JD against your CV

```bash
python src/compare.py \
  --jd data/inputs/jd.txt \
  --cv data/inputs/cv.txt \
  --out data/outputs/analysis_result.json
```

Output is a strict JSON with match score, gap analysis, ATS keyword check, and a 72-hour action plan.

## Privacy

All personal data lives inside `data/` which is fully gitignored. The vector store, documents, and API keys **never leave your machine** (embeddings run locally).
