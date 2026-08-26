# PDF Insight — Multimodal RAG Chat App

Upload one or two PDFs and ask grounded questions. The assistant retrieves **text + tables + figures** as an evidence bundle, answers with **page-level citations**, checks weak evidence, and keeps everything **in RAM only** (no Docker, no database, no uploaded-file cache).

## Run it locally

1. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and set a free [Google AI Studio](https://aistudio.google.com/app/apikey) key.

4. Start the app:

   ```bash
   uvicorn main:app --reload
   ```

5. Open http://127.0.0.1:8000

Refreshing the browser or restarting the server clears the in-memory index. That is intentional.

## Architecture

```
USER → FastAPI
        → Document ingestion (text / structured tables / figures; OCR if scanned)
        → Multimodal embeddings (Gemini Embedding 2)
        → In-memory vector store
        → Query analyzer → modality planner → retrieve → evidence bundle
        → Gemini answer → citation rewrite → evidence checker
        → Confidence + charts + sources → browser
```

Ephemeral in-memory retrieval minimizes infrastructure and persistent document storage, while sacrificing persistence and horizontal scalability.

## What you get

| Capability | Behavior |
|---|---|
| Cross-modal evidence bundles | A figure hit also pulls caption + same-page text/tables |
| Page-grounded citations | `[S3]` becomes `[Table 3, Page 12]`; click jumps to the source card |
| Document-only toggle | Strict PDF grounding, or PDF + clearly separated general knowledge |
| Research mode | Related sections, evidence focus, follow-ups |
| Document comparison | PDF A + PDF B → comparison agent over both indexes |
| Structured tables | Markdown + header/row data; charts prefer real table numbers |
| Scanned PDFs | In-memory OCR via Gemini vision when selectable text is missing |
| Confidence panel | Evidence strength, pages, retrieval confidence; abstain when weak |
| Evidence checker | Flags numeric claims not present in retrieved evidence |
| Observability | `/metrics` and `/health` record latency/route/citation stats (never document text) |
| Evaluation | 60 probe questions, routing benchmark, adversarial suite at `/eval/offline` |

## Useful endpoints

- `POST /explain` — ingest / ask (fields: `file`, optional `file_b`, `message`, `session_id`, `answer_mode`, `research_mode`)
- `GET /health` — liveness + metric snapshot
- `GET /metrics` — request latency, routes, citation proxy
- `GET /eval/offline` — routing accuracy + adversarial handling (no PDF upload)

## Evaluation (resume-ready)

```bash
curl -s http://127.0.0.1:8000/eval/offline | python -m json.tool
```

The harness measures **agent routing accuracy**, **adversarial handling**, and defines retrieval/generation/citation metrics for document-specific runs (`evaluation/evaluation_dataset.json`).

## Caps (free-tier friendly)

500,000 text characters · 120 chunks · 20 tables · 8 figures · 2 MB figure bytes · OCR on up to 12 scanned pages. History keeps the last 8 turns in RAM.
