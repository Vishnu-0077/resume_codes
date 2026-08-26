"""FastAPI backend for an in-memory multimodal PDF RAG assistant."""

import base64
import json
import os
import re

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from google import genai

from rag import citation_details, evaluate_answer, ingest_pdf, retrieve

load_dotenv()

# Content is held only in RAM. This protects both local storage and a free-tier quota.
MAX_PDF_CHARACTERS = 500_000
MAX_HISTORY_ENTRIES = 8
DEFAULT_MESSAGE = "Explain this document clearly and in detail"
document_sessions: dict[str, dict[str, object]] = {}

app = FastAPI(title="PDF Insight RAG")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_error_handler(_: Request, exc: HTTPException) -> JSONResponse:
    """Use a single JSON error shape that the chat UI can display."""
    return JSONResponse(status_code=exc.status_code, content={"error": str(exc.detail)})


def _string_list(value: object) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def normalize_charts(value: object) -> list[dict]:
    """Permit only compact, numeric chart data before sending it to the browser."""
    if not isinstance(value, list):
        return []
    charts: list[dict] = []
    for chart in value[:2]:
        if not isinstance(chart, dict):
            continue
        chart_type, labels, datasets = chart.get("type"), chart.get("labels"), chart.get("datasets")
        if chart_type not in {"bar", "line"} or not isinstance(labels, list) or not isinstance(datasets, list):
            continue
        labels = [str(label) for label in labels[:12]]
        valid_datasets = []
        for dataset in datasets[:3]:
            if not isinstance(dataset, dict) or not isinstance(dataset.get("values"), list):
                continue
            values = dataset["values"][: len(labels)]
            valid = len(labels) > 0 and len(values) == len(labels)
            valid = valid and all(not isinstance(item, bool) and isinstance(item, (int, float)) for item in values)
            if valid:
                valid_datasets.append({"label": str(dataset.get("label", "Series")), "values": values})
        if valid_datasets:
            charts.append(
                {
                    "title": str(chart.get("title", "Chart")),
                    "description": str(chart.get("description", "")),
                    "type": chart_type,
                    "labels": labels,
                    "datasets": valid_datasets,
                }
            )
    return charts


def parse_llm_response(raw_text: str) -> dict:
    """Parse the structured answer while preserving a readable fallback on malformed JSON."""
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", raw_text).strip()
    try:
        data = json.loads(cleaned)
        if not isinstance(data, dict):
            raise ValueError("Expected an object")
        return {
            "title": str(data.get("title", "Document analysis")),
            "summary": str(data.get("summary", "")),
            "detailed_explanation": str(data.get("detailed_explanation", "")),
            "key_points": _string_list(data.get("key_points")),
            "possible_questions": _string_list(data.get("possible_questions")),
            "citation_ids": _string_list(data.get("citations")),
            "charts": normalize_charts(data.get("charts")),
        }
    except (json.JSONDecodeError, ValueError, TypeError):
        return {
            "title": "Document analysis",
            "summary": "",
            "detailed_explanation": raw_text,
            "key_points": [],
            "possible_questions": [],
            "citation_ids": [],
            "charts": [],
        }


def recent_history(session: dict[str, object]) -> str:
    """Keep a compact conversational memory without persisting it anywhere."""
    history = session.get("history", [])
    if not isinstance(history, list):
        return ""
    entries = [entry for entry in history[-MAX_HISTORY_ENTRIES:] if isinstance(entry, dict)]
    return "\n".join(
        f"{entry.get('role', 'User')}: {entry.get('content', '')}" for entry in entries
    )[-16_000:]


def provider_error(exc: Exception) -> HTTPException:
    """Convert Gemini failures into useful, secret-safe messages for the browser."""
    text = str(exc).lower()
    if "api key" in text or "authentication" in text or "401" in text:
        detail = "Gemini rejected GEMINI_API_KEY. Update .env with an active Google AI Studio key and restart Uvicorn."
    elif "resource_exhausted" in text or "429" in text or "quota" in text:
        detail = "The Gemini free-tier quota has been reached. Please wait for it to reset and try again."
    elif "no longer available" in text or "not_found" in text or "404" in text:
        detail = "A configured Gemini model is unavailable. Restart the server to load the latest settings."
    else:
        detail = "Gemini could not complete this request. Check the server terminal and try again."
    return HTTPException(status_code=500, detail=detail)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "storage": "in-memory only"}


@app.post("/explain")
async def explain_pdf(
    file: UploadFile | None = File(default=None),
    message: str = Form(default=""),
    session_id: str = Form(default=""),
) -> dict:
    """Ingest a new PDF or use its RAM-only vector index to answer a question."""
    session_id = session_id.strip()
    if not session_id or len(session_id) > 100:
        raise HTTPException(status_code=400, detail="The browser chat session is invalid. Refresh and try again.")

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "your_gemini_key_here":
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not configured. Add a free Google AI Studio key to .env and restart Uvicorn.",
        )
    client = genai.Client(api_key=api_key)

    if file:
        if file.content_type not in {"application/pdf", "application/x-pdf"} and not (
            file.filename or ""
        ).lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Please upload a PDF file.")
        try:
            pdf_bytes = await file.read()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Could not read the uploaded PDF.") from exc
        if not pdf_bytes:
            raise HTTPException(status_code=400, detail="The uploaded PDF is empty.")
        try:
            vector_store, truncated, counts = ingest_pdf(client, pdf_bytes, MAX_PDF_CHARACTERS)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise provider_error(exc) from exc
        # A new upload starts a new document collection for this browser chat.
        document_sessions[session_id] = {
            "store": vector_store,
            "truncated": truncated,
            "filename": file.filename or "PDF document",
            "counts": counts,
            "history": [],
        }
    else:
        if session_id not in document_sessions:
            raise HTTPException(
                status_code=400,
                detail="Attach a PDF first. After that, you can ask follow-up questions without re-uploading it.",
            )

    session = document_sessions[session_id]
    instruction = message.strip() or DEFAULT_MESSAGE
    try:
        evidence = retrieve(client, session["store"], instruction)
    except Exception as exc:
        raise provider_error(exc) from exc

    system_prompt = """You are a careful multimodal RAG document assistant. Answer using only the
retrieved PDF sources and any supplied figures. Do not use outside knowledge to fill gaps. Cite factual
claims inside detailed_explanation as [S1], [S2], and so on; only cite source IDs shown in the evidence.
Explain uncertainty when evidence is weak or incomplete.

Return ONLY valid JSON with this exact shape:
{
  "title": "string",
  "summary": "string",
  "detailed_explanation": "string",
  "key_points": ["string"],
  "possible_questions": ["string"],
  "citations": ["S1"],
  "charts": [{
    "title": "string",
    "description": "string",
    "type": "bar or line",
    "labels": ["string"],
    "datasets": [{"label": "string", "values": [1, 2]}]
  }]
}
Write an appropriately detailed explanation. Create at most two bar or line charts, and only when the
retrieved evidence contains explicit numerical values. Never fabricate values; otherwise return []."""
    user_prompt = f"""Recent conversation:
{recent_history(session) or "No earlier messages."}

Current question: {instruction}

Agent route: {", ".join(evidence.route)}

Retrieved evidence:
{evidence.context}"""
    multimodal_input: list[dict] = [{"type": "text", "text": user_prompt}]
    for image in evidence.images:
        multimodal_input.append(
            {
                "type": "image",
                "data": base64.b64encode(image.image_bytes).decode("utf-8"),
                "mime_type": image.mime_type,
            }
        )

    try:
        response = client.interactions.create(
            model="gemini-3.6-flash",
            input=multimodal_input,
            system_instruction=system_prompt,
            response_format=[
                {
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "summary": {"type": "string"},
                            "detailed_explanation": {"type": "string"},
                            "key_points": {"type": "array", "items": {"type": "string"}},
                            "possible_questions": {"type": "array", "items": {"type": "string"}},
                            "citations": {"type": "array", "items": {"type": "string"}},
                            "charts": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "title": {"type": "string"},
                                        "description": {"type": "string"},
                                        "type": {"type": "string", "enum": ["bar", "line"]},
                                        "labels": {"type": "array", "items": {"type": "string"}},
                                        "datasets": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "label": {"type": "string"},
                                                    "values": {"type": "array", "items": {"type": "number"}},
                                                },
                                                "required": ["label", "values"],
                                            },
                                        },
                                    },
                                    "required": ["title", "description", "type", "labels", "datasets"],
                                },
                            },
                        },
                        "required": [
                            "title", "summary", "detailed_explanation", "key_points",
                            "possible_questions", "citations", "charts",
                        ],
                    },
                }
            ],
        )
        raw_text = (response.output_text or "").strip()
        if not raw_text:
            raise RuntimeError("Gemini returned no text.")
    except Exception as exc:
        raise provider_error(exc) from exc

    result = parse_llm_response(raw_text)
    citations = citation_details(evidence, result.pop("citation_ids"))
    result["citations"] = citations
    result["evaluation"] = evaluate_answer(evidence, citations)
    result["document_loaded"] = True
    result["document_name"] = str(session["filename"])
    result["ingestion"] = session["counts"]
    result["truncated"] = bool(session["truncated"])
    if result["truncated"]:
        result["notice"] = f"Only the first {MAX_PDF_CHARACTERS:,} PDF characters were indexed for this session."

    history = session.setdefault("history", [])
    if isinstance(history, list):
        history.extend(
            [
                {"role": "User", "content": instruction},
                {"role": "Assistant", "content": f"{result['title']}: {result['summary']}"},
            ]
        )
        session["history"] = history[-MAX_HISTORY_ENTRIES:]
    return result


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", include_in_schema=False)
async def home() -> FileResponse:
    return FileResponse("static/index.html")
