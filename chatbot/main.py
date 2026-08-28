"""FastAPI backend for an in-memory multimodal PDF RAG assistant."""

from __future__ import annotations

import base64
import json
import os
import re
import time

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from google import genai

from agent import (
    analyze_query,
    check_claims_against_evidence,
    citation_quality,
)
from evaluation import run_offline_suite
from metrics import RequestMetric, metrics_store
from rag import (
    charts_from_tables,
    citation_details,
    evaluate_answer,
    ingest_pdf,
    retrieve,
    retrieve_multi,
)

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
            "related_sections": _string_list(data.get("related_sections")),
            "general_knowledge_notes": str(data.get("general_knowledge_notes", "")),
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
            "related_sections": [],
            "general_knowledge_notes": "",
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


def _client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "your_gemini_key_here":
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not configured. Add a free Google AI Studio key to .env and restart Uvicorn.",
        )
    return genai.Client(api_key=api_key)


def _session_or_400(session_id: str) -> dict[str, object]:
    session_id = session_id.strip()
    if not session_id or len(session_id) > 100:
        raise HTTPException(status_code=400, detail="The browser chat session is invalid. Refresh and try again.")
    if session_id not in document_sessions:
        raise HTTPException(
            status_code=400,
            detail="Attach a PDF first. After that, you can ask follow-up questions without re-uploading it.",
        )
    return document_sessions[session_id]


async def _read_pdf(file: UploadFile) -> bytes:
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
    return pdf_bytes


def _system_prompt(answer_mode: str, research_mode: bool, comparison: bool) -> str:
    if answer_mode == "document_plus_knowledge":
        grounding = (
            "Prefer retrieved PDF evidence for all factual claims. You may add brief general-knowledge "
            "clarifications only in general_knowledge_notes, clearly separated from document-grounded content. "
            "Never mix outside knowledge into detailed_explanation without labeling it."
        )
    else:
        grounding = (
            "Answer using ONLY the retrieved PDF sources and any supplied figures. "
            "Do not use outside knowledge to fill gaps. If evidence is insufficient, say so clearly."
        )
    research = ""
    if research_mode:
        research = (
            " Research mode: organize the answer as evidence-backed findings, highlight relevant "
            "figures/tables, list related_sections (page/label hints), and propose sharp follow-up questions."
        )
    compare = ""
    if comparison:
        compare = (
            " Comparison mode: evidence may come from Doc A and Doc B. Explicitly contrast methodologies, "
            "datasets, results, limitations, and conclusions. Cite sources with their Doc labels."
        )
    return f"""You are a careful multimodal RAG document assistant. {grounding}{research}{compare}

Cite factual claims inside detailed_explanation as [S1], [S2], and so on; only cite source IDs shown in the evidence.
When citing, prefer page-aware phrasing such as (see [S1]). Explain uncertainty when evidence is weak.
If the evidence assessment says LOW strength or asks you to abstain, do not invent an answer.

Return ONLY valid JSON with this exact shape:
{{
  "title": "string",
  "summary": "string",
  "detailed_explanation": "string",
  "key_points": ["string"],
  "possible_questions": ["string"],
  "related_sections": ["string"],
  "general_knowledge_notes": "string",
  "citations": ["S1"],
  "charts": [{{
    "title": "string",
    "description": "string",
    "type": "bar or line",
    "labels": ["string"],
    "datasets": [{{"label": "string", "values": [1, 2]}}]
  }}]
}}
Write an appropriately detailed explanation. Create at most two bar or line charts, and only when the
retrieved evidence contains explicit numerical values. Prefer structured table values over guessing.
Never fabricate values; otherwise return []. Leave general_knowledge_notes empty in document-only mode."""


def _rewrite_citations(text: str, citations: list[dict]) -> str:
    """Replace [S#] with page-grounded markers like [Page 12, Table 3]."""
    lookup = {item["id"]: item for item in citations}

    def repl(match: re.Match[str]) -> str:
        key = match.group(1).upper()
        item = lookup.get(key)
        if not item:
            return match.group(0)
        label = item.get("label") or item.get("modality", "Source").title()
        page = item.get("page")
        return f"[{label}, Page {page}]"

    return re.sub(r"\[(S\d+)\]", repl, text)


def _call_gemini(client: genai.Client, system_prompt: str, user_prompt: str, images: list) -> str:
    multimodal_input: list[dict] = [{"type": "text", "text": user_prompt}]
    for image in images:
        multimodal_input.append(
            {
                "type": "image",
                "data": base64.b64encode(image.image_bytes).decode("utf-8"),
                "mime_type": image.mime_type,
            }
        )
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
                        "related_sections": {"type": "array", "items": {"type": "string"}},
                        "general_knowledge_notes": {"type": "string"},
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
                        "title",
                        "summary",
                        "detailed_explanation",
                        "key_points",
                        "possible_questions",
                        "related_sections",
                        "general_knowledge_notes",
                        "citations",
                        "charts",
                    ],
                },
            }
        ],
    )
    raw_text = (response.output_text or "").strip()
    if not raw_text:
        raise RuntimeError("Gemini returned no text.")
    return raw_text


def _build_answer_payload(
    session: dict[str, object],
    instruction: str,
    evidence,
    result: dict,
    answer_mode: str,
    research_mode: bool,
    request_id: str,
    retrieval_ms: float,
    llm_ms: float,
    total_ms: float,
) -> dict:
    citations = citation_details(evidence, result.pop("citation_ids"))
    # Prefer structured-table charts; fall back to model charts.
    table_charts = normalize_charts(charts_from_tables(evidence))
    model_charts = result.get("charts") or []
    result["charts"] = table_charts or model_charts

    unsupported = check_claims_against_evidence(result.get("detailed_explanation", ""), evidence.context)
    if unsupported and answer_mode == "document_only":
        # Flag unsupported numeric claims rather than silently shipping them.
        result["detailed_explanation"] = (
            result["detailed_explanation"].rstrip()
            + "\n\nEvidence checker: some numeric claims were not found in the retrieved sources and should be treated cautiously."
        )

    result["detailed_explanation"] = _rewrite_citations(result["detailed_explanation"], citations)
    result["citations"] = citations
    result["evidence_bundle"] = evidence.evidence_bundle
    result["evaluation"] = evaluate_answer(evidence, citations, unsupported_claims=unsupported)
    cite_stats = citation_quality(
        [item["id"] for item in citations],
        [f"S{index}" for index in range(1, len(evidence.sources) + 1)],
        result["detailed_explanation"],
    )
    result["evaluation"].update(cite_stats)

    assessment = evidence.assessment
    result["confidence"] = {
        "evidence_strength": getattr(assessment, "strength", "LOW"),
        "sources_used": getattr(assessment, "sources_used", len(evidence.sources)),
        "pages": getattr(assessment, "pages", []),
        "retrieval_confidence": getattr(assessment, "retrieval_confidence", 0.0),
        "abstain": bool(getattr(assessment, "abstain", False)),
        "reason": getattr(assessment, "reason", ""),
        "unsupported_claims": unsupported[:5],
    }
    result["agent"] = {
        "intent": evidence.plan.intent,
        "route": evidence.route,
        "route_label": evidence.route_label,
        "figure_refs": evidence.plan.figure_refs,
        "table_refs": evidence.plan.table_refs,
        "page_refs": evidence.plan.page_refs,
        "notes": evidence.plan.notes,
    }
    result["answer_mode"] = answer_mode
    result["research_mode"] = research_mode
    if answer_mode == "document_only":
        result["general_knowledge_notes"] = ""

    if getattr(assessment, "abstain", False) and getattr(assessment, "strength", "") == "LOW":
        result["title"] = "Insufficient evidence"
        result["summary"] = assessment.reason or "Not enough grounded evidence."
        result["detailed_explanation"] = (
            assessment.reason
            or "I couldn't find sufficient evidence in the document to answer this confidently."
        )
        result["key_points"] = ["Retrieval confidence was too low to answer safely."]
        result["charts"] = []

    primary = session.get("documents", {}).get("A") or session.get("store")
    filename = session.get("filename")
    counts = session.get("counts", {})
    if primary is not None and hasattr(primary, "filename") and primary.filename:
        filename = primary.filename
    result["document_loaded"] = True
    result["document_name"] = str(filename or "PDF document")
    result["documents"] = {
        key: getattr(store, "filename", key)
        for key, store in (session.get("documents") or {}).items()
    }
    result["ingestion"] = counts
    result["truncated"] = bool(session.get("truncated"))
    result["ocr_used"] = bool(session.get("ocr_used"))
    result["request_id"] = request_id
    result["latency"] = {
        "retrieval_ms": round(retrieval_ms, 1),
        "llm_ms": round(llm_ms, 1),
        "total_ms": round(total_ms, 1),
    }
    if result["truncated"]:
        result["notice"] = f"Only the first {MAX_PDF_CHARACTERS:,} PDF characters were indexed for this session."
    if result.get("ocr_used"):
        result["notice"] = (result.get("notice") or "") + " OCR was used for scanned pages (in memory only)."

    metrics_store.record(
        RequestMetric(
            request_id=request_id,
            route=list(evidence.route),
            intent=evidence.plan.intent,
            answer_mode=answer_mode,
            chunks_retrieved=len(evidence.sources),
            citation_count=len(citations),
            retrieval_ms=retrieval_ms,
            llm_ms=llm_ms,
            total_ms=total_ms,
            evidence_strength=str(getattr(assessment, "strength", "LOW")),
            retrieval_confidence=float(getattr(assessment, "retrieval_confidence", 0.0)),
        )
    )

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


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "storage": "in-memory only", "metrics": metrics_store.summary()}


@app.get("/metrics")
async def metrics() -> dict:
    return metrics_store.summary()


@app.get("/eval/offline")
async def eval_offline() -> dict:
    """Run routing + adversarial benchmarks without touching document storage."""
    return run_offline_suite()


@app.post("/explain")
async def explain_pdf(
    file: UploadFile | None = File(default=None),
    file_b: UploadFile | None = File(default=None),
    message: str = Form(default=""),
    session_id: str = Form(default=""),
    answer_mode: str = Form(default="document_only"),
    research_mode: str = Form(default="false"),
) -> dict:
    """Ingest PDF(s) in RAM and answer with the multimodal evidence workflow."""
    started = time.perf_counter()
    request_id = metrics_store.new_request_id()
    session_id = session_id.strip()
    if not session_id or len(session_id) > 100:
        raise HTTPException(status_code=400, detail="The browser chat session is invalid. Refresh and try again.")

    answer_mode = answer_mode.strip().lower()
    if answer_mode not in {"document_only", "document_plus_knowledge"}:
        answer_mode = "document_only"
    research = research_mode.strip().lower() in {"1", "true", "yes", "on"}

    client = _client()

    if file:
        pdf_bytes = await _read_pdf(file)
        try:
            vector_store, truncated, counts = ingest_pdf(
                client, pdf_bytes, MAX_PDF_CHARACTERS, doc_id="A", filename=file.filename or "PDF A"
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise provider_error(exc) from exc
        document_sessions[session_id] = {
            "store": vector_store,
            "documents": {"A": vector_store},
            "truncated": truncated,
            "filename": file.filename or "PDF document",
            "counts": counts,
            "ocr_used": bool(vector_store.ocr_used),
            "history": [],
        }

    if file_b:
        if session_id not in document_sessions:
            raise HTTPException(status_code=400, detail="Upload PDF A before attaching PDF B for comparison.")
        pdf_bytes_b = await _read_pdf(file_b)
        try:
            store_b, truncated_b, counts_b = ingest_pdf(
                client, pdf_bytes_b, MAX_PDF_CHARACTERS, doc_id="B", filename=file_b.filename or "PDF B"
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise provider_error(exc) from exc
        session = document_sessions[session_id]
        documents = dict(session.get("documents") or {})
        documents["B"] = store_b
        session["documents"] = documents
        session["truncated"] = bool(session.get("truncated")) or truncated_b
        session["counts_b"] = counts_b
        session["ocr_used"] = bool(session.get("ocr_used")) or bool(store_b.ocr_used)

    if session_id not in document_sessions:
        raise HTTPException(
            status_code=400,
            detail="Attach a PDF first. After that, you can ask follow-up questions without re-uploading it.",
        )

    session = document_sessions[session_id]
    instruction = message.strip() or DEFAULT_MESSAGE
    plan = analyze_query(instruction)
    documents = session.get("documents") or {"A": session["store"]}
    comparison = "B" in documents and (
        plan.is_comparison or "compare" in instruction.lower() or "both" in instruction.lower()
    )

    retrieval_started = time.perf_counter()
    try:
        if comparison:
            evidence = retrieve_multi(client, [documents["A"], documents["B"]], instruction, plan=plan)
        else:
            evidence = retrieve(client, documents.get("A") or session["store"], instruction, plan=plan)
    except Exception as exc:
        metrics_store.record(
            RequestMetric(
                request_id=request_id,
                route=[],
                intent=plan.intent,
                answer_mode=answer_mode,
                chunks_retrieved=0,
                citation_count=0,
                retrieval_ms=(time.perf_counter() - retrieval_started) * 1000,
                llm_ms=0.0,
                total_ms=(time.perf_counter() - started) * 1000,
                evidence_strength="LOW",
                retrieval_confidence=0.0,
                api_failure=True,
            )
        )
        raise provider_error(exc) from exc
    retrieval_ms = (time.perf_counter() - retrieval_started) * 1000

    # Early abstain for clearly missing evidence — skip the LLM call.
    if evidence.assessment.abstain and evidence.assessment.strength == "LOW" and (
        plan.figure_refs or plan.table_refs or plan.page_refs
    ):
        result = {
            "title": "Insufficient evidence",
            "summary": evidence.assessment.reason,
            "detailed_explanation": evidence.assessment.reason,
            "key_points": ["The requested figure, table, or page was not found in the indexed document."],
            "possible_questions": [
                "What figures are available in this PDF?",
                "Summarize the main results table.",
                "What are the paper's limitations?",
            ],
            "related_sections": [],
            "general_knowledge_notes": "",
            "charts": [],
            "citation_ids": [],
        }
        return _build_answer_payload(
            session,
            instruction,
            evidence,
            result,
            answer_mode,
            research,
            request_id,
            retrieval_ms,
            0.0,
            (time.perf_counter() - started) * 1000,
        )

    system_prompt = _system_prompt(answer_mode, research, comparison)
    user_prompt = f"""Recent conversation:
{recent_history(session) or "No earlier messages."}

Current question: {instruction}

Query intent: {evidence.plan.intent}
Agent route: {", ".join(evidence.route)} ({evidence.route_label})
Evidence strength: {evidence.assessment.strength} (confidence={evidence.assessment.retrieval_confidence})
Assessment notes: {evidence.assessment.reason or "n/a"}

Retrieved evidence bundle:
{evidence.context}"""

    llm_started = time.perf_counter()
    try:
        raw_text = _call_gemini(client, system_prompt, user_prompt, evidence.images)
    except Exception as exc:
        metrics_store.record(
            RequestMetric(
                request_id=request_id,
                route=list(evidence.route),
                intent=evidence.plan.intent,
                answer_mode=answer_mode,
                chunks_retrieved=len(evidence.sources),
                citation_count=0,
                retrieval_ms=retrieval_ms,
                llm_ms=(time.perf_counter() - llm_started) * 1000,
                total_ms=(time.perf_counter() - started) * 1000,
                evidence_strength=str(evidence.assessment.strength),
                retrieval_confidence=float(evidence.assessment.retrieval_confidence),
                api_failure=True,
            )
        )
        raise provider_error(exc) from exc
    llm_ms = (time.perf_counter() - llm_started) * 1000

    result = parse_llm_response(raw_text)
    return _build_answer_payload(
        session,
        instruction,
        evidence,
        result,
        answer_mode,
        research,
        request_id,
        retrieval_ms,
        llm_ms,
        (time.perf_counter() - started) * 1000,
    )


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", include_in_schema=False)
async def home() -> FileResponse:
    return FileResponse("static/index.html")