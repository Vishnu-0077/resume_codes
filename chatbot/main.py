"""Local PDF explainer API and static web UI."""

import json
import os
import re

import pymupdf
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from google import genai

load_dotenv()

# 500k characters covers most books/reports while remaining comfortably below
# Gemini's context window. An unbounded prompt can exceed provider limits.
MAX_PDF_CHARACTERS = 500_000
DEFAULT_MESSAGE = "Explain this document clearly"

# Version 1 keeps the extracted text only in local server memory, per browser chat.
# Restarting Uvicorn clears this cache; no PDF content is written to disk.
document_sessions: dict[str, dict[str, str | bool]] = {}

app = FastAPI(title="PDF Explainer")

# This permits a separately served local frontend during development too.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_error_handler(_: Request, exc: HTTPException) -> JSONResponse:
    """Keep API errors consistent and easy for the frontend to display."""
    return JSONResponse(status_code=exc.status_code, content={"error": str(exc.detail)})


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Read every text layer in an uploaded PDF without saving it to disk."""
    try:
        document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        try:
            return "\n".join(page.get_text() for page in document).strip()
        finally:
            document.close()
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file could not be read as a PDF.",
        ) from exc


def parse_llm_response(raw_text: str) -> dict:
    """Parse Gemini's JSON, with a useful text-only fallback if it is malformed."""
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", raw_text).strip()
    try:
        data = json.loads(cleaned)
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object")
        return {
            "title": str(data.get("title", "PDF explanation")),
            "summary": str(data.get("summary", "")),
            "detailed_explanation": str(data.get("detailed_explanation", "")),
            "key_points": data.get("key_points", []) if isinstance(data.get("key_points", []), list) else [],
            "possible_questions": data.get("possible_questions", [])
            if isinstance(data.get("possible_questions", []), list)
            else [],
        }
    except (json.JSONDecodeError, ValueError, TypeError):
        return {
            "title": "PDF explanation",
            "summary": "",
            "detailed_explanation": raw_text,
            "key_points": [],
            "possible_questions": [],
        }


@app.get("/health")
async def health() -> dict:
    """A lightweight endpoint for checking that the backend is alive."""
    return {"status": "ok"}


@app.post("/explain")
async def explain_pdf(
    file: UploadFile | None = File(default=None),
    message: str = Form(default=""),
    session_id: str = Form(default=""),
) -> dict:
    """Use a newly uploaded or cached PDF, then return Gemini's explanation."""
    session_id = session_id.strip()
    if not session_id or len(session_id) > 100:
        raise HTTPException(status_code=400, detail="The browser chat session is invalid. Refresh and try again.")

    if file:
        if file.content_type not in {"application/pdf", "application/x-pdf"} and not (
            file.filename or ""
        ).lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Please upload a PDF file.")
        try:
            pdf_bytes = await file.read()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Could not read the uploaded file.") from exc
        if not pdf_bytes:
            raise HTTPException(status_code=400, detail="The uploaded PDF is empty.")

        extracted_text = extract_pdf_text(pdf_bytes)
        if not extracted_text:
            raise HTTPException(
                status_code=400,
                detail="No selectable text was found in this PDF. It may be a scanned document.",
            )

        truncated = len(extracted_text) > MAX_PDF_CHARACTERS
        document_text = extracted_text[:MAX_PDF_CHARACTERS]
        # A new upload replaces the document used by this browser conversation.
        document_sessions[session_id] = {
            "text": document_text,
            "truncated": truncated,
            "filename": file.filename or "PDF document",
        }
    else:
        cached_document = document_sessions.get(session_id)
        if not cached_document:
            raise HTTPException(
                status_code=400,
                detail="Attach a PDF first. After that, you can ask follow-up questions without re-uploading it.",
            )
        document_text = str(cached_document["text"])
        truncated = bool(cached_document["truncated"])
    instruction = message.strip() or DEFAULT_MESSAGE

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    # Stop before making a request when the example value was copied unchanged.
    if not api_key or api_key == "your_gemini_key_here":
        raise HTTPException(
            status_code=500,
            detail=(
                "GEMINI_API_KEY is not configured. Create a free API key in Google AI Studio, "
                "add it to .env, then restart Uvicorn."
            ),
        )

    # Keep the format predictable so the browser can render each section cleanly.
    system_prompt = """You explain PDFs clearly and accurately. Respond with ONLY valid JSON,
with no Markdown fences or extra text, matching exactly this schema:
{
  \"title\": \"string\",
  \"summary\": \"string\",
  \"detailed_explanation\": \"string\",
  \"key_points\": [\"string\"],
  \"possible_questions\": [\"string\"]
}
Use simple language unless the user requests otherwise."""
    user_prompt = f"User instruction: {instruction}\n\nPDF text:\n{document_text}"

    try:
        # The current Gemini SDK uses the Interactions API. No paid tools are enabled.
        client = genai.Client(api_key=api_key)
        response = client.interactions.create(
            model="gemini-3.6-flash",
            input=user_prompt,
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
                        },
                        "required": [
                            "title",
                            "summary",
                            "detailed_explanation",
                            "key_points",
                            "possible_questions",
                        ],
                    },
                }
            ],
        )
        raw_text = (response.output_text or "").strip()
        if not raw_text:
            raise RuntimeError("The language model returned no text.")
    except Exception as exc:
        # Tell the user exactly what to do, without exposing a secret in the browser.
        error_text = str(exc).lower()
        if "api key" in error_text or "authentication" in error_text or "401" in error_text:
            detail = (
                "Gemini rejected GEMINI_API_KEY. Create or copy a key from Google AI Studio, "
                "update .env, and restart Uvicorn."
            )
        elif "resource_exhausted" in error_text or "429" in error_text or "quota" in error_text:
            detail = "The Gemini free-tier quota has been reached. Please wait and try again later."
        elif "no longer available" in error_text or "not_found" in error_text or "404" in error_text:
            detail = "The configured Gemini model is unavailable. Restart the server so it loads the latest model setting."
        else:
            detail = "Unable to generate an explanation. Check the server terminal for details and try again."
        raise HTTPException(
            status_code=500,
            detail=detail,
        ) from exc

    result = parse_llm_response(raw_text)
    result["truncated"] = truncated
    result["document_loaded"] = True
    if truncated:
        result["notice"] = (
            f"This PDF was long, so only the first {MAX_PDF_CHARACTERS:,} characters were used."
        )
    return result


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", include_in_schema=False)
async def home() -> FileResponse:
    return FileResponse("static/index.html")
