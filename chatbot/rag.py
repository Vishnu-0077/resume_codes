"""Small in-memory multimodal RAG building blocks for the PDF chat app.

Nothing in this module writes to disk: vectors, extracted tables, and a small
selection of figure bytes exist only until the FastAPI process is restarted.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Iterable

import pymupdf
from google import genai
from google.genai import types

EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIMENSIONS = 768
TEXT_CHUNK_SIZE = 1_400
TEXT_CHUNK_OVERLAP = 180
MAX_TEXT_CHUNKS = 120
MAX_TABLES = 20
MAX_IMAGES = 8
MAX_IMAGE_BYTES = 2 * 1024 * 1024


@dataclass
class IndexRecord:
    """One searchable text, table, or figure item held only in RAM."""

    source_id: str
    modality: str
    page: int
    content: str
    vector: list[float]
    norm: float
    mime_type: str | None = None
    image_bytes: bytes | None = None


@dataclass
class InMemoryVectorStore:
    """A tiny vector DB replacement: cosine search over one document in memory."""

    records: list[IndexRecord] = field(default_factory=list)

    def search(self, query_vector: list[float], modalities: set[str], limit: int = 6) -> list[tuple[IndexRecord, float]]:
        query_norm = _norm(query_vector)
        if not query_norm:
            return []
        results = []
        for record in self.records:
            if record.modality not in modalities or not record.norm:
                continue
            score = sum(left * right for left, right in zip(query_vector, record.vector)) / (query_norm * record.norm)
            results.append((record, score))
        return sorted(results, key=lambda item: item[1], reverse=True)[:limit]


@dataclass
class RetrievedEvidence:
    route: list[str]
    sources: list[tuple[IndexRecord, float]]
    context: str
    images: list[IndexRecord]


def _norm(vector: Iterable[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _chunks(text: str) -> list[str]:
    """Split page text into overlapping, readable chunks without extra packages."""
    cleaned = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not cleaned:
        return []
    chunks = []
    start = 0
    while start < len(cleaned) and len(chunks) < MAX_TEXT_CHUNKS:
        end = min(start + TEXT_CHUNK_SIZE, len(cleaned))
        if end < len(cleaned):
            boundary = max(cleaned.rfind("\n", start + 600, end), cleaned.rfind(". ", start + 600, end))
            if boundary > start:
                end = boundary + 1
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(cleaned):
            break
        start = max(end - TEXT_CHUNK_OVERLAP, start + 1)
    return chunks


def _table_as_text(table: object) -> str:
    """Turn a detected PyMuPDF table into a compact searchable representation."""
    try:
        rows = table.extract()
    except Exception:
        return ""
    rendered_rows = []
    for row in rows[:30]:
        if not isinstance(row, list):
            continue
        rendered_rows.append(" | ".join(str(cell or "").strip() for cell in row[:12]))
    return "\n".join(row for row in rendered_rows if row.strip())


def _extract_artifacts(pdf_bytes: bytes, max_characters: int) -> tuple[list[dict], bool]:
    """Extract text chunks, detected tables, and small embedded figures directly from memory."""
    artifacts: list[dict] = []
    characters_seen = 0
    text_chunks = table_count = image_count = image_bytes_total = 0
    seen_xrefs: set[int] = set()
    truncated = False

    try:
        document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError("The uploaded file could not be read as a PDF.") from exc

    try:
        for page_number, page in enumerate(document, start=1):
            page_text = page.get_text().strip()
            remaining = max_characters - characters_seen
            if remaining <= 0:
                truncated = True
                break
            if len(page_text) > remaining:
                page_text = page_text[:remaining]
                truncated = True
            characters_seen += len(page_text)

            for part_number, chunk in enumerate(_chunks(page_text), start=1):
                if text_chunks >= MAX_TEXT_CHUNKS:
                    break
                artifacts.append(
                    {
                        "modality": "text",
                        "page": page_number,
                        "content": chunk,
                        "source_id": f"text-p{page_number}-{part_number}",
                    }
                )
                text_chunks += 1

            if table_count < MAX_TABLES:
                try:
                    tables = page.find_tables().tables
                except Exception:
                    tables = []
                for table_number, table in enumerate(tables, start=1):
                    if table_count >= MAX_TABLES:
                        break
                    table_text = _table_as_text(table)
                    if table_text:
                        artifacts.append(
                            {
                                "modality": "table",
                                "page": page_number,
                                "content": f"Table on page {page_number}:\n{table_text}",
                                "source_id": f"table-p{page_number}-{table_number}",
                            }
                        )
                        table_count += 1

            if image_count < MAX_IMAGES and image_bytes_total < MAX_IMAGE_BYTES:
                for image_number, image_info in enumerate(page.get_images(full=True), start=1):
                    xref = image_info[0]
                    if xref in seen_xrefs or image_count >= MAX_IMAGES:
                        continue
                    seen_xrefs.add(xref)
                    try:
                        image = document.extract_image(xref)
                        image_data = image["image"]
                        extension = image.get("ext", "png").lower()
                    except Exception:
                        continue
                    if not image_data or image_bytes_total + len(image_data) > MAX_IMAGE_BYTES:
                        continue
                    if extension not in {"png", "jpg", "jpeg"}:
                        continue
                    mime_type = "image/jpeg" if extension in {"jpg", "jpeg"} else "image/png"
                    artifacts.append(
                        {
                            "modality": "image",
                            "page": page_number,
                            "content": f"Figure {image_number} extracted from page {page_number}",
                            "source_id": f"image-p{page_number}-{image_number}",
                            "mime_type": mime_type,
                            "image_bytes": image_data,
                        }
                    )
                    image_count += 1
                    image_bytes_total += len(image_data)
    finally:
        document.close()

    if not any(item["modality"] == "text" for item in artifacts):
        raise ValueError("No selectable text was found in this PDF. It may be a scanned document.")
    return artifacts, truncated


def _embed_batch(client: genai.Client, artifacts: list[dict]) -> list[list[float]]:
    """Embed a small batch; each artifact becomes one vector in a unified space."""
    contents = []
    for item in artifacts:
        if item["modality"] == "image":
            contents.append(
                types.Content(
                    parts=[types.Part.from_bytes(data=item["image_bytes"], mime_type=item["mime_type"])]
                )
            )
        else:
            contents.append(types.Content(parts=[types.Part.from_text(text=item["content"])]))
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=contents,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=EMBEDDING_DIMENSIONS,
        ),
    )
    return [list(embedding.values) for embedding in response.embeddings]


def ingest_pdf(client: genai.Client, pdf_bytes: bytes, max_characters: int) -> tuple[InMemoryVectorStore, bool, dict[str, int]]:
    """Build a RAM-only multimodal vector index for a newly uploaded PDF."""
    artifacts, truncated = _extract_artifacts(pdf_bytes, max_characters)
    records: list[IndexRecord] = []
    # Gemini Embedding 2 accepts at most six images in one request; text batches stay light too.
    batches: list[list[dict]] = []
    batch: list[dict] = []
    image_count = 0
    for artifact in artifacts:
        would_exceed_images = artifact["modality"] == "image" and image_count >= 6
        if len(batch) >= 24 or would_exceed_images:
            batches.append(batch)
            batch, image_count = [], 0
        batch.append(artifact)
        if artifact["modality"] == "image":
            image_count += 1
    if batch:
        batches.append(batch)

    for batch in batches:
        vectors = _embed_batch(client, batch)
        if len(vectors) != len(batch):
            raise RuntimeError("Gemini returned an incomplete embedding batch.")
        for item, vector in zip(batch, vectors):
            records.append(
                IndexRecord(
                    source_id=item["source_id"],
                    modality=item["modality"],
                    page=item["page"],
                    content=item["content"],
                    vector=vector,
                    norm=_norm(vector),
                    mime_type=item.get("mime_type"),
                    image_bytes=item.get("image_bytes"),
                )
            )
    counts = {modality: sum(record.modality == modality for record in records) for modality in ("text", "table", "image")}
    return InMemoryVectorStore(records), truncated, counts


def route_query(query: str) -> list[str]:
    """A transparent, deterministic query-routing agent for the three RAG paths."""
    text = query.lower()
    route = {"text"}
    if any(term in text for term in ("table", "tabular", "row", "column", "metric", "data", "number", "trend", "compare", "chart", "graph")):
        route.add("table")
    if any(term in text for term in ("image", "figure", "diagram", "visual", "photo", "illustration", "chart", "graph")):
        route.add("image")
    return sorted(route)


def retrieve(client: genai.Client, store: InMemoryVectorStore, question: str) -> RetrievedEvidence:
    """Route a question, embed it, and retrieve the strongest cross-modal evidence."""
    route = route_query(question)
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[question],
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=EMBEDDING_DIMENSIONS,
        ),
    )
    query_vector = list(response.embeddings[0].values)
    sources = store.search(query_vector, set(route), limit=6)
    # A query about a table/figure still benefits from nearby written context.
    if not sources and "text" not in route:
        sources = store.search(query_vector, {"text"}, limit=4)

    context_parts = []
    images = []
    for number, (record, score) in enumerate(sources, start=1):
        source_label = f"S{number}"
        if record.modality == "image":
            context_parts.append(f"[{source_label}] Figure on PDF page {record.page}. Inspect the attached figure directly.")
            if record.image_bytes and record.mime_type:
                images.append(record)
        else:
            excerpt = record.content[:2_000]
            context_parts.append(f"[{source_label}] {record.modality.title()} from PDF page {record.page}:\n{excerpt}")
    return RetrievedEvidence(route=route, sources=sources, context="\n\n".join(context_parts), images=images[:2])


def citation_details(evidence: RetrievedEvidence, cited_ids: object) -> list[dict]:
    """Return only citations that the model made to the evidence it actually received."""
    wanted = {str(value).upper() for value in cited_ids} if isinstance(cited_ids, list) else set()
    details = []
    for number, (record, score) in enumerate(evidence.sources, start=1):
        source_id = f"S{number}"
        if source_id in wanted:
            details.append(
                {
                    "id": source_id,
                    "modality": record.modality,
                    "page": record.page,
                    "excerpt": record.content[:220] if record.modality != "image" else "Figure retrieved for visual inspection.",
                    "score": round(score, 3),
                }
            )
    return details


def evaluate_answer(evidence: RetrievedEvidence, citations: list[dict]) -> dict:
    """Lightweight evaluation: checks retrieval strength and citation grounding without extra calls."""
    scores = [score for _, score in evidence.sources]
    average_score = sum(scores) / len(scores) if scores else 0.0
    coverage = len(citations) / len(evidence.sources) if evidence.sources else 0.0
    if not evidence.sources:
        groundedness = "low"
    elif coverage >= 0.5 and average_score >= 0.25:
        groundedness = "high"
    else:
        groundedness = "medium"
    return {
        "route": evidence.route,
        "retrieved_sources": len(evidence.sources),
        "citation_coverage": round(coverage, 2),
        "average_retrieval_score": round(average_score, 3),
        "groundedness": groundedness,
    }
