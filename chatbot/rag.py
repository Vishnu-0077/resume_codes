"""Small in-memory multimodal RAG building blocks for the PDF chat app.

Nothing in this module writes to disk: vectors, extracted tables, OCR text, and a
small selection of figure bytes exist only until the FastAPI process is restarted.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Iterable

import pymupdf
from google import genai
from google.genai import types

from agent import QueryPlan, analyze_query, assess_evidence, expected_route_label

EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIMENSIONS = 768
TEXT_CHUNK_SIZE = 1_400
TEXT_CHUNK_OVERLAP = 180
MAX_TEXT_CHUNKS = 120
MAX_TABLES = 20
MAX_IMAGES = 8
MAX_IMAGE_BYTES = 2 * 1024 * 1024
MAX_OCR_PAGES = 12
OCR_MODEL = "gemini-3.6-flash"


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
    label: str = ""
    bbox: tuple[float, float, float, float] | None = None
    caption: str = ""
    table_data: dict | None = None
    doc_id: str = "A"
    page_count: int = 0


@dataclass
class InMemoryVectorStore:
    """A tiny vector DB replacement: cosine search over one document in memory."""

    records: list[IndexRecord] = field(default_factory=list)
    page_count: int = 0
    doc_id: str = "A"
    ocr_used: bool = False
    filename: str = ""

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

    def by_page(self, page: int, modalities: set[str] | None = None) -> list[IndexRecord]:
        return [
            record
            for record in self.records
            if record.page == page and (modalities is None or record.modality in modalities)
        ]


@dataclass
class RetrievedEvidence:
    route: list[str]
    route_label: str
    plan: QueryPlan
    sources: list[tuple[IndexRecord, float]]
    context: str
    images: list[IndexRecord]
    evidence_bundle: list[dict]
    assessment: object


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


def _structured_table(table: object, page_number: int, table_number: int) -> tuple[str, dict, str]:
    """Preserve table structure as markdown + machine-readable rows for charts/answers."""
    try:
        rows = table.extract()
    except Exception:
        return "", {}, ""
    if not rows:
        return "", {}, ""

    normalized = []
    for row in rows[:40]:
        if not isinstance(row, list):
            continue
        normalized.append([str(cell or "").strip() for cell in row[:16]])
    if not normalized:
        return "", {}, ""

    headers = normalized[0]
    body = normalized[1:] if len(normalized) > 1 else []
    # Prefer a header-looking first row; otherwise synthesize column names.
    if body and sum(1 for cell in headers if cell) >= max(1, len(headers) // 2):
        pass
    else:
        headers = [f"Col {index}" for index in range(1, len(normalized[0]) + 1)]
        body = normalized

    width = max(len(headers), max((len(row) for row in body), default=0))
    headers = (headers + [""] * width)[:width]
    body = [(row + [""] * width)[:width] for row in body[:30]]

    label = f"Table {table_number}"
    lines = [label, "", " | ".join(headers), " | ".join("---" for _ in headers)]
    for row in body:
        lines.append(" | ".join(row))
    markdown = "\n".join(lines)
    table_data = {"label": label, "headers": headers, "rows": body, "page": page_number}
    content = f"{label} on page {page_number}:\n{markdown}"
    return content, table_data, label


def _nearby_caption(page_text_blocks: list[tuple], image_bbox: tuple[float, float, float, float] | None) -> str:
    """Pick a short caption-like line near an image (usually below it)."""
    if not image_bbox:
        return ""
    x0, y0, x1, y1 = image_bbox
    candidates = []
    for block in page_text_blocks:
        bx0, by0, bx1, by1, text = block
        snippet = " ".join(str(text).split())
        if not snippet or len(snippet) > 220:
            continue
        vertically_near = (by0 >= y1 - 8 and by0 <= y1 + 90) or (by1 <= y0 + 8 and by1 >= y0 - 60)
        horizontally_overlap = not (bx1 < x0 - 40 or bx0 > x1 + 40)
        if vertically_near and horizontally_overlap:
            distance = abs(by0 - y1)
            candidates.append((distance, snippet))
    if not candidates:
        return ""
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _page_blocks(page: pymupdf.Page) -> list[tuple]:
    blocks = []
    try:
        for block in page.get_text("blocks"):
            if len(block) >= 5 and isinstance(block[4], str):
                blocks.append((float(block[0]), float(block[1]), float(block[2]), float(block[3]), block[4]))
    except Exception:
        return []
    return blocks


def _ocr_page_with_gemini(client: genai.Client, page: pymupdf.Page) -> str:
    """OCR a scanned page in memory via Gemini vision — no files written."""
    try:
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False)
        image_bytes = pixmap.tobytes("png")
    except Exception:
        return ""
    try:
        response = client.models.generate_content(
            model=OCR_MODEL,
            contents=[
                types.Content(
                    parts=[
                        types.Part.from_text(
                            text="Extract all readable text from this scanned PDF page. "
                            "Preserve reading order. Return plain text only."
                        ),
                        types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                    ]
                )
            ],
        )
        return (response.text or "").strip()
    except Exception:
        return ""


def _extract_artifacts(
    client: genai.Client,
    pdf_bytes: bytes,
    max_characters: int,
    doc_id: str = "A",
) -> tuple[list[dict], bool, int, bool]:
    """Extract text chunks, structured tables, and figures; OCR when needed."""
    artifacts: list[dict] = []
    characters_seen = 0
    text_chunks = table_count = image_count = image_bytes_total = 0
    seen_xrefs: set[int] = set()
    truncated = False
    ocr_used = False
    table_index = 0
    figure_index = 0
    page_count = 0

    try:
        document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError("The uploaded file could not be read as a PDF.") from exc

    try:
        page_count = document.page_count
        selectable = sum(len(page.get_text().strip()) for page in document)
        needs_ocr = selectable < 40

        for page_number, page in enumerate(document, start=1):
            page_text = page.get_text().strip()
            if needs_ocr and page_number <= MAX_OCR_PAGES and len(page_text) < 20:
                ocr_text = _ocr_page_with_gemini(client, page)
                if ocr_text:
                    page_text = ocr_text
                    ocr_used = True

            remaining = max_characters - characters_seen
            if remaining <= 0:
                truncated = True
                break
            if len(page_text) > remaining:
                page_text = page_text[:remaining]
                truncated = True
            characters_seen += len(page_text)
            blocks = _page_blocks(page)
            page_rect = tuple(page.rect)  # type: ignore[arg-type]

            for part_number, chunk in enumerate(_chunks(page_text), start=1):
                if text_chunks >= MAX_TEXT_CHUNKS:
                    break
                artifacts.append(
                    {
                        "modality": "text",
                        "page": page_number,
                        "content": chunk,
                        "source_id": f"{doc_id}:text-p{page_number}-{part_number}",
                        "label": f"Page {page_number}",
                        "bbox": page_rect,
                        "caption": "",
                        "table_data": None,
                        "doc_id": doc_id,
                        "page_count": page_count,
                    }
                )
                text_chunks += 1

            if table_count < MAX_TABLES:
                try:
                    tables = page.find_tables().tables
                except Exception:
                    tables = []
                for local_number, table in enumerate(tables, start=1):
                    if table_count >= MAX_TABLES:
                        break
                    table_index += 1
                    content, table_data, label = _structured_table(table, page_number, table_index)
                    if not content:
                        continue
                    bbox = None
                    try:
                        bbox = tuple(float(value) for value in table.bbox)  # type: ignore[attr-defined]
                    except Exception:
                        bbox = page_rect
                    artifacts.append(
                        {
                            "modality": "table",
                            "page": page_number,
                            "content": content,
                            "source_id": f"{doc_id}:table-p{page_number}-{local_number}",
                            "label": label,
                            "bbox": bbox,
                            "caption": "",
                            "table_data": table_data,
                            "doc_id": doc_id,
                            "page_count": page_count,
                        }
                    )
                    table_count += 1

            if image_count < MAX_IMAGES and image_bytes_total < MAX_IMAGE_BYTES:
                for local_number, image_info in enumerate(page.get_images(full=True), start=1):
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
                    bbox = None
                    try:
                        rects = page.get_image_rects(xref)
                        if rects:
                            rect = rects[0]
                            bbox = (float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))
                    except Exception:
                        bbox = None
                    figure_index += 1
                    caption = _nearby_caption(blocks, bbox)
                    label = f"Figure {figure_index}"
                    content = f"{label} extracted from page {page_number}"
                    if caption:
                        content = f"{content}. Caption: {caption}"
                    artifacts.append(
                        {
                            "modality": "image",
                            "page": page_number,
                            "content": content,
                            "source_id": f"{doc_id}:image-p{page_number}-{local_number}",
                            "label": label,
                            "bbox": bbox,
                            "caption": caption,
                            "table_data": None,
                            "doc_id": doc_id,
                            "page_count": page_count,
                            "mime_type": mime_type,
                            "image_bytes": image_data,
                        }
                    )
                    image_count += 1
                    image_bytes_total += len(image_data)
    finally:
        document.close()

    if not any(item["modality"] == "text" for item in artifacts):
        raise ValueError(
            "No readable text was found in this PDF, even after OCR. "
            "Try a clearer scan or a text-based PDF."
        )
    return artifacts, truncated, page_count, ocr_used


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


def ingest_pdf(
    client: genai.Client,
    pdf_bytes: bytes,
    max_characters: int,
    doc_id: str = "A",
    filename: str = "",
) -> tuple[InMemoryVectorStore, bool, dict[str, int]]:
    """Build a RAM-only multimodal vector index for a newly uploaded PDF."""
    artifacts, truncated, page_count, ocr_used = _extract_artifacts(client, pdf_bytes, max_characters, doc_id=doc_id)
    records: list[IndexRecord] = []
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
                    label=item.get("label", ""),
                    bbox=item.get("bbox"),
                    caption=item.get("caption", ""),
                    table_data=item.get("table_data"),
                    doc_id=item.get("doc_id", doc_id),
                    page_count=item.get("page_count", page_count),
                )
            )
    counts = {modality: sum(record.modality == modality for record in records) for modality in ("text", "table", "image")}
    if ocr_used:
        counts["ocr_pages"] = min(page_count, MAX_OCR_PAGES)
    store = InMemoryVectorStore(records=records, page_count=page_count, doc_id=doc_id, ocr_used=ocr_used, filename=filename)
    return store, truncated, counts


def route_query(query: str) -> list[str]:
    """Backward-compatible modality route for the three RAG paths."""
    return analyze_query(query).modalities


def _boost_requested_artifacts(store: InMemoryVectorStore, plan: QueryPlan, scored: list[tuple[IndexRecord, float]]) -> list[tuple[IndexRecord, float]]:
    """Ensure explicitly requested figures/tables are present in the hit list when indexed."""
    existing_ids = {record.source_id for record, _ in scored}
    extras: list[tuple[IndexRecord, float]] = []
    for record in store.records:
        if record.source_id in existing_ids:
            continue
        label_l = record.label.lower()
        content_l = record.content.lower()
        wanted = False
        for figure in plan.figure_refs:
            if record.modality == "image" and (f"figure {figure}" in label_l or f"figure {figure}" in content_l):
                wanted = True
        for table in plan.table_refs:
            if record.modality == "table" and (f"table {table}" in label_l or f"table {table}" in content_l):
                wanted = True
        for page in plan.page_refs:
            if record.page == page:
                wanted = True
        if wanted:
            extras.append((record, 0.99))
    return extras + scored


def build_evidence_bundle(store: InMemoryVectorStore, sources: list[tuple[IndexRecord, float]]) -> list[tuple[IndexRecord, float]]:
    """Cross-modal expansion: for each hit, attach same-page caption/text/table/figure siblings."""
    bundled: list[tuple[IndexRecord, float]] = []
    seen: set[str] = set()
    pages = {record.page for record, _ in sources}

    def add(record: IndexRecord, score: float) -> None:
        if record.source_id in seen:
            return
        seen.add(record.source_id)
        bundled.append((record, score))

    for record, score in sources:
        add(record, score)
        for sibling in store.by_page(record.page):
            if sibling.source_id == record.source_id:
                continue
            # Prefer complementary modalities on the same page.
            if sibling.modality != record.modality or sibling.modality == "text":
                sibling_score = max(score * 0.92, 0.2)
                add(sibling, sibling_score)

    # Keep the bundle focused: primary hits first, then a few page companions.
    primary_ids = {record.source_id for record, _ in sources}
    primary = [(record, score) for record, score in bundled if record.source_id in primary_ids]
    companions = [(record, score) for record, score in bundled if record.source_id not in primary_ids]
    companions = sorted(companions, key=lambda item: item[1], reverse=True)
    # Cap companions so prompts stay within free-tier comfort.
    limited = primary + companions[:6]
    # Prefer pages that appeared in the original hits.
    limited = sorted(
        limited,
        key=lambda item: (0 if item[0].page in pages else 1, -item[1]),
    )
    return limited[:10]


def _format_source(source_label: str, record: IndexRecord) -> str:
    page_bit = f"Page {record.page}"
    label_bit = record.label or record.modality.title()
    header = f"[{source_label}] {label_bit} · {page_bit} · {record.modality}"
    if record.doc_id and record.doc_id != "A":
        header += f" · Doc {record.doc_id}"
    if record.modality == "image":
        caption = record.caption or "Inspect the attached figure directly."
        return f"{header}\n{record.content}\nNearby caption/context: {caption}"
    if record.modality == "table" and record.table_data:
        return f"{header}\n{record.content}"
    return f"{header}\n{record.content[:2_000]}"


def retrieve(
    client: genai.Client,
    store: InMemoryVectorStore,
    question: str,
    plan: QueryPlan | None = None,
) -> RetrievedEvidence:
    """Route a question, retrieve cross-modal hits, and expand into an evidence bundle."""
    plan = plan or analyze_query(question)
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[question],
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=EMBEDDING_DIMENSIONS,
        ),
    )
    query_vector = list(response.embeddings[0].values)
    sources = store.search(query_vector, set(plan.modalities), limit=6)
    if not sources and "text" not in plan.modalities:
        sources = store.search(query_vector, {"text"}, limit=4)
    sources = _boost_requested_artifacts(store, plan, sources)
    sources = build_evidence_bundle(store, sources)

    context_parts = []
    images: list[IndexRecord] = []
    bundle_meta: list[dict] = []
    for number, (record, score) in enumerate(sources, start=1):
        source_label = f"S{number}"
        context_parts.append(_format_source(source_label, record))
        if record.modality == "image" and record.image_bytes and record.mime_type and len(images) < 3:
            images.append(record)
        bundle_meta.append(
            {
                "id": source_label,
                "source_id": record.source_id,
                "modality": record.modality,
                "page": record.page,
                "label": record.label,
                "score": round(score, 3),
                "bbox": list(record.bbox) if record.bbox else None,
                "caption": record.caption,
                "doc_id": record.doc_id,
            }
        )

    assessment = assess_evidence(sources, plan, page_count=store.page_count)
    return RetrievedEvidence(
        route=plan.modalities,
        route_label=expected_route_label(plan.modalities),
        plan=plan,
        sources=sources,
        context="\n\n".join(context_parts),
        images=images,
        evidence_bundle=bundle_meta,
        assessment=assessment,
    )


def retrieve_multi(
    client: genai.Client,
    stores: list[InMemoryVectorStore],
    question: str,
    plan: QueryPlan | None = None,
) -> RetrievedEvidence:
    """Retrieve independently from each document and merge evidence for comparison mode."""
    plan = plan or analyze_query(question)
    merged_sources: list[tuple[IndexRecord, float]] = []
    images: list[IndexRecord] = []
    context_parts: list[str] = []
    bundle_meta: list[dict] = []
    number = 1
    for store in stores:
        evidence = retrieve(client, store, question, plan=plan)
        for record, score in evidence.sources[:5]:
            source_label = f"S{number}"
            context_parts.append(_format_source(source_label, record))
            bundle_meta.append(
                {
                    "id": source_label,
                    "source_id": record.source_id,
                    "modality": record.modality,
                    "page": record.page,
                    "label": record.label,
                    "score": round(score, 3),
                    "bbox": list(record.bbox) if record.bbox else None,
                    "caption": record.caption,
                    "doc_id": record.doc_id,
                }
            )
            merged_sources.append((record, score))
            if record.modality == "image" and record.image_bytes and record.mime_type and len(images) < 4:
                images.append(record)
            number += 1
    assessment = assess_evidence(merged_sources, plan)
    return RetrievedEvidence(
        route=plan.modalities,
        route_label=expected_route_label(plan.modalities),
        plan=plan,
        sources=merged_sources,
        context="\n\n".join(context_parts),
        images=images,
        evidence_bundle=bundle_meta,
        assessment=assessment,
    )


def citation_details(evidence: RetrievedEvidence, cited_ids: object) -> list[dict]:
    """Resolve [S#] citations into page-level grounded source cards."""
    wanted = {str(value).upper().replace("[", "").replace("]", "") for value in cited_ids} if isinstance(cited_ids, list) else set()
    # Also accept citations embedded only in prose if the model forgot the array.
    details = []
    for number, (record, score) in enumerate(evidence.sources, start=1):
        source_id = f"S{number}"
        if wanted and source_id not in wanted:
            continue
        display = record.label or record.modality.title()
        details.append(
            {
                "id": source_id,
                "modality": record.modality,
                "page": record.page,
                "label": display,
                "display": f"{display}, Page {record.page}",
                "excerpt": record.content[:220] if record.modality != "image" else (record.caption or "Figure retrieved for visual inspection."),
                "score": round(score, 3),
                "bbox": list(record.bbox) if record.bbox else None,
                "source_id": record.source_id,
                "doc_id": record.doc_id,
                "table_data": record.table_data,
            }
        )
    return details


def charts_from_tables(evidence: RetrievedEvidence) -> list[dict]:
    """Build chart specs directly from structured table data when possible."""
    charts: list[dict] = []
    for record, _ in evidence.sources:
        if not record.table_data or len(charts) >= 2:
            continue
        headers = record.table_data.get("headers") or []
        rows = record.table_data.get("rows") or []
        if len(headers) < 2 or not rows:
            continue
        label_idx = 0
        value_cols = []
        for index, header in enumerate(headers[1:], start=1):
            numeric_values = []
            for row in rows:
                if index >= len(row):
                    numeric_values = []
                    break
                raw = str(row[index]).replace("%", "").replace(",", "").strip()
                try:
                    numeric_values.append(float(raw))
                except ValueError:
                    numeric_values = []
                    break
            if numeric_values and len(numeric_values) == len(rows):
                value_cols.append((header or f"Series {index}", numeric_values))
        if not value_cols:
            continue
        labels = [str(row[label_idx]) for row in rows if row]
        charts.append(
            {
                "title": record.label or "Table chart",
                "description": f"Values taken directly from {record.label} on page {record.page}.",
                "type": "bar",
                "labels": labels[:12],
                "datasets": [
                    {"label": name, "values": values[:12]}
                    for name, values in value_cols[:3]
                ],
            }
        )
    return charts


def evaluate_answer(evidence: RetrievedEvidence, citations: list[dict], unsupported_claims: list[str] | None = None) -> dict:
    """Runtime evaluation: retrieval strength, citation coverage, and evidence confidence."""
    scores = [score for _, score in evidence.sources]
    average_score = sum(scores) / len(scores) if scores else 0.0
    coverage = len(citations) / len(evidence.sources) if evidence.sources else 0.0
    assessment = evidence.assessment
    unsupported = unsupported_claims or []
    if not evidence.sources:
        groundedness = "low"
    elif coverage >= 0.5 and average_score >= 0.25 and not unsupported:
        groundedness = "high"
    else:
        groundedness = "medium" if evidence.sources else "low"
    if getattr(assessment, "strength", "") == "LOW":
        groundedness = "low"
    return {
        "route": evidence.route,
        "route_label": evidence.route_label,
        "intent": evidence.plan.intent,
        "retrieved_sources": len(evidence.sources),
        "citation_coverage": round(coverage, 2),
        "average_retrieval_score": round(average_score, 3),
        "groundedness": groundedness,
        "evidence_strength": getattr(assessment, "strength", "LOW"),
        "retrieval_confidence": getattr(assessment, "retrieval_confidence", 0.0),
        "pages": getattr(assessment, "pages", []),
        "unsupported_claim_rate": round(len(unsupported) / max(1, len(unsupported) + max(len(citations), 1)), 3),
        "unsupported_claims": unsupported[:5],
    }
