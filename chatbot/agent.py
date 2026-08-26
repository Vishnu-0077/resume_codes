"""Query analysis, modality planning, evidence checking, and confidence scoring.

Deterministic and transparent — no extra agent frameworks, no disk I/O.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


INTENTS = ("factual", "comparison", "explanation", "numerical", "visual", "adversarial")


@dataclass
class QueryPlan:
    """Result of the query-analyzer + modality-planner stage."""

    intent: str
    modalities: list[str]
    figure_refs: list[int] = field(default_factory=list)
    table_refs: list[int] = field(default_factory=list)
    page_refs: list[int] = field(default_factory=list)
    is_adversarial: bool = False
    is_comparison: bool = False
    notes: list[str] = field(default_factory=list)


@dataclass
class EvidenceAssessment:
    strength: str  # HIGH / MEDIUM / LOW
    retrieval_confidence: float
    sources_used: int
    pages: list[int]
    abstain: bool
    reason: str
    unsupported_claims: list[str] = field(default_factory=list)
    flagged_claims: list[str] = field(default_factory=list)


def analyze_query(query: str) -> QueryPlan:
    """Classify intent and extract figure/table/page references from the question."""
    text = query.lower().strip()
    figure_refs = [int(match) for match in re.findall(r"\b(?:figure|fig\.?)\s*(\d+)\b", text)]
    table_refs = [int(match) for match in re.findall(r"\btable\s*(\d+)\b", text)]
    page_refs = [int(match) for match in re.findall(r"\bpage\s*(\d+)\b", text)]

    adversarial_cues = (
        "why is this false", "prove that", "the pdf says", "according to page",
        "according to the paper", "patient's diagnosis", "patient diagnosis", "is false",
        "contradict", "ignore the pdf", "ignore the document", "invent ", "confirm this",
        "secretly", "why is that correct", "explain why that claim is false",
        "explain why x is false",
    )
    is_adversarial = any(cue in text for cue in adversarial_cues) or (
        ("says" in text or "said" in text) and ("false" in text or "wrong" in text)
    ) or ("invent" in text and ("accuracy" in text or "number" in text or "metric" in text))

    is_comparison = any(
        term in text for term in ("compare", "comparison", "versus", " vs ", "difference between", "both papers")
    )

    if is_adversarial:
        intent = "adversarial"
    elif is_comparison or "compare" in text:
        intent = "comparison"
    elif figure_refs or any(term in text for term in ("figure", "diagram", "visual", "illustration", "image", "photo")):
        intent = "visual"
    elif table_refs or any(
        term in text
        for term in ("accuracy", "f1", "precision", "recall", "percent", "%", "metric", "score", "number", "how many")
    ):
        intent = "numerical"
    elif any(term in text for term in ("why", "how does", "explain", "methodology", "limitation", "approach")):
        intent = "explanation"
    else:
        intent = "factual"

    modalities = plan_modalities(text, intent, figure_refs, table_refs)
    notes = []
    if figure_refs:
        notes.append(f"Requested figure(s): {figure_refs}")
    if table_refs:
        notes.append(f"Requested table(s): {table_refs}")
    if page_refs:
        notes.append(f"Requested page(s): {page_refs}")

    return QueryPlan(
        intent=intent,
        modalities=modalities,
        figure_refs=figure_refs,
        table_refs=table_refs,
        page_refs=page_refs,
        is_adversarial=is_adversarial,
        is_comparison=is_comparison,
        notes=notes,
    )


def plan_modalities(text: str, intent: str, figure_refs: list[int], table_refs: list[int]) -> list[str]:
    """Decide which RAG paths to open for this question."""
    route = {"text"}
    table_terms = (
        "table", "tabular", "row", "column", "metric", "data", "number", "trend",
        "compare", "chart", "graph", "accuracy", "f1", "precision", "recall", "score",
    )
    image_terms = ("image", "figure", "diagram", "visual", "photo", "illustration", "chart", "graph")

    if intent in {"numerical", "comparison"} or table_refs or any(term in text for term in table_terms):
        route.add("table")
    if intent == "visual" or figure_refs or any(term in text for term in image_terms):
        route.add("image")
    if intent == "comparison" and ("figure" in text or "table" in text):
        route.update({"table", "image"})
    return sorted(route)


def expected_route_label(modalities: list[str]) -> str:
    """Map modality set to the benchmark label used in agent routing eval."""
    mods = set(modalities)
    if mods >= {"text", "table", "image"} or mods == {"table", "image"}:
        return "MULTIMODAL"
    if mods == {"text", "table"} or mods == {"table"}:
        return "TABLE" if mods == {"table"} else "TABLE"
    if "image" in mods and "table" not in mods:
        return "IMAGE"
    if "table" in mods and "image" not in mods:
        return "TABLE"
    return "TEXT"


def assess_evidence(
    sources: list[tuple[object, float]],
    plan: QueryPlan,
    page_count: int | None = None,
) -> EvidenceAssessment:
    """Score retrieval strength and decide whether the system should abstain."""
    scores = [float(score) for _, score in sources]
    average = sum(scores) / len(scores) if scores else 0.0
    pages = sorted({int(getattr(record, "page", 0)) for record, _ in sources if getattr(record, "page", 0)})
    sources_used = len(sources)

    missing_refs: list[str] = []
    if plan.figure_refs:
        found_figures = {
            int(match)
            for record, _ in sources
            for match in re.findall(r"figure\s*(\d+)", getattr(record, "content", "").lower())
        }
        found_figures |= {
            int(match)
            for record, _ in sources
            for match in re.findall(r"figure\s*(\d+)", str(getattr(record, "label", "")).lower())
        }
        for figure in plan.figure_refs:
            if figure not in found_figures and not any(
                getattr(record, "modality", "") == "image" for record, _ in sources
            ):
                missing_refs.append(f"Figure {figure}")

    if plan.table_refs:
        found_tables = {
            int(match)
            for record, _ in sources
            for match in re.findall(r"table\s*(\d+)", f"{getattr(record, 'label', '')} {getattr(record, 'content', '')}".lower())
        }
        for table in plan.table_refs:
            if table not in found_tables and not any(getattr(record, "modality", "") == "table" for record, _ in sources):
                missing_refs.append(f"Table {table}")

    if page_count is not None:
        for page in plan.page_refs:
            if page > page_count:
                missing_refs.append(f"Page {page} (document has {page_count} pages)")

    abstain = False
    reason = ""
    if missing_refs:
        abstain = True
        reason = "Requested evidence was not found in the document: " + ", ".join(missing_refs)
        strength = "LOW"
        confidence = min(average, 0.2)
    elif not sources or average < 0.18:
        abstain = True
        reason = "I couldn't find sufficient evidence in the document to answer this confidently."
        strength = "LOW"
        confidence = average
    elif average >= 0.35 and sources_used >= 3:
        strength = "HIGH"
        confidence = average
    elif average >= 0.22:
        strength = "MEDIUM"
        confidence = average
    else:
        strength = "LOW"
        confidence = average
        reason = "Retrieved evidence is weak; treat the answer cautiously."

    return EvidenceAssessment(
        strength=strength,
        retrieval_confidence=round(confidence, 3),
        sources_used=sources_used,
        pages=pages,
        abstain=abstain and strength == "LOW",
        reason=reason,
    )


def extract_numeric_claims(text: str) -> list[str]:
    """Pull likely factual numeric claims for evidence checking."""
    claims = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        cleaned = sentence.strip()
        if re.search(r"\d", cleaned) and len(cleaned) > 12:
            claims.append(cleaned)
    return claims[:12]


def check_claims_against_evidence(answer_text: str, evidence_text: str) -> list[str]:
    """Flag numeric claims whose distinctive numbers never appear in retrieved evidence.

    This is a lightweight support check — not a full NLI model.
    """
    evidence_lower = evidence_text.lower()
    unsupported: list[str] = []
    for claim in extract_numeric_claims(answer_text):
        numbers = re.findall(r"\d+(?:\.\d+)?%?", claim)
        distinctive = [number for number in numbers if number not in {"1", "2", "3"}]
        if not distinctive:
            continue
        if not any(number.lower() in evidence_lower for number in distinctive):
            unsupported.append(claim)
    return unsupported


def citation_quality(cited_ids: list[str], available_ids: list[str], answer_text: str) -> dict:
    """Compute citation precision/recall proxies from the model output."""
    cited = {item.upper() for item in cited_ids}
    available = {item.upper() for item in available_ids}
    valid = cited & available
    mentioned = {match.upper() for match in re.findall(r"\[(S\d+)\]", answer_text, flags=re.I)}
    precision = len(valid) / len(cited) if cited else 1.0
    recall = len(valid) / len(available) if available else 0.0
    return {
        "citation_precision": round(precision, 3),
        "citation_recall": round(recall, 3),
        "cited_in_text": sorted(mentioned),
        "valid_citations": sorted(valid),
        "invalid_citations": sorted(cited - available),
    }
