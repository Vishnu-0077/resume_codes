"""In-memory observability for the PDF RAG assistant.

Records request metrics only — never document contents — and lives entirely in RAM.
"""

from __future__ import annotations

import statistics
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from threading import Lock


@dataclass
class RequestMetric:
    request_id: str
    route: list[str]
    intent: str
    answer_mode: str
    chunks_retrieved: int
    citation_count: int
    retrieval_ms: float
    llm_ms: float
    total_ms: float
    evidence_strength: str
    retrieval_confidence: float
    api_failure: bool = False
    timestamp: float = field(default_factory=time.time)


class MetricsStore:
    """Ring buffer of recent request metrics (RAM only, no disk writes)."""

    def __init__(self, capacity: int = 200) -> None:
        self._lock = Lock()
        self._items: deque[RequestMetric] = deque(maxlen=capacity)

    def new_request_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def record(self, metric: RequestMetric) -> None:
        with self._lock:
            self._items.append(metric)

    def summary(self) -> dict:
        with self._lock:
            items = list(self._items)
        if not items:
            return {
                "requests": 0,
                "average_retrieval_ms": 0.0,
                "average_llm_ms": 0.0,
                "average_total_ms": 0.0,
                "average_chunks_retrieved": 0.0,
                "average_citation_count": 0.0,
                "citation_coverage_proxy": 0.0,
                "api_failures": 0,
                "evidence_strength_counts": {},
                "route_counts": {},
            }

        def avg(values: list[float]) -> float:
            return round(statistics.fmean(values), 2) if values else 0.0

        strength_counts: dict[str, int] = {}
        route_counts: dict[str, int] = {}
        for item in items:
            strength_counts[item.evidence_strength] = strength_counts.get(item.evidence_strength, 0) + 1
            route_key = "+".join(item.route) if item.route else "none"
            route_counts[route_key] = route_counts.get(route_key, 0) + 1

        cited = [item.citation_count / item.chunks_retrieved for item in items if item.chunks_retrieved]
        return {
            "requests": len(items),
            "average_retrieval_ms": avg([item.retrieval_ms for item in items]),
            "average_llm_ms": avg([item.llm_ms for item in items]),
            "average_total_ms": avg([item.total_ms for item in items]),
            "average_chunks_retrieved": avg([float(item.chunks_retrieved) for item in items]),
            "average_citation_count": avg([float(item.citation_count) for item in items]),
            "citation_coverage_proxy": round(statistics.fmean(cited), 3) if cited else 0.0,
            "api_failures": sum(1 for item in items if item.api_failure),
            "evidence_strength_counts": strength_counts,
            "route_counts": route_counts,
            "recent": [asdict(item) for item in items[-10:]],
        }


metrics_store = MetricsStore()
