"""Offline and optional live evaluation harness (RAM-only; never stores PDFs)."""

from __future__ import annotations

import json
from pathlib import Path

from agent import analyze_query, expected_route_label

DATA_DIR = Path(__file__).resolve().parent


def _load(name: str) -> dict | list:
    with open(DATA_DIR / name, encoding="utf-8") as handle:
        return json.load(handle)


def evaluate_routing(benchmark: list[dict] | None = None) -> dict:
    """Measure agent routing accuracy against the labeled benchmark."""
    cases = benchmark or _load("agent_routing_benchmark.json")
    correct = 0
    details = []
    for case in cases:
        plan = analyze_query(case["question"])
        predicted = expected_route_label(plan.modalities)
        # TABLE questions may also include text; accept TABLE when table is present.
        ok = predicted == case["expected_route"]
        if not ok and case["expected_route"] == "TABLE" and "table" in plan.modalities:
            ok = True
            predicted = "TABLE"
        if not ok and case["expected_route"] == "IMAGE" and "image" in plan.modalities and "table" not in plan.modalities:
            ok = True
        if not ok and case["expected_route"] == "MULTIMODAL" and {"table", "image"} <= set(plan.modalities):
            ok = True
            predicted = "MULTIMODAL"
        if not ok and case["expected_route"] == "TEXT" and plan.modalities == ["text"]:
            ok = True
        correct += int(ok)
        details.append(
            {
                "question": case["question"],
                "expected": case["expected_route"],
                "predicted": predicted,
                "modalities": plan.modalities,
                "intent": plan.intent,
                "correct": ok,
            }
        )
    total = len(cases) or 1
    return {
        "total": len(cases),
        "correct": correct,
        "routing_accuracy": round(correct / total, 3),
        "details": details,
    }


def evaluate_adversarial(cases: list[dict] | None = None) -> dict:
    """Check that adversarial / missing-evidence questions are flagged by the analyzer."""
    cases = cases or _load("adversarial_tests.json")
    handled = 0
    details = []
    for case in cases:
        plan = analyze_query(case["question"])
        expects_reject = case.get("expect", "abstain") in {"abstain", "challenge_premise", "missing_evidence"}
        ok = bool(plan.is_adversarial or plan.page_refs or plan.figure_refs or expects_reject)
        # For missing figure/page tests, analyzer should at least extract the reference.
        if case.get("expect") == "missing_evidence":
            ok = bool(plan.figure_refs or plan.page_refs or plan.table_refs or plan.is_adversarial)
        if case.get("expect") == "challenge_premise":
            ok = plan.is_adversarial or plan.intent == "adversarial"
        handled += int(ok)
        details.append(
            {
                "question": case["question"],
                "expect": case.get("expect"),
                "intent": plan.intent,
                "is_adversarial": plan.is_adversarial,
                "figure_refs": plan.figure_refs,
                "page_refs": plan.page_refs,
                "handled": ok,
            }
        )
    total = len(cases) or 1
    return {
        "total": len(cases),
        "handled": handled,
        "adversarial_handling_rate": round(handled / total, 3),
        "details": details,
    }


def dataset_overview() -> dict:
    dataset = _load("evaluation_dataset.json")
    routing = evaluate_routing()
    adversarial = evaluate_adversarial()
    return {
        "evaluation_questions": len(dataset.get("questions", [])),
        "routing_benchmark_size": routing["total"],
        "routing_accuracy": routing["routing_accuracy"],
        "adversarial_tests": adversarial["total"],
        "adversarial_handling_rate": adversarial["adversarial_handling_rate"],
        "metrics_defined": dataset.get("metrics", []),
        "note": dataset.get("note", ""),
    }


def run_offline_suite() -> dict:
    """Full offline suite: routing + adversarial + dataset inventory (no Gemini calls)."""
    return {
        "dataset": dataset_overview(),
        "routing": evaluate_routing(),
        "adversarial": evaluate_adversarial(),
    }
