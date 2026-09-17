"""Reproducible benchmark for Sarathī's retrieval and answer contracts.

The default run needs only the Python standard library. It compares a deliberately
simple single-pass baseline with a field-aware planner-style retriever. Model-level
metrics are enabled with a JSONL predictions file; see README.md for its schema.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "GPT_Input_DB(Sheet1).csv"
BENCHMARK_PATH = ROOT / "evaluation" / "benchmark.json"
RESULTS_PATH = ROOT / "evaluation" / "results" / "retrieval_results.json"


def norm(value: Any) -> str:
    """Normalize punctuation and units enough for robust benchmark matching."""
    text = str(value or "").lower().replace("–", "-").replace("—", "-")
    text = text.replace("×", "x").replace("*", "*")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokens(value: Any) -> List[str]:
    text = norm(value)
    # Keep useful compound tokens such as 0.6*v, 300-600, and km/h.
    return re.findall(r"[a-z]+(?:/[a-z]+)?|\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?|\*", text)


def load_rows() -> Dict[int, Dict[str, str]]:
    with DATA_PATH.open(newline="", encoding="utf-8-sig") as handle:
        rows = {}
        for row in csv.DictReader(handle):
            rows[int(row["S. No."])] = row
        return rows


def row_text(row: Mapping[str, str]) -> str:
    return " ".join(row.get(field, "") for field in ("problem", "category", "type", "data", "code", "clause"))


def overlap(query: Iterable[str], document: Iterable[str]) -> int:
    return len(set(query).intersection(document))


def phrase_hits(query: str, document: str) -> int:
    """Count informative two-word query phrases also present in the evidence."""
    query_words = tokens(query)
    document_text = norm(document)
    hits = 0
    for left, right in zip(query_words, query_words[1:]):
        if len(left) > 2 and len(right) > 2 and f"{left} {right}" in document_text:
            hits += 1
    return hits


def naive_rank(question: str, rows: Mapping[int, Mapping[str, str]]) -> List[int]:
    """Single-pass lexical baseline: every row is scored the same way."""
    query_tokens = tokens(question)
    scored: List[Tuple[float, int]] = []
    for row_id, row in rows.items():
        doc_tokens = tokens(row_text(row))
        score = overlap(query_tokens, doc_tokens)
        # Tie-break on row order, mirroring a basic LIMIT query.
        scored.append((float(score), -row_id))
    scored.sort(reverse=True)
    return [-row_id for _, row_id in scored]


ALIASES = {
    "unsignalised": "un-signalised",
    "un signalised": "un-signalised",
    "speed breaker": "speed hump",
    "road stud": "road studs",
    "road marking": "marking",
    "signs": "sign",
    "kmph": "km/h",
}


def expanded_question(question: str) -> str:
    text = norm(question)
    for source, target in ALIASES.items():
        text = text.replace(source, target)
    return text


def improved_rank(question: str, rows: Mapping[int, Mapping[str, str]]) -> List[int]:
    """Planner-style approximation used for a dependency-light regression test.

    It gives priority to exact domain fields (type/category/problem), preserves
    numeric conditions, and then uses evidence-word overlap. It does not use gold
    answers or gold row IDs.
    """
    query = expanded_question(question)
    query_tokens = tokens(query)
    scored: List[Tuple[float, int]] = []
    for row_id, row in rows.items():
        row_type = norm(row.get("type"))
        row_category = norm(row.get("category"))
        row_problem = norm(row.get("problem"))
        data = norm(row.get("data"))
        score = float(overlap(query_tokens, tokens(data)))
        score += 4 * phrase_hits(query, data)
        type_tokens = tokens(row_type)
        category_tokens = tokens(row_category)
        problem_tokens = tokens(row_problem)
        score += 8 * overlap(query_tokens, type_tokens)
        score += 3 * overlap(query_tokens, category_tokens)
        score += 2 * overlap(query_tokens, problem_tokens)
        if row_type and row_type in query:
            score += 12
        # Multi-word concepts should beat rows that merely share “sign” or “marking”.
        for phrase in (row_type, row_problem, row_category):
            phrase_words = [word for word in tokens(phrase) if len(word) > 2]
            if len(phrase_words) >= 2 and all(word in query_tokens for word in phrase_words):
                score += 5
        # Conditions such as 70 km/h or 600 mm are strong retrieval signals.
        query_numbers = set(re.findall(r"\d+(?:\.\d+)?", query))
        row_numbers = set(re.findall(r"\d+(?:\.\d+)?", data))
        score += 1.5 * len(query_numbers.intersection(row_numbers))
        scored.append((score, -row_id))
    scored.sort(reverse=True)
    return [-row_id for _, row_id in scored]


def recall_at_k(ranked: Sequence[int], relevant: Sequence[int], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]).intersection(relevant)) / float(len(set(relevant)))


def source_key(source: Mapping[str, str]) -> Tuple[str, str]:
    return norm(source.get("code")), norm(source.get("clause"))


def required_answer_match(answer: str, expected: Sequence[str]) -> bool:
    answer_text = norm(answer)
    return all(norm(fact) in answer_text for fact in expected)


CLAIM_PATTERN = re.compile(
    r"(?:\d+(?:\.\d+)?(?:\s*(?:-|to|x)\s*\d+(?:\.\d+)?)?\s*(?:mm|m|km/h|meters?|numbers?)?|\b(?:white|yellow|red-white|yellow-yellow|white-white)\b)",
    re.IGNORECASE,
)


def detected_claims(answer: str) -> List[str]:
    return [norm(match) for match in CLAIM_PATTERN.findall(answer)]


def evidence_text(ids: Sequence[int], rows: Mapping[int, Mapping[str, str]]) -> str:
    return norm(" ".join(row_text(rows[row_id]) for row_id in ids if row_id in rows))


def prediction_metrics(predictions: Sequence[Mapping[str, Any]], benchmark: Sequence[Mapping[str, Any]], rows: Mapping[int, Mapping[str, str]]) -> Dict[str, Any]:
    by_id = {item["id"]: item for item in benchmark}
    citation_hits = 0
    citation_total = 0
    citation_required = 0
    correct = 0
    unsupported_claims = 0
    total_claims = 0
    latencies = []
    prompt_tokens = []
    completion_tokens = []
    for prediction in predictions:
        item = by_id.get(prediction.get("id"))
        if not item:
            continue
        if required_answer_match(prediction.get("answer", ""), item.get("expected_answers", [])):
            correct += 1
        gold = {source_key(source) for source in item.get("gold_sources", [])}
        citation_required += len(gold)
        for citation in prediction.get("citations", []):
            citation_total += 1
            citation_hits += int(source_key(citation) in gold)
        retrieved = prediction.get("retrieved_ids", item.get("relevant_ids", []))
        evidence = evidence_text(retrieved, rows)
        for claim in detected_claims(prediction.get("answer", "")):
            total_claims += 1
            unsupported_claims += int(claim not in evidence)
        if isinstance(prediction.get("latency_ms"), (int, float)):
            latencies.append(float(prediction["latency_ms"]))
        if isinstance(prediction.get("prompt_tokens"), (int, float)):
            prompt_tokens.append(float(prediction["prompt_tokens"]))
        if isinstance(prediction.get("completion_tokens"), (int, float)):
            completion_tokens.append(float(prediction["completion_tokens"]))
    return {
        "evaluated_predictions": len(predictions),
        "answer_correctness": correct / len(predictions) if predictions else None,
        "citation_accuracy": citation_hits / citation_total if citation_total else None,
        "citation_coverage": citation_hits / citation_required if citation_required else None,
        "unsupported_claim_rate": unsupported_claims / total_claims if total_claims else None,
        "mean_latency_ms": sum(latencies) / len(latencies) if latencies else None,
        "mean_prompt_tokens": sum(prompt_tokens) / len(prompt_tokens) if prompt_tokens else None,
        "mean_completion_tokens": sum(completion_tokens) / len(completion_tokens) if completion_tokens else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Sarathī retrieval and optional answer predictions.")
    parser.add_argument("--predictions", type=Path, help="JSONL answer predictions to score.")
    parser.add_argument("--output", type=Path, default=RESULTS_PATH, help="Output JSON path.")
    args = parser.parse_args()

    rows = load_rows()
    benchmark = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    systems = {"naive_rag": {}, "planner_hybrid_rag": {}}
    for system_name, ranker in (("naive_rag", naive_rank), ("planner_hybrid_rag", improved_rank)):
        start = time.perf_counter()
        per_query = []
        for item in benchmark:
            ranked = ranker(item["question"], rows)
            per_query.append({
                "id": item["id"],
                "top_5": ranked[:5],
                "recall_at_1": recall_at_k(ranked, item["relevant_ids"], 1),
                "recall_at_3": recall_at_k(ranked, item["relevant_ids"], 3),
                "recall_at_5": recall_at_k(ranked, item["relevant_ids"], 5),
            })
        elapsed_ms = (time.perf_counter() - start) * 1000
        systems[system_name] = {
            "queries": len(per_query),
            "mean_retrieval_latency_ms": elapsed_ms / len(benchmark),
            "recall_at_1": sum(x["recall_at_1"] for x in per_query) / len(per_query),
            "recall_at_3": sum(x["recall_at_3"] for x in per_query) / len(per_query),
            "recall_at_5": sum(x["recall_at_5"] for x in per_query) / len(per_query),
            "per_query": per_query,
        }

    output: Dict[str, Any] = {
        "benchmark": str(BENCHMARK_PATH.relative_to(ROOT)),
        "dataset_rows": len(rows),
        "systems": systems,
        "answer_metrics": None,
        "notes": "Retrieval metrics use the checked-in 50-row CSV. Answer metrics require model predictions.",
    }
    if args.predictions:
        predictions = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
        grouped = {}
        for prediction in predictions:
            grouped.setdefault(prediction.get("system", "predictions"), []).append(prediction)
        output["answer_metrics"] = {
            system: prediction_metrics(items, benchmark, rows)
            for system, items in grouped.items()
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "naive_rag": {key: systems["naive_rag"][key] for key in ("recall_at_1", "recall_at_3", "recall_at_5", "mean_retrieval_latency_ms")},
        "planner_hybrid_rag": {key: systems["planner_hybrid_rag"][key] for key in ("recall_at_1", "recall_at_3", "recall_at_5", "mean_retrieval_latency_ms")},
        "answer_metrics": output["answer_metrics"],
    }, indent=2))


if __name__ == "__main__":
    main()
