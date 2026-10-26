"""Evaluation of the memory retriever on a small sample dataset.

The dataset (``dataset.json`` next to this file) contains synthetic memories and
queries with the ids of the memories that should be retrieved for each query.

Run the tests:
    uv run pytest tests/retrieval/test_retrieval_eval.py

Print the full evaluation report:
    uv run python tests/retrieval/test_retrieval_eval.py
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from personal_assistant.objects.memory import Memory
from personal_assistant.retriever.retrieval import Retriever

DATASET_PATH = Path(__file__).resolve().parent / "dataset.json"
TOP_K = 5  # same value the application uses


# ---------------------------------------------------------------- metrics

def hit_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """1.0 if at least one relevant id is in the top k results, else 0.0."""
    return 1.0 if any(doc_id in relevant for doc_id in retrieved[:k]) else 0.0


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of the relevant ids that appear in the top k results."""
    if not relevant:
        return 0.0
    return len(relevant.intersection(retrieved[:k])) / len(relevant)


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of the top k results that are relevant."""
    if k <= 0:
        return 0.0
    return sum(1 for doc_id in retrieved[:k] if doc_id in relevant) / k


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    """1 / rank of the first relevant result, or 0.0 if none is retrieved."""
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------- evaluation

def load_dataset(path: Path = DATASET_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
    return data


def build_memories(dataset: dict[str, Any]) -> list[Memory]:
    return [
        Memory(
            memory_id=item["memory_id"],
            content=item["content"],
            timestamp="2026-01-01T00:00:00Z",
            importance=1.0,
            episode_id="eval_episode",
        )
        for item in dataset["memories"]
    ]


def evaluate(dataset: dict[str, Any], k: int = TOP_K) -> dict[str, Any]:
    """Run every query through the retriever and compute metrics.

    Returns per-query results and average metrics, overall and per query type.
    """
    retriever = Retriever(build_memories(dataset))

    per_query = []
    for q in dataset["queries"]:
        retrieved = [m.memory_id for m in retriever.retrieve_relevant_memories(q["query"], top_n=k)]
        relevant = set(q["relevant_memory_ids"])
        per_query.append({
            "query_id": q["query_id"],
            "query": q["query"],
            "type": q["type"],
            "retrieved": retrieved,
            "relevant": sorted(relevant),
            f"hit@{k}": hit_at_k(retrieved, relevant, k),
            f"recall@{k}": recall_at_k(retrieved, relevant, k),
            f"precision@{k}": precision_at_k(retrieved, relevant, k),
            "mrr": reciprocal_rank(retrieved, relevant),
        })

    metric_names = [f"hit@{k}", f"recall@{k}", f"precision@{k}", "mrr"]

    def average(rows: list[dict[str, Any]]) -> dict[str, float]:
        return {name: sum(r[name] for r in rows) / len(rows) for name in metric_names}

    groups: dict[str, list[dict[str, Any]]] = {"overall": per_query}
    for row in per_query:
        groups.setdefault(row["type"], []).append(row)

    return {
        "k": k,
        "per_query": per_query,
        "summary": {name: {"n": len(rows), **average(rows)} for name, rows in groups.items()},
    }


def format_report(results: dict[str, Any]) -> str:
    k = results["k"]
    metric_names = [f"hit@{k}", f"recall@{k}", f"precision@{k}", "mrr"]
    lines = [f"{'group':<12}{'n':>4}" + "".join(f"{m:>14}" for m in metric_names)]
    for group, stats in results["summary"].items():
        lines.append(
            f"{group:<12}{stats['n']:>4}" + "".join(f"{stats[m]:>14.3f}" for m in metric_names)
        )
    misses = [r for r in results["per_query"] if r[f"hit@{k}"] == 0.0]
    lines.append("")
    lines.append(f"Queries with no relevant memory in top {k} ({len(misses)}):")
    for r in misses:
        lines.append(f"  [{r['type']}] {r['query_id']} {r['query']!r} expected={r['relevant']}")
    return "\n".join(lines)


# ---------------------------------------------------------------- tests

@pytest.fixture(scope="module")
def dataset() -> dict[str, Any]:
    return load_dataset()


def test_metrics_on_known_ranking() -> None:
    retrieved = ["a", "b", "c", "d"]
    relevant = {"b", "d", "x"}
    assert hit_at_k(retrieved, relevant, 1) == 0.0
    assert hit_at_k(retrieved, relevant, 2) == 1.0
    assert recall_at_k(retrieved, relevant, 4) == pytest.approx(2 / 3)
    assert precision_at_k(retrieved, relevant, 4) == pytest.approx(0.5)
    assert reciprocal_rank(retrieved, relevant) == pytest.approx(0.5)
    assert reciprocal_rank(retrieved, {"z"}) == 0.0


def test_dataset_is_consistent(dataset: dict[str, Any]) -> None:
    memory_ids = [m["memory_id"] for m in dataset["memories"]]
    query_ids = [q["query_id"] for q in dataset["queries"]]
    assert len(memory_ids) == len(set(memory_ids)), "duplicate memory_id"
    assert len(query_ids) == len(set(query_ids)), "duplicate query_id"
    for q in dataset["queries"]:
        assert q["relevant_memory_ids"], f"{q['query_id']} has no relevant memories"
        assert set(q["relevant_memory_ids"]) <= set(memory_ids), f"{q['query_id']} unknown id"
        assert q["type"] in {"lexical", "paraphrase"}


def test_retrieval_evaluation_runs(dataset: dict[str, Any]) -> None:
    results = evaluate(dataset)
    print("\n" + format_report(results))

    assert len(results["per_query"]) == len(dataset["queries"])
    for stats in results["summary"].values():
        for name, value in stats.items():
            if name != "n":
                assert 0.0 <= value <= 1.0


if __name__ == "__main__":
    print(format_report(evaluate(load_dataset())))
