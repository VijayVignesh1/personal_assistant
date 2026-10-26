"""Evaluation of the memory retriever on a small sample dataset.

The dataset (``dataset.json`` next to this file) contains synthetic memories and
queries with the ids of the memories that should be retrieved for each query.

Run the tests:
    uv run pytest tests/retrieval/test_retrieval_eval.py

Print the full evaluation report, including the BM25 / embedding / hybrid comparison
(optionally with another embedding model, e.g. Qwen/Qwen3-Embedding-0.6B):
    uv run python tests/retrieval/test_retrieval_eval.py [embedding_model_name]
"""

import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from personal_assistant.models.base import BaseEmbeddingModel
from personal_assistant.models.embeddingModel import embeddingModel
from personal_assistant.objects.memory import Memory
from personal_assistant.retriever.retrieval import RetrievalMethod, Retriever

DATASET_PATH = Path(__file__).resolve().parent / "dataset.json"
TOP_K = 5  # same value the application uses

# (label, method, bm25_weight) of every retriever configuration in the comparison
COMPARED_METHODS: list[tuple[str, RetrievalMethod, float]] = [
    ("bm25", "bm25", 0.5),
    ("embedding", "embedding", 0.5),
    ("hybrid w=0.3", "hybrid", 0.3),
    ("hybrid w=0.5", "hybrid", 0.5),
    ("hybrid w=0.7", "hybrid", 0.7),
]


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


def build_memories(
    dataset: dict[str, Any], embedding_model: BaseEmbeddingModel | None = None
) -> list[Memory]:
    """Build the dataset memories, embedded with ``embedding_model`` as MemoryBuilder does."""
    return [
        Memory(
            memory_id=item["memory_id"],
            content=item["content"],
            timestamp="2026-01-01T00:00:00Z",
            importance=1.0,
            episode_id="eval_episode",
            embedding=embedding_model(item["content"]) if embedding_model else None,
            model_name=embedding_model.model_name if embedding_model else None,
        )
        for item in dataset["memories"]
    ]


def evaluate(
    dataset: dict[str, Any],
    k: int = TOP_K,
    memories: list[Memory] | None = None,
    embedding_model: BaseEmbeddingModel | None = None,
    method: RetrievalMethod = "bm25",
    bm25_weight: float = 0.5,
) -> dict[str, Any]:
    """Run every query through the retriever and compute metrics.

    Returns per-query results and average metrics, overall and per query type.
    """
    if memories is None:
        memories = build_memories(dataset, embedding_model)
    retriever = Retriever(
        memories, embedding_model=embedding_model, method=method, bm25_weight=bm25_weight
    )

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


def compare_methods(
    dataset: dict[str, Any], embedding_model: BaseEmbeddingModel, k: int = TOP_K
) -> dict[str, dict[str, Any]]:
    """Evaluate every configuration in COMPARED_METHODS on the same embedded memories."""
    memories = build_memories(dataset, embedding_model)
    return {
        label: evaluate(dataset, k, memories, embedding_model, method, bm25_weight)
        for label, method, bm25_weight in COMPARED_METHODS
    }


def format_comparison(results_by_method: dict[str, dict[str, Any]], model_name: str) -> str:
    """One table of Hit@k, Recall@k and MRR per method and query type."""
    k = next(iter(results_by_method.values()))["k"]
    metric_names = [f"hit@{k}", f"recall@{k}", "mrr"]
    lines = [
        f"Embedding model: {model_name}",
        f"{'method':<14}{'group':<12}{'n':>4}" + "".join(f"{m:>12}" for m in metric_names),
    ]
    for label, results in results_by_method.items():
        for group, stats in results["summary"].items():
            lines.append(
                f"{label:<14}{group:<12}{stats['n']:>4}"
                + "".join(f"{stats[m]:>12.3f}" for m in metric_names)
            )
    return "\n".join(lines)


# ---------------------------------------------------------------- tests

@pytest.fixture(scope="module")
def dataset() -> dict[str, Any]:
    return load_dataset()


@pytest.fixture(scope="module")
def embedding_model() -> BaseEmbeddingModel:
    return embeddingModel()  # the small default model


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


def test_retrieval_method_comparison(
    dataset: dict[str, Any], embedding_model: BaseEmbeddingModel
) -> None:
    results_by_method = compare_methods(dataset, embedding_model)
    print("\n" + format_comparison(results_by_method, embedding_model.model_name))

    assert list(results_by_method) == [label for label, _, _ in COMPARED_METHODS]
    for results in results_by_method.values():
        assert len(results["per_query"]) == len(dataset["queries"])
        for stats in results["summary"].values():
            for name, value in stats.items():
                if name != "n":
                    assert 0.0 <= value <= 1.0


if __name__ == "__main__":
    data = load_dataset()
    print(format_report(evaluate(data)))
    model = embeddingModel(*sys.argv[1:2])
    print()
    print(format_comparison(compare_methods(data, model), model.model_name))
