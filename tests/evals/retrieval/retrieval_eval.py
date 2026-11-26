import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import mlflow

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
    ("hybrid w_0.3", "hybrid", 0.3),
    ("hybrid w_0.5", "hybrid", 0.5),
    ("hybrid w_0.7", "hybrid", 0.7),
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


async def build_memories(
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
            embedding=await embedding_model(item["content"]) if embedding_model else None,
            model_name=embedding_model.model_name if embedding_model else None,
        )
        for item in dataset["memories"]
    ]


async def evaluate(
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
        memories = await build_memories(dataset, embedding_model)
    retriever = Retriever(
        memories, embedding_model=embedding_model, method=method, bm25_weight=bm25_weight
    )

    per_query = []
    for q in dataset["queries"]:
        retrieved = [m.memory_id for m in await retriever.retrieve_relevant_memories(q["query"], top_n=k)]
        relevant = set(q["relevant_memory_ids"])
        per_query.append({
            "query_id": q["query_id"],
            "query": q["query"],
            "type": q["type"],
            "retrieved": retrieved,
            "relevant": sorted(relevant),
            f"hit_at_{k}": hit_at_k(retrieved, relevant, k),
            f"recall_at_{k}": recall_at_k(retrieved, relevant, k),
            f"precision_at_{k}": precision_at_k(retrieved, relevant, k),
            "mrr": reciprocal_rank(retrieved, relevant),
        })

    metric_names = [f"hit_at_{k}", f"recall_at_{k}", f"precision_at_{k}", "mrr"]

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
    metric_names = [f"hit_at_{k}", f"recall_at_{k}", f"precision_at_{k}", "mrr"]
    lines = [f"{'group':<12}{'n':>4}" + "".join(f"{m:>14}" for m in metric_names)]
    for group, stats in results["summary"].items():
        lines.append(
            f"{group:<12}{stats['n']:>4}" + "".join(f"{stats[m]:>14.3f}" for m in metric_names)
        )
    misses = [r for r in results["per_query"] if r[f"hit_at_{k}"] == 0.0]
    lines.append("")
    lines.append(f"Queries with no relevant memory in top {k} ({len(misses)}):")
    for r in misses:
        lines.append(f"  [{r['type']}] {r['query_id']} {r['query']!r} expected={r['relevant']}")
    return "\n".join(lines)


class WithoutQueryPrompt(BaseEmbeddingModel):
    """Wraps an embedding model so that queries are encoded like documents (no prompt)."""

    def __init__(self, model: BaseEmbeddingModel) -> None:
        super().__init__(model.model_name)  # same name, so stored embeddings are used
        self.model = model

    async def __call__(self, input: str) -> bytes:
        return await self.model(input)


async def compare_methods(
    dataset: dict[str, Any], embedding_model: BaseEmbeddingModel, k: int = TOP_K
) -> dict[str, dict[str, Any]]:
    """Evaluate every configuration in COMPARED_METHODS on the same embedded memories.

    If the model encodes queries with a prompt, the embedding and hybrid configurations
    are evaluated again without it. Memories are embedded once and reused.
    """
    memories = await build_memories(dataset, embedding_model)
    results = {
        label: await evaluate(dataset, k, memories, embedding_model, method, bm25_weight)
        for label, method, bm25_weight in COMPARED_METHODS
    }
    if getattr(embedding_model, "query_prompt_name", None):
        plain_queries = WithoutQueryPrompt(embedding_model)
        for label, method, bm25_weight in COMPARED_METHODS:
            if method != "bm25":
                results[f"{label} --no prompt--"] = await evaluate(
                    dataset, k, memories, plain_queries, method, bm25_weight
                )
    return results


def format_comparison(results_by_method: dict[str, dict[str, Any]], model_name: str) -> str:
    """One table of Hit@k, Recall@k and MRR per method and query type."""
    k = next(iter(results_by_method.values()))["k"]
    metric_names = [f"hit_at_{k}", f"recall_at_{k}", "mrr"]
    width = max(len("method"), *(len(label) for label in results_by_method)) + 2
    lines = [
        f"Embedding model: {model_name}",
        f"{'method':<{width}}{'group':<12}{'n':>4}" + "".join(f"{m:>12}" for m in metric_names),
    ]
    for label, results in results_by_method.items():
        for group, stats in results["summary"].items():
            lines.append(
                f"{label:<{width}}{group:<12}{stats['n']:>4}"
                + "".join(f"{stats[m]:>12.3f}" for m in metric_names)
            )
    return "\n".join(lines)


def log_comparison_results_to_mlflow(
    results: dict,  compared_methods: list[tuple[str, str, float]], experiment_name: str = "retrieval-evaluation"
    ) -> None:
    """
    Log evaluation results to MLflow, one run per retrieval method.
    """
    mlflow.set_tracking_uri("sqlite:///../mlflow.db")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="retrieval_evaluation"):

        for method, data in results.items():

            # Log summary metrics
            summary = data["summary"]

            for category, metrics in summary.items():

                for metric_name, value in metrics.items():

                    # Log metrics with unique names
                    mlflow.log_metric(
                        f"{method}/{category}/{metric_name}",
                        float(value)
                    )

            # Log per-query results as JSON
            artifact_path = Path(f"{method}_results.json")

            with open(artifact_path, "w") as f:
                json.dump(data["per_query"], f, indent=2)

            mlflow.log_artifact(str(artifact_path))

async def main() -> None:
    data = load_dataset()
    print(format_report(await evaluate(data)))
    model = embeddingModel(*sys.argv[1:2])

    # replace - by _ in model name for display purposes
    model.model_name = model.model_name.replace("-", "_")

    print()
    results_by_method = await compare_methods(data, model)

    log_comparison_results_to_mlflow(results_by_method, COMPARED_METHODS, experiment_name="retrieval-evaluation")
    print(format_comparison(results_by_method, model.model_name))


if __name__ == "__main__":
    asyncio.run(main())