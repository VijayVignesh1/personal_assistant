import json

import mlflow
from longmemeval_response_generator import OUTPUT_FILE
from mlflow.genai.scorers import (
    Correctness,
    Guidelines,
    RelevanceToQuery,
)

# 1. MLflow tracking
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("conversation-evaluation")

# 2. Local Ollama judge
judge_model = "ollama:/gemma3:4b"

with open(OUTPUT_FILE, "r") as f:
    records = json.load(f)

eval_data = []

for record in records:
    memories = [
        mem["content"]
        for mem in record["relevant_memories"]
    ]

    eval_data.append({
        "inputs": {
            "question": record["question"],
            "retrieved_memories": memories
        },
        "outputs": record["output"],
        "expectations": {
            "expected_response": record["expected_output"]
        }
    })

judge_model = "ollama:/gemma3:4b"

scorers = [
    Correctness(model=judge_model),
    RelevanceToQuery(model=judge_model),

    Guidelines(
        name="memory_groundedness",
        guidelines="""
        Evaluate whether the generated answer is supported
        by the retrieved memories.

        - The answer should use information from retrieved memories.
        - The answer should not invent personal facts.
        - If the memories do not contain enough information,
          the answer should acknowledge uncertainty.
        """,
        model=judge_model
    )
]

results = mlflow.genai.evaluate(
    data=eval_data,
    scorers=scorers
)

print("\nEvaluation Results:")
print(results.metrics)