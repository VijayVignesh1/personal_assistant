from typing import Any, Literal

import mlflow
from mlflow.genai.judges import make_judge

hallucination_judge = make_judge(
    name="session_hallucination",
    model="ollama:/gemma3:4b",
    instructions="""
    You are an expert evaluator of hallucinations in conversational AI.

    Evaluate the entire conversation:

    {{conversation}}

    Your task is to identify factual claims made by the assistant that
    are unsupported, fabricated, or contradicted by available evidence.

    Evaluation criteria:

    1. Identify factual claims made by the assistant.
    2. Check whether claims are supported by the available conversation
       context and evidence.
    3. Identify fabricated facts, incorrect details, and unsupported
       assumptions.
    4. Check whether the assistant contradicts information provided
       earlier by the user.
    5. Do not classify reasonable uncertainty or clearly stated opinions
       as hallucinations.
    6. Do not assume that a claim is true simply because it sounds
       plausible.
    7. If there is insufficient evidence to verify a claim, distinguish
       unverifiable information from demonstrably false information.

    Return one of these assessments:

    - no_hallucination: No significant unsupported or fabricated claims.
    - minor_hallucination: Minor unsupported claims that do not
      substantially affect the conversation.
    - major_hallucination: Significant fabricated, misleading, or
      factually incorrect claims.
    """,
    feedback_value_type=Literal[
        "no_hallucination",
        "minor_hallucination",
        "major_hallucination"
    ],
)


def get_session_traces(episode_id: str) -> Any:

    experiment = mlflow.get_experiment_by_name(
        "personal_assistant"
    )

    traces = mlflow.search_traces(
        experiment_ids=[experiment.experiment_id],  # type: ignore[union-attr]
        filter_string=(
            f"metadata.`mlflow.trace.session` = '{episode_id}'"
        ),
        return_type="list",
    )

    return traces

def evaluate_hallucination(episode_id: str) -> Any:

    traces = get_session_traces(episode_id)

    if not traces:
        raise ValueError(
            f"No MLflow traces found for episode {episode_id}"
        )

    results = mlflow.genai.evaluate(
        data=traces,
        scorers=[hallucination_judge],
    )

    return results