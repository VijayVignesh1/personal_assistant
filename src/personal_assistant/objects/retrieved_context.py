from dataclasses import dataclass


@dataclass
class RetrievedContext:
    id: str
    message_id: str
    memory_id: str | None
    retrieved_content: str
    retrieval_timestamp: str
    similarity_score: float | None = None
    retrieval_rank: int | None = None