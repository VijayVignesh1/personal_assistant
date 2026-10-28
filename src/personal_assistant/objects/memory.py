from dataclasses import dataclass


@dataclass
class Memory:
    content: str
    timestamp: str
    importance: float
    episode_id: str
    memory_id: str
    embedding: bytes | None = None
    model_name: str | None = None